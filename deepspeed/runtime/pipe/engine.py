# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import MethodType
from collections import OrderedDict
from functools import reduce
from operator import mul

import torch
from deepspeed import comm as dist

from deepspeed.utils import logger
from deepspeed.utils.timer import ThroughputTimer
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.bf16_optimizer import BF16_Optimizer

from ..engine import DeepSpeedEngine, MEMORY_OPT_ALLREDUCE_SIZE
from deepspeed.utils.timer import FORWARD_MICRO_TIMER, FORWARD_GLOBAL_TIMER, BACKWARD_MICRO_TIMER, \
    BACKWARD_GLOBAL_TIMER, BACKWARD_INNER_MICRO_TIMER, BACKWARD_INNER_GLOBAL_TIMER, \
    BACKWARD_REDUCE_MICRO_TIMER, BACKWARD_REDUCE_GLOBAL_TIMER, \
    STEP_MICRO_TIMER, STEP_GLOBAL_TIMER

from ..utils import PartitionedTensor
from ..dataloader import RepeatingLoader
from ..zero.config import ZeroStageEnum
from ..activation_checkpointing import checkpointing as ds_checkpointing

from .module import PipelineModule, PipelineError
from . import p2p
from . import schedule

TARGET_ID = -2
LOG_STAGE = -2
DATA_PARALLEL_ID = -2

BATCH_INPUT_TIMER = 'batch_input'
TRAIN_BATCH_TIMER = 'train_batch'
PIPE_SEND_OUTPUT_TIMER = 'pipe_send_output'
PIPE_SEND_GRAD_TIMER = 'pipe_send_grad'
PIPE_RECV_INPUT_TIMER = 'pipe_recv_input'
PIPE_RECV_GRAD_TIMER = 'pipe_recv_grad'

# The buffer size to store the meta data for each tensor.
TENSOR_META_SIZE = 256


def is_even(number):
    return number % 2 == 0


mem_alloced = 0
mem_cached = 0


def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


class PipelineEngine(DeepSpeedEngine):
    """ A training engine hybrid pipeline, data, and model parallel training.

    This engine is created by ``deepspeed.initialize()`` when a :class:`PipelineModule`
    is provided.
    """
    ID_TO_DTYPE = [
        torch.float32, torch.float64, torch.complex64, torch.complex128, torch.float16, torch.bfloat16, torch.uint8,
        torch.int8, torch.int16, torch.int32, torch.int64, torch.bool
    ]
    DTYPE_TO_ID = {dtype: id_ for id_, dtype in enumerate(ID_TO_DTYPE)}

    def __init__(self, has_bool_tensors=False, *super_args, **super_kwargs):
        super().__init__(*super_args, **super_kwargs)
        assert isinstance(self.module, PipelineModule), "model must base PipelineModule"

        assert self.zero_optimization_stage(
        ) < ZeroStageEnum.gradients, "ZeRO-2 and ZeRO-3 are incompatible with pipeline parallelism"

        # We schedule the all-reduces, so disable it in super().backward()
        self.enable_backward_allreduce = False
        self.has_bool_tensors = has_bool_tensors
        self.eval_return_logits = False
        self.outputs = None
        # BF16 Optimizer is hardcoded for fp32 gradient accumulation
        self.using_bf16_optimizer = type(self.optimizer) == BF16_Optimizer

        # used to disable the pipeline all-reduce when used with 1-bit Adam/1-bit LAMB
        self.pipeline_enable_backward_allreduce = True

        if self.elasticity_enabled():
            if not self.is_elastic_model_parallel_supported():
                assert not self.elasticity_enabled(), "Elasticity is not currently supported" \
                " with pipeline parallelism."

        # pipeline step for logging
        self.log_batch_step_id = -1

        self.micro_batch_size = self.train_micro_batch_size_per_gpu()
        self.micro_batches = self.gradient_accumulation_steps()

        # Set Grid and Communication Groups
        self.grid = self.module._grid
        if self.grid.get_global_rank() == 0:
            logger.info(f'CONFIG: micro_batches={self.micro_batches} '
                        f'micro_batch_size={self.micro_batch_size}')

        self.global_rank = self.grid.get_global_rank()

        assert self.dp_world_size == self.grid.data_parallel_size
        assert self.train_batch_size() == \
            self.micro_batch_size * self.micro_batches * self.grid.data_parallel_size

        #  Set Stage Inf
        self.num_stages = self.grid.pipe_parallel_size
        self.stage_id = self.grid.get_stage_id()
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1

        self.data_iterator = None
        self.batch_fn = None

        self._force_grad_boundary = False

        self.batch_timer = ThroughputTimer(self._config.timers_config,
                                           batch_size=self.train_batch_size(),
                                           logging_fn=self.tput_log,
                                           monitor_memory=False,
                                           steps_per_output=self.steps_per_print())

        # PipelineEngine needs to handle data loading specially due to only the first
        # and last stages loading inputs/labels. We construct a sampler that uses
        if self.training_data:
            self._build_data_iter(self.training_data)

        self.is_pipe_parallel = self.grid.pipe_parallel_size > 1
        self.is_data_parallel = self.grid.data_parallel_size > 1
        self.is_model_parallel = self.grid.model_parallel_size > 1

        # Partition input/output buffers
        # XXX temporarily disable while I revert some partition hacks.
        assert isinstance(self._config.pipeline['pipe_partitioned'], bool)
        assert isinstance(self._config.pipeline['grad_partitioned'], bool)
        self.is_pipe_partitioned = self.is_model_parallel and self._config.pipeline['pipe_partitioned']
        self.is_grad_partitioned = self.is_model_parallel and self._config.pipeline['grad_partitioned']
        logger.info(f'is_pipe_partitioned= {self.is_pipe_partitioned} '
                    f'is_grad_partitioned= {self.is_grad_partitioned}')

        model_parameters = filter(lambda p: p.requires_grad, self.module.parameters())
        num_params = sum([p.numel() for p in model_parameters])
        unique_params = num_params
        # Subtract tied parameters if we don't own them
        if self.module.tied_comms:
            tied_params = 0
            for key, d in self.module.tied_comms.items():
                if self.global_rank != min(d['ranks']):
                    tied_params += sum(p.numel() for p in d['module'].parameters())
            unique_params -= tied_params
        params_tensor = torch.LongTensor(data=[num_params, unique_params]).to(self.device)
        dist.all_reduce(params_tensor, group=self.grid.get_model_parallel_group())
        params_tensor = params_tensor.tolist()
        total_params = params_tensor[0]
        unique_params = params_tensor[1]
        if self.grid.data_parallel_id == 0:
            logger.info(f'RANK={self.global_rank} '
                        f'STAGE={self.stage_id} '
                        f'LAYERS={self.module._local_stop - self.module._local_start} '
                        f'[{self.module._local_start}, {self.module._local_stop}) '
                        f'STAGE_PARAMS={num_params} ({num_params/1e6:0.3f}M) '
                        f'TOTAL_PARAMS={total_params} ({total_params/1e6:0.3f}M) '
                        f'UNIQUE_PARAMS={unique_params} ({unique_params/1e6:0.3f}M)')

        #initialize peer-2-peer communication and allreduce groups
        if self.is_pipe_parallel:
            p2p.init_process_groups(self.grid)

        # Pipeline buffers
        self.num_pipe_buffers = 0
        self.pipe_buffers = {
            'inputs': [],  # batch input and received activations
            'labels': [],  # labels from batch input
            'outputs': [],  # activations
            'output_tensors': [],  # tensor object to preserve backward graph
        }
        self.pipe_recv_buf = None
        self.grad_layer = None
        self._grad_layer_buf = []

        self.meta_buffer = None

        self.first_output_send = True
        self.first_gradient_send = True
        self.pipe_partition_input_meta_cache = None
        self.pipe_partition_output_meta_cache = None
        self.pipe_partition_grad_meta_cache = None
        self.grad_partition_grad_layer_meta_cache = None

        #stores the loss for the current micro batch being processed
        self.loss = torch.tensor(0.0).to(self.device)

        #stores the loss for the entire batch
        self.total_loss = None
        self.total_additional_losses = None
        self.agg_loss = torch.tensor(0.0, requires_grad=False).to(self.device)
        self.dp_group_loss = torch.tensor(0.0, requires_grad=False).to(self.device)

        # stores aggregated-DP train final loss and aggregated-DP additional losses, if any
        # additional losses are stored as dict: {loss-name: agg-loss}
        self.agg_train_loss = None
        self.agg_additional_losses = None

        if self._config.pipeline['activation_checkpoint_interval'] > 0:
            self.module.activation_checkpoint_interval = self._config.pipeline['activation_checkpoint_interval']
            # set use_reentrant default to True.
            if self._config.pipeline.get('use_reentrant') is None:
                self._config.pipeline['use_reentrant'] = True
            if self._config.pipeline['use_reentrant'] is False:
                # set activation_checkpoint_func to non_reentrant_checkpoint func.
                self.module.activation_checkpoint_func = ds_checkpointing.non_reentrant_checkpoint
                if self.grid.get_global_rank() == 0:
                    logger.info(f'CONFIG: activation_checkpoint_func=non_reentrant_checkpoint')
        if self.module.activation_checkpoint_interval > 0:
            self.module._precompute_checkpointable_values()

        self.module.checkpoint_parallel_write_pipeline = self._config.checkpoint_parallel_write_pipeline

        if self.is_last_stage():
            self.loss_model = self.module.loss_fn

        self.has_attention_mask = self.module.__class__.__name__ == 'GPT2ModelPipe'
        # Initialize pipeline communicators. Just send a 0.
        if is_even(self.stage_id):
            if not self.is_last_stage():
                p2p.send(self.loss, self.next_stage)
            if not self.is_first_stage():
                p2p.recv(self.loss, self.prev_stage)
        else:
            if not self.is_first_stage():
                p2p.recv(self.loss, self.prev_stage)
            if not self.is_last_stage():
                p2p.send(self.loss, self.next_stage)

        # XXX look into timer reporting timing
        # Initialize some timers because of early weirdness.
        if self.wall_clock_breakdown():
            self.timers(FORWARD_MICRO_TIMER).start()
            self.timers(FORWARD_MICRO_TIMER).stop()
            self.timers(BACKWARD_MICRO_TIMER).start()
            self.timers(BACKWARD_MICRO_TIMER).stop()
            self.timers(BACKWARD_INNER_MICRO_TIMER).start()
            self.timers(BACKWARD_INNER_MICRO_TIMER).stop()
            self.timers(BACKWARD_REDUCE_MICRO_TIMER).start()
            self.timers(BACKWARD_REDUCE_MICRO_TIMER).stop()
            self.timers(BACKWARD_REDUCE_GLOBAL_TIMER).start()
            self.timers(BACKWARD_REDUCE_GLOBAL_TIMER).stop()
            self.timers(STEP_MICRO_TIMER).start()
            self.timers(STEP_MICRO_TIMER).stop()

        self.dynamic_shape = self.module.dynamic_shape

    def set_has_attention_mask(self, value):
        assert isinstance(value, bool)
        self.has_attention_mask = value

    def _build_data_iter(self, dataset):
        sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                  num_replicas=self.dp_world_size,
                                                                  rank=self.mpu.get_data_parallel_rank(),
                                                                  shuffle=False)
        # Build a loader and make it repeating.
        pipe_dataloader = self.deepspeed_io(dataset, data_sampler=sampler)
        pipe_dataloader = RepeatingLoader(pipe_dataloader)
        self.set_dataloader(pipe_dataloader)

    def _exec_reduce_tied_grads(self):
        # We need to run this first to write to self.averaged_gradients;
        # since this class turns `enable_backward_allreduce` off,
        # `self.overlapping_partition_gradients_reduce_epilogue()` defined in the DeepSpeedEngine
        # never actually runs. I suspect this is because of efficiency problems; get_flat_partition in
        # stage2.py might do something expensive; someone will have to look into that later. But
        # in the meantime, this fixes ZeRO2 + Pipelining enough to run a demo. Further profiling
        # needed to decide if it actually breaks everything.
        # (see https://github.com/EleutherAI/gpt-neox/issues/62#issuecomment-761471944)
        if self.zero_optimization_partition_gradients():
            self.optimizer.overlapping_partition_gradients_reduce_epilogue()

        weight_group_list = self.module.get_tied_weights_and_groups()
        for weight, group in weight_group_list:
            grad = weight._hp_grad if self.using_bf16_optimizer else weight.grad
            if grad is not None:
                dist.all_reduce(grad, group=group)

    def _exec_reduce_grads(self):
        self._force_grad_boundary = True
        if self.pipeline_enable_backward_allreduce:
            if self.using_bf16_optimizer:
                # PP+BF16 work for ZeRO Stage 1
                self._bf16_reduce_grads()
            else:
                self.allreduce_gradients(bucket_size=MEMORY_OPT_ALLREDUCE_SIZE)
        self._force_grad_boundary = False

    def _bf16_reduce_grads(self):
        self.buffered_allreduce_fallback(grads=None, elements_per_buffer=MEMORY_OPT_ALLREDUCE_SIZE)

    def _reserve_pipe_buffers(self, num_buffers):
        """Ensure that each pipeline buffer has at least ``num_buffers`` slots.

        This method only reserves slots and does not allocate tensors.

        Args:
            num_buffers (int): The number of buffers to reserve.
        """
        if self.num_pipe_buffers >= num_buffers:
            return

        num_added = num_buffers - self.num_pipe_buffers
        for key in self.pipe_buffers:
            self.pipe_buffers[key].extend([None] * num_added)
        self.num_pipe_buffers = num_buffers

    def reset_activation_shape(self):
        """Reset the buffers when the shape of activation and gradient change.
        For example, for curriculum learning that changes the seqlen of each
        sample, we need to call this whenever the seqlen is going to change.
        """
        self.first_output_send = True
        self.pipe_recv_buf = None
        self.grad_layer = None
        self._grad_layer_buf = []
        self.meta_buffer = None

        self.pipe_partition_input_meta_cache = None
        self.pipe_partition_output_meta_cache = None
        self.pipe_partition_grad_meta_cache = None
        self.grad_partition_grad_layer_meta_cache = None

    def train_batch(self, data_iter=None):
        """Progress the pipeline to train the next batch of data. The engine will ingest
        ``self.train_batch_size()`` total samples collectively across all workers.


        An iterator that over training data should be provided as an argument
        unless ``deepspeed.initialize()`` was provided a training set. In that event,
        the training data will automatically be read.


        .. warning::
            A total of ``self.gradient_accumulation_steps()`` entries will be pulled
            from ``data_iter`` by each pipeline. There must be sufficient
            data left in ``data_iter`` or else a ``StopIteration`` will halt training.

            DeepSpeed provides a convenience class :class:`deepspeed.utils.RepeatingLoader`
            that wraps data loaders to automatically restart upon a ``StopIteration``.

        Args:
            data_iter (Iterator, optional): Iterator of training data.

        Returns:
            The arithmetic mean of the losses computed this batch.
        """
        if not torch._C.is_grad_enabled():
            raise RuntimeError(f'train_batch() requires gradients enabled. Use eval_batch() instead.')

        # Curriculum learning could change activation shape
        if self.curriculum_enabled_legacy():
            new_difficulty = self.curriculum_scheduler_legacy.update_difficulty( \
                self.global_steps + 1)
            if self.global_steps == 0 or self.curriculum_scheduler_legacy.first_step:
                self.reset_activation_shape()
                self.curriculum_scheduler_legacy.first_step = False
            elif new_difficulty != self.curriculum_scheduler_legacy.get_difficulty( \
                self.global_steps):
                self.reset_activation_shape()

        if data_iter is not None:
            self.set_dataiterator(data_iter)

        self.module.train()
        self.total_loss = None
        self.total_additional_losses = None
        self._compute_loss = True

        # Do the work
        self.timers(TRAIN_BATCH_TIMER).start()
        sched = schedule.TrainSchedule(micro_batches=self.micro_batches,
                                       stages=self.num_stages,
                                       stage_id=self.stage_id)
        self._exec_schedule(sched)

        with torch.no_grad():
            self.agg_train_loss = self._aggregate_total_loss()

        self.timers(TRAIN_BATCH_TIMER).stop()

        if self.global_steps % self.steps_per_print() == 0:
            if self.global_rank == 0:
                elapsed = self.timers(TRAIN_BATCH_TIMER).elapsed(reset=True) / 1000.0
                iter_time = elapsed / self.steps_per_print()
                tput = self.train_batch_size() / iter_time
                log_str = f'steps: {self.global_steps} loss: {self.agg_train_loss:0.4f} '
                if self.agg_additional_losses is not None:
                    for loss_name, loss_value in self.agg_additional_losses.items():
                        log_str += f'{loss_name}: {loss_value.item():0.4f} '
                log_str += f'iter time (s): {iter_time:0.3f} samples/sec: {tput:0.3f}'
                print(log_str)
            else:
                self.timers(TRAIN_BATCH_TIMER).elapsed(reset=True)

        # Monitoring
        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [(f'Train/Samples/train_loss', self.agg_train_loss.mean().item(),
                                    self.global_samples)]
            self.monitor.write_events(self.summary_events)

        if self.wall_clock_breakdown() and self.global_steps % self.steps_per_print() == 0:
            self.timers.log([
                PIPE_SEND_OUTPUT_TIMER,
                PIPE_SEND_GRAD_TIMER,
                PIPE_RECV_INPUT_TIMER,
                PIPE_RECV_GRAD_TIMER,
            ])

        # TODO: should return precisely what loss returned and allow others to be queried?
        return self.agg_train_loss

    def eval_batch(self,
                   data_iter,
                   return_logits=False,
                   compute_loss=True,
                   reduce_output='avg',
                   bcast_loss=True,
                   num_micro_batches=None):
        """Evaluate the pipeline on a batch of data from ``data_iter``. The
        engine will evaluate ``self.train_batch_size()`` total samples
        collectively across all workers.

        This method is equivalent to:

        .. code-block:: python

            module.eval()
            with torch.no_grad():
                output = module(batch)

        .. warning::
            A total of ``self.gradient_accumulation_steps()`` entries will be pulled
            from ``data_iter`` by each pipeline. There must be sufficient
            data left in ``data_iter`` or else a ``StopIteration`` will halt training.

            DeepSpeed provides a convenience class :class:`deepspeed.utils.RepeatingLoader`
            that wraps data loaders to automatically restart upon a ``StopIteration``.

        Args:
            data_iter (Iterator): Iterator of data to evaluate.

        Returns:
            The arithmetic mean of the losses computed this batch.
        """
        self.eval_return_logits = return_logits
        self.module.eval()

        # Curriculum learning could change activation shape
        if self.curriculum_enabled_legacy():
            new_difficulty = self.curriculum_scheduler_legacy.update_difficulty( \
                self.global_steps + 1)
            if self.global_steps == 0 or self.curriculum_scheduler_legacy.first_step:
                self.reset_activation_shape()
                self.curriculum_scheduler_legacy.first_step = False
            elif new_difficulty != self.curriculum_scheduler_legacy.get_difficulty( \
                self.global_steps):
                self.reset_activation_shape()

        eval_output = None

        self._compute_loss = compute_loss

        # Use the provided data iterator
        train_iterator = self.data_iterator
        self.set_dataiterator(data_iter)

        # set the number micro batches in case the user chose value than training
        micro_batches = self.micro_batches if num_micro_batches is None else num_micro_batches

        # Do the work
        sched = schedule.InferenceSchedule(micro_batches=micro_batches, stages=self.num_stages, stage_id=self.stage_id)

        # prevent dead-lock with multiple evals sequence
        dist.barrier()

        with torch.no_grad():
            self._exec_schedule(sched)

        if self.is_last_stage():
            eval_output = self._reduce_outputs(self.fwd_outputs, reduce=reduce_output, micro_batches=micro_batches)

        if compute_loss and (bcast_loss or self.monitor.enabled):
            eval_output = self._bcast_pipe_scalar(eval_output)

        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [(f'Train/Samples/eval_loss', eval_output.mean().item(), self.global_samples)]
            self.monitor.write_events(self.summary_events)

        # Restore the training iterator
        self.set_dataiterator(train_iterator)

        # Reset any buffers that may have been populated during the forward passes.
        #ds_checkpointing.reset()
        self.eval_return_logits = False
        if return_logits:
            outputs = self.outputs
            self.outputs = None
            return eval_output, outputs
        return eval_output

    def set_train_batch_size(self, train_batch_size):
        """Adjust the global batch size by increasing or decreasing the number of
        micro-batches (i.e., gradient accumulation steps). The size of each micro-batch
        (i.e., ``train_micro_batch_size_per_gpu``) is not changed.
        Args:
            train_batch_size (int): The new global batch size for training.
        Raises:
            ValueError: if ``train_batch_size`` is not divisible by the
                configured micro-batch size and data parallelism.
        """
        super().set_train_batch_size(train_batch_size)
        self.micro_batches = self.gradient_accumulation_steps()

    def is_first_stage(self):
        """True if this process is in the first stage in the pipeline."""
        return self.stage_id == 0

    def is_last_stage(self):
        """True if this process is in the last stage in the pipeline."""
        return self.stage_id == self.num_stages - 1

    def _reduce_outputs(self, outputs, reduce='avg', reduce_dp=True, micro_batches=None):
        if reduce is None:
            return outputs

        if reduce.lower() == 'avg':
            # first sum over all microbatches
            if torch.is_tensor(outputs[0]):
                reduced = sum(outputs)
            else:
                assert isinstance(outputs, (list, tuple))
                reduced = [torch.zeros_like(o) for o in outputs[0]]
                for idx, out in outputs:
                    reduced[idx] += out

            # Average over the microbatches
            reduced = self._scale_loss_by_gas(reduced, eval_micro_batches=micro_batches)

            # Average over DP groups
            if reduce_dp and self.is_data_parallel:
                if torch.is_tensor(reduced):
                    dist.all_reduce(reduced, group=self.mpu.get_data_parallel_group())
                    reduced /= self.dp_world_size
                else:
                    for idx in range(len(reduced)):
                        dist.all_reduce(reduced[idx], group=self.mpu.get_data_parallel_group())
                        reduced[idx] /= self.dp_world_size

            return reduced
        else:
            raise NotImplementedError(f'reduction type {reduce} not supported.')

    def _bcast_pipe_scalar(self, data, src_rank=None, dtype=torch.float32):
        # Default to last stage (e.g., for broadcasting loss)
        if src_rank is None:
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
        assert src_rank in self.grid.pp_group

        if self.global_rank == src_rank:
            result = data.clone().detach().type(dtype).to(self.device)
        else:
            result = torch.Tensor([0.]).type(dtype).to(self.device)

        dist.broadcast(tensor=result, src=src_rank, group=self.mpu.get_pipe_parallel_group())

        return result

    def _aggregate_total_loss(self):
        # Scale loss, average among DP ranks, and bcast loss to the rest of my DP group
        if self.is_last_stage():
            # Scale loss and additional losses, if any
            loss = self._scale_loss_by_gas(self.total_loss)
            self.agg_additional_losses = self.total_additional_losses
            if self.agg_additional_losses is not None:
                self.agg_additional_losses = OrderedDict({
                    loss_name: self._scale_loss_by_gas(_loss.clone().detach())
                    for loss_name, _loss in self.agg_additional_losses.items()
                })

            self.dp_group_loss = loss.clone().detach()
            agg_loss = self.dp_group_loss.clone().detach()
            #print(f'RANK={self.global_rank} bcast SENDER src={self.global_rank} group={self.grid.pp_group}', flush=True)

            # Average loss across all data-parallel groups
            if self.is_data_parallel:
                if self.agg_additional_losses is None:
                    dist.all_reduce(agg_loss, group=self.mpu.get_data_parallel_group())
                    agg_loss /= self.dp_world_size
                else:
                    # use a single reduce op for agg_loss and additional losses, if any
                    assert '__train_loss__' not in self.agg_additional_losses.keys()
                    tensors = OrderedDict({'__train_loss__': agg_loss})
                    tensors.update(self.agg_additional_losses.items())
                    flat_tensor = torch.cat([t.clone().reshape(-1).detach() for t in tensors.values()])
                    dist.all_reduce(flat_tensor, group=self.mpu.get_data_parallel_group())
                    flat_tensor /= self.dp_world_size
                    offset = 0
                    reduced_tensor = {}
                    for name, t in tensors.items():
                        n_elem = t.numel()
                        reduced_tensor[name] = flat_tensor[offset:offset + n_elem].clone().detach().reshape(t.shape)
                        offset += n_elem
                    agg_loss = reduced_tensor['__train_loss__']
                    self.agg_additional_losses = OrderedDict(
                        {name: reduced_tensor[name]
                         for name in self.agg_additional_losses.keys()})

            assert self.global_rank in self.grid.pp_group
            losses = [self.dp_group_loss, agg_loss]
            if self.agg_additional_losses is not None:
                losses += list(self.agg_additional_losses.values())
            losses = torch.stack(losses).float()
            if self.is_pipe_parallel:
                dist.broadcast(tensor=losses, src=self.global_rank, group=self.mpu.get_pipe_parallel_group())
        else:
            # Get loss from last stage
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
            assert src_rank in self.grid.pp_group
            # losses to reduce are: dp_group_loss, agg_loss, model additional losses
            # therefore: 2 + n_additional_losses
            additional_losses = self.module.get_additional_losses()
            n_additional_losses = 0 if additional_losses is None else len(additional_losses)
            losses = torch.Tensor([0.] * (2 + n_additional_losses)).to(self.device)
            dist.broadcast(tensor=losses, src=src_rank, group=self.grid.get_pipe_parallel_group())
            self.dp_group_loss = losses[0].clone().detach()
            agg_loss = losses[1].clone().detach()
            if additional_losses is not None:
                self.agg_additional_losses = OrderedDict({
                    name: losses[2 + i].clone().detach()
                    for i, name in enumerate(additional_losses.keys())
                })
        return agg_loss

    def set_dataloader(self, loader):
        """"""
        if self.is_first_stage() or self.is_last_stage():
            self.training_dataloader = loader
            self.data_iterator = iter(self.training_dataloader)

    def set_dataiterator(self, iterator):
        """ Store an iterator to sample for training data. """
        if self.is_first_stage() or self.is_last_stage():
            self.training_dataloader = None
            self.data_iterator = iterator

    def set_batch_fn(self, fn):
        """Execute a post-processing function on input data.

        Args:
            fn (function): The function to run.
        """
        self.batch_fn = fn

    def is_gradient_accumulation_boundary(self):
        """True if the engine is executing a gradient reduction or optimizer step instruction.

        This is overridden from :class:`DeepSpeedEngine` to force reductions
        and steps when the pipeline engine is instructed to do so.

        Returns:
            bool: whether reductions and optimizer steps should occur.
        """
        return self._force_grad_boundary

    def log_for_device(self, *msg):
        if LOG_STAGE == self.stage_id or LOG_STAGE == -1:
            if DATA_PARALLEL_ID == self.grid.data_parallel_id or DATA_PARALLEL_ID == -1:
                print(
                    f'RANK={dist.get_rank()} '
                    f'PIPE-ID={self.stage_id} '
                    f'DATA-ID={self.grid.data_parallel_id} '
                    f'MBATCH-ID={self.microbatch_id} '
                    f'STEP-ID={self.log_batch_step_id} '
                    '::',
                    *msg,
                    flush=True)

    def tput_log(self, *msg):
        if self.global_rank == 0 and self.global_steps % self.steps_per_print() == 0:
            print(*msg)

    def _next_batch(self):
        # If using 3D parallelism, only some first-stage ranks may do IO
        batch = None
        if self.data_iterator is not None:
            batch = next(self.data_iterator)

        # Any post-processing, like broadcasting across a slice-parallel group.
        if self.batch_fn:
            batch = self.batch_fn(batch)

        return batch

    def _exec_forward_pass(self, buffer_id):
        self.tput_timer.start()
        self.mem_status('BEFORE FWD', reset_max=True)

        if isinstance(self.pipe_buffers['inputs'][buffer_id], tuple):
            inputs = tuple(t.clone() for t in self.pipe_buffers['inputs'][buffer_id])
        else:
            inputs = self.pipe_buffers['inputs'][buffer_id].clone()

        # collect the partitioned input from the previous stage
        if self.is_pipe_partitioned and not self.is_first_stage():
            if self.pipe_partition_input_meta_cache is None:
                self.pipe_partition_input_meta_cache = inputs[0].to('cpu')
            part_input = PartitionedTensor.from_meta(meta=self.pipe_partition_input_meta_cache,
                                                     local_part=inputs[1],
                                                     group=self.grid.get_slice_parallel_group())

            inputs = (part_input.full(), *inputs[2:])
            inputs[0].requires_grad = True
            # skip mask
            #inputs[1].requires_grad = True
            part_input = None
            inputs = inputs[0] if len(inputs) == 1 else inputs
            self.pipe_buffers['inputs'][buffer_id] = inputs

        # inputs has no gradient because it is from a cloned tensor
        outputs = super().forward(inputs)

        # Reset activation checkpointing buffers.
        # Need to call this between evaluation iterations
        if not self.module.training:
            ds_checkpointing.reset()

        # Partition the outputs if we are not the last stage
        if self.is_pipe_partitioned and not self.is_last_stage():
            if isinstance(outputs, tuple):
                first_output = outputs[0]
                # TODO: Improve pipe partitioning to pass multiple tensors that require grads
                assert all([torch.is_tensor(elt) and elt.requires_grad is False for elt in outputs[1:]])
                outputs_tail = outputs[1:]
            elif torch.is_tensor(outputs):
                first_output = outputs
                outputs_tail = []
            else:
                raise ValueError("expecting a tensor or a tuple of tensors")
            part = PartitionedTensor(tensor=first_output, group=self.grid.get_slice_parallel_group())
            # Clear the large output data, but save the computation graph
            first_output.data = torch.zeros(1, device=first_output.data.device)
            self.pipe_buffers['output_tensors'][buffer_id] = first_output
            # Inject the partitioned tensor into the output before sending
            outputs = (part.to_meta(), part.data(), *outputs_tail)
            part = None

        self.pipe_buffers['outputs'][buffer_id] = outputs

        # Optionally compute loss on the last device
        if self.is_last_stage():
            if self._compute_loss and self.module.loss_fn is not None:
                labels = self.pipe_buffers['labels'][buffer_id]
                self.loss = self.module.loss_fn(outputs, labels)
            else:
                # Some models just return loss from forward()
                self.loss = outputs
            if self.eval_return_logits:
                self.outputs = outputs

            if isinstance(self.loss, torch.Tensor):
                self.fwd_outputs.append(self.loss.detach())
            else:
                self.fwd_outputs.append([l.detach() for l in self.loss])

            def add_to_total_loss(_total_loss, _loss):
                if isinstance(_loss, torch.Tensor):
                    if _total_loss is None:
                        _total_loss = torch.zeros_like(_loss)
                    _total_loss += _loss.detach()
                else:
                    if _total_loss is None:
                        _total_loss = [torch.zeros_like(_l) for _l in _loss]
                    for _idx, _l in enumerate(_loss):
                        _total_loss[_idx] += _l.detach()
                return _total_loss

            self.total_loss = add_to_total_loss(self.total_loss, self.loss)

            # aggregate additional losses across gradient accumulation steps
            additional_losses = self.module.get_additional_losses()
            if additional_losses is not None:
                if self.total_additional_losses is None:
                    self.total_additional_losses = OrderedDict()
                for name, loss in additional_losses.items():
                    total = self.total_additional_losses[name] if name in self.total_additional_losses else None
                    self.total_additional_losses[name] = add_to_total_loss(total, loss)

    def _exec_backward_pass(self, buffer_id):
        assert self.optimizer is not None, "must provide optimizer during " \
                                           "init in order to use backward"

        self.mem_status('BEFORE BWD', reset_max=True)

        # The last stage just runs backward on the loss using DeepSpeed's typical
        # mechanisms.
        if self.is_last_stage():
            super().backward(self.loss)
            self.mem_status('AFTER BWD')
            return

        outputs = self.pipe_buffers['outputs'][buffer_id]

        if self.wall_clock_breakdown():
            self.timers(BACKWARD_MICRO_TIMER).start()
            self.timers(BACKWARD_GLOBAL_TIMER).start()
            self.timers(BACKWARD_INNER_MICRO_TIMER).start()
            self.timers(BACKWARD_INNER_GLOBAL_TIMER).start()

        # Reconstruct if we previously partitioned the output. We must be
        # careful to also restore the computational graph of the tensors we partitioned.
        if self.is_pipe_partitioned:
            if self.is_grad_partitioned:
                if self.pipe_partition_output_meta_cache is None:
                    self.pipe_partition_output_meta_cache = outputs[0].to('cpu')
                part_output = PartitionedTensor.from_meta(meta=self.pipe_partition_output_meta_cache,
                                                          local_part=outputs[1],
                                                          group=self.grid.get_slice_parallel_group())
                self.pipe_buffers['output_tensors'][buffer_id].data = part_output.full()
                outputs = (self.pipe_buffers['output_tensors'][buffer_id], *outputs[2:])
            else:
                # Already restored from partition
                self.pipe_buffers['output_tensors'][buffer_id].data = outputs[0]
                outputs = (self.pipe_buffers['output_tensors'][buffer_id], *outputs[1:])

        grad_tensors = self.grad_layer
        if self.is_grad_partitioned:
            #print(f'RANK={self.global_rank} BEFORE-BWD restoring grad={self.grad_layer[0].size()} {self.grad_layer[1].size()}')
            if self.grad_partition_grad_layer_meta_cache is None:
                self.grad_partition_grad_layer_meta_cache = self.grad_layer[0].to('cpu')
            part_grad = PartitionedTensor.from_meta(meta=self.grad_partition_grad_layer_meta_cache,
                                                    local_part=self.grad_layer[1],
                                                    group=self.grid.get_slice_parallel_group())
            grad_tensors = (part_grad.full(), *grad_tensors[2:])
            part_grad = None
            #print(f'RANK={self.global_rank} BEFORE-BWD restored grad={self.grad_layer[0].size()} {self.grad_layer[1].size()}')

        if self.using_bf16_optimizer and not self.is_last_stage():
            # manually call because we don't call optimizer.backward()
            self.optimizer.clear_lp_grads()

        # This handles either a single tensor or tuple of tensors.
        if isinstance(outputs, tuple):
            out_tensors = [t for t in outputs if t.is_floating_point()]
            assert len(out_tensors) == len(grad_tensors)
            torch.autograd.backward(tensors=out_tensors, grad_tensors=grad_tensors)
        else:
            torch.autograd.backward(tensors=(outputs, ), grad_tensors=(grad_tensors, ))

        if self.using_bf16_optimizer and not self.is_last_stage():
            # manually call because we don't call optimizer.backward()
            if not self._config.bfloat16_config.immediate_grad_update:
                self.optimizer.update_hp_grads(clear_lp_grads=False)

        # Free up the memory from the output of forward()
        self.pipe_buffers['output_tensors'][buffer_id] = None
        self.pipe_buffers['outputs'][buffer_id] = None
        grad_tensors = None

        if self.wall_clock_breakdown():
            self.timers(BACKWARD_INNER_MICRO_TIMER).stop()
            self.timers(BACKWARD_INNER_GLOBAL_TIMER).stop()
            self.timers(BACKWARD_MICRO_TIMER).stop()
            self.timers(BACKWARD_GLOBAL_TIMER).stop()

        self.mem_status('AFTER BWD')

    def _exec_load_micro_batch(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers(BATCH_INPUT_TIMER).start()

        batch = self._next_batch()

        if self.is_first_stage():
            loaded = None
            if torch.is_tensor(batch[0]):
                loaded = batch[0].clone().to(self.device).detach()
                if self._config.pipeline['activation_checkpoint_interval'] > 0 and self._config.pipeline[
                        'use_reentrant']:
                    loaded.requires_grad = loaded.is_floating_point()
            else:
                assert isinstance(batch[0], (tuple, list))
                # Assume list or tuple
                loaded = []
                for x in batch[0]:
                    assert torch.is_tensor(x)
                    mine = x.clone().detach().to(self.device)
                    if self._config.pipeline['activation_checkpoint_interval'] > 0 and self._config.pipeline[
                            'use_reentrant']:
                        mine.requires_grad = mine.is_floating_point()
                    loaded.append(mine)
                loaded = tuple(loaded)

            self.pipe_buffers['inputs'][buffer_id] = loaded

        if self.is_last_stage():
            loaded = batch[1]
            if torch.is_tensor(batch[1]):
                loaded = batch[1].to(self.device)
            # XXX: torch 1.6.0 DataLoader will auto convert tuple to list
            elif isinstance(batch[1], (tuple, list)):
                loaded = []
                for x in batch[1]:
                    assert torch.is_tensor(x)
                    x = x.to(self.device).detach()
                    loaded.append(x)
                loaded = tuple(loaded)

            self.pipe_buffers['labels'][buffer_id] = loaded

        if self.wall_clock_breakdown():
            self.timers(BATCH_INPUT_TIMER).stop()

    def _send_tensor_meta(self, buffer, recv_stage):
        """ Communicate metadata about upcoming p2p transfers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list)
            * num_tensors if type=list
            foreach tensor in buffer:
                * ndims
                * shape
        """
        meta_buffer = torch.empty(TENSOR_META_SIZE, dtype=torch.int32, device=self.device)
        if isinstance(buffer, torch.Tensor):
            meta_buf_list = [
                0,  # type of data (0: tensor, 1: list (unused), 2: tuple)
                self.DTYPE_TO_ID[buffer.dtype],  # dtype
                len(buffer.size())  # ndims
            ]
            meta_buf_list.extend(buffer.size())
            assert len(
                meta_buf_list
            ) <= TENSOR_META_SIZE, f"Buffer for metadata is too small. Current buffer size: {TENSOR_META_SIZE} but required {len(meta_buf_list)}"
            meta_buffer[:len(meta_buf_list)].copy_(torch.tensor(meta_buf_list, dtype=torch.int32))
            p2p.send(meta_buffer, recv_stage)

        elif isinstance(buffer, tuple):
            meta_buf_list = [
                2,  # type of data (0: tensor, 1: list (unused), 2: tuple)
                len(buffer)  # num_tensors
            ]

            for tensor in buffer:
                assert isinstance(tensor, torch.Tensor)
                meta_buf_list.append(self.DTYPE_TO_ID[tensor.dtype])
                meta_buf_list.append(len(tensor.size()))
                meta_buf_list.extend(tensor.size())

            assert len(
                meta_buf_list
            ) <= TENSOR_META_SIZE, f"Buffer for metadata is too small. Current buffer size: {TENSOR_META_SIZE} but required {len(meta_buf_list)}"
            meta_buffer[:len(meta_buf_list)].copy_(torch.tensor(meta_buf_list, dtype=torch.int32))
            p2p.send(meta_buffer, recv_stage)

        else:
            raise NotImplementedError(f'Could not send meta type {type(buffer)}')

        # Useful for performance debugging.
        '''
        if self.grid.data_parallel_id == 0:
            print(f'STAGE={self.stage_id} pipe-send-volume: {send_bytes/1024**2:0.2f}MB')
        '''

    def _recv_tensor_meta(self, send_stage):
        """Receive metadata about upcoming p2p transfers and return allocated buffers.

        Returns:
            Allocated buffer for receiving from send_stage.
        """
        buffer = torch.empty(TENSOR_META_SIZE, dtype=torch.int32, device=self.device)
        p2p.recv(buffer, send_stage)

        recv_type = buffer[0].item()

        # A single tensor will be sent.
        if recv_type == 0:
            recv_dtype = self.ID_TO_DTYPE[buffer[1].item()]
            recv_ndims = buffer[2].item()
            recv_shape = buffer[3:3 + recv_ndims].tolist()
            return self._allocate_or_extend_buffers(0, recv_shape, recv_dtype)

        # List or tuple of tensors (recv_type == 1 (list) is currently unused)
        elif recv_type == 1 or recv_type == 2:
            num_tensors = buffer[1].item()

            buffers = []
            offset = 2
            for idx in range(num_tensors):
                recv_dtype = self.ID_TO_DTYPE[buffer[offset].item()]
                recv_ndims = buffer[offset + 1].item()
                recv_shape = buffer[offset + 2:offset + 2 + recv_ndims].tolist()
                offset += 2 + recv_ndims

                buffers.append(self._allocate_or_extend_buffers(idx, recv_shape, recv_dtype))

            # Convert to tuples if requested.
            if recv_type == 2:
                buffers = tuple(buffers)
            return buffers

        else:
            raise NotImplementedError(f'Could not receive type {type(recv_type)}')

    def _exec_send_activations(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers(PIPE_SEND_OUTPUT_TIMER).start()

        outputs = self.pipe_buffers['outputs'][buffer_id]

        # NCCL does not like to send torch.BoolTensor types, so cast the mask to half().
        # We could do char, but with half() we can eventually flatten with other fp16
        # messages (TODO)
        if self.has_attention_mask or self.has_bool_tensors:
            outputs = list(outputs)
            outputs[-1] = outputs[-1].half()
            outputs = tuple(outputs)

        if self.dynamic_shape or self.first_output_send:
            self.first_output_send = False
            self._send_tensor_meta(outputs, self.next_stage)

        if isinstance(outputs, torch.Tensor):
            p2p.send(outputs, self.next_stage)
        elif isinstance(outputs, tuple):
            for idx, buffer in enumerate(outputs):
                p2p.send(buffer, self.next_stage)
        else:
            raise NotImplementedError('Could not send output of type '
                                      f'{type(outputs)}')

        # Restore the boolean tensor
        if self.has_attention_mask or self.has_bool_tensors:
            outputs = list(outputs)
            outputs[-1] = outputs[-1].bool()
            outputs = tuple(outputs)

        if self.wall_clock_breakdown():
            self.timers(PIPE_SEND_OUTPUT_TIMER).stop()

    def _exec_send_grads(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers(PIPE_SEND_GRAD_TIMER).start()

        inputs = self.pipe_buffers['inputs'][buffer_id]

        # Partition the gradient
        if self.is_grad_partitioned:
            if isinstance(inputs, tuple):
                first_input = inputs[0]
                assert all([torch.is_tensor(elt) for elt in inputs[1:]])
                inputs_grad_tail = [elt.grad for elt in inputs[1:]]
            elif torch.is_tensor(inputs):
                first_input = inputs
                inputs_grad_tail = []
            else:
                raise ValueError("expecting a tensor or a tuple of tensors")
            assert torch.is_tensor(first_input)
            part = PartitionedTensor(tensor=first_input.grad, group=self.grid.get_slice_parallel_group())

            inputs = (part.to_meta(), part.data(), *inputs_grad_tail)

        # XXX Terrible hack
        # Drop the attention mask from the input buffer here. It does not have
        # a grad that needs to be communicated. We free the buffer immediately
        # after, so no need to restore it. The receiver also has a hack that skips
        # the recv. This is because NCCL does not let us send torch.BoolTensor :-(.
        if self.has_attention_mask or self.has_bool_tensors:
            inputs = list(inputs)
            inputs.pop()
            inputs = tuple(inputs)

        if isinstance(inputs, torch.Tensor):
            assert inputs.grad is not None
            p2p.send(inputs.grad, self.prev_stage)
        else:
            # XXX terrible hacky branch
            if self.is_grad_partitioned:
                # First two sends are partitioned gradient
                p2p.send(inputs[0], self.prev_stage)
                p2p.send(inputs[1], self.prev_stage)
            else:
                for idx, buffer in enumerate(inputs):
                    # Skip tensors that will not produce a grad
                    if not buffer.is_floating_point():
                        assert buffer.grad is None
                        continue
                    assert buffer.grad is not None
                    p2p.send(buffer.grad, self.prev_stage)

        # We can free up the input buffer now
        self.pipe_buffers['inputs'][buffer_id] = None

        if self.wall_clock_breakdown():
            self.timers(PIPE_SEND_GRAD_TIMER).stop()

    def _exec_recv_activations(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers(PIPE_RECV_INPUT_TIMER).start()

        recvd = None

        # Allocate the buffer if necessary
        if self.dynamic_shape or self.pipe_recv_buf is None:
            self.pipe_recv_buf = self._recv_tensor_meta(self.prev_stage)

        if isinstance(self.pipe_recv_buf, torch.Tensor):
            p2p.recv(self.pipe_recv_buf, self.prev_stage)
            recvd = self.pipe_recv_buf.clone().detach()
            recvd.requires_grad = recvd.is_floating_point()
        else:
            assert isinstance(self.pipe_recv_buf, tuple)
            recvd = [None] * len(self.pipe_recv_buf)
            for idx, buffer in enumerate(self.pipe_recv_buf):
                assert torch.is_tensor(buffer)
                # XXX hardcode meta type
                if self.is_pipe_partitioned and idx == 0 and buffer.dtype != torch.long:
                    if self.meta_buffer is None:
                        self.meta_buffer = torch.zeros(buffer.size(), dtype=torch.long, device=self.device)
                    buffer = self.meta_buffer

                p2p.recv(buffer, self.prev_stage)
                recvd[idx] = buffer.clone().detach()

            # NCCL does not like to send torch.BoolTensor types, so un-cast the
            # attention mask
            if self.has_attention_mask or self.has_bool_tensors:
                recvd[-1] = recvd[-1].bool()

            recvd = tuple(recvd)

            for buffer in recvd:
                buffer.requires_grad = buffer.is_floating_point()

        self.pipe_buffers['inputs'][buffer_id] = recvd

        if self.wall_clock_breakdown():
            self.timers(PIPE_RECV_INPUT_TIMER).stop()

    def _exec_recv_grads(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers(PIPE_RECV_GRAD_TIMER).start()

        outputs = self.pipe_buffers['outputs'][buffer_id]
        # XXX these shapes are hardcoded for Megatron
        # Restore partitioned output if it was partitioned and we are sending full gradients
        if self.is_pipe_partitioned and not self.is_grad_partitioned:
            if self.pipe_partition_grad_meta_cache is None:
                self.pipe_partition_grad_meta_cache = outputs[0].to('cpu')
            part_output = PartitionedTensor.from_meta(meta=self.pipe_partition_grad_meta_cache,
                                                      local_part=outputs[1],
                                                      group=self.grid.get_slice_parallel_group())
            outputs[0].data = part_output.full()
            outputs = (outputs[0], *outputs[2:])
            # save for backward
            self.pipe_buffers['outputs'][buffer_id] = outputs

        # Allocate gradient if necessary
        if self.dynamic_shape or self.grad_layer is None:
            if isinstance(outputs, torch.Tensor):
                self.grad_layer = self._allocate_or_extend_buffers(0, list(outputs.size()), outputs.dtype)
            else:
                # XXX This is a HACK
                # When we exchange activations/gradients, the two pipe stages
                # need to issue the send/recv with the same buffer sizes or
                # else there is a deadlock. The is_floating_point() filter is
                # used to avoid sending gradients for tensors that do not
                # produce gradients. When TP>1, we partition the first
                # activations/gradients across TP ranks to save communication
                # volume and memory. That partitioned tensor is represented as
                # two tensors: a 1/TPth chunk of the original data and also a
                # small LongTensor storing the metadata used to reconstruct on
                # the other side. When combined, the floating point filter also
                # filtered out the metadata tensor. This quick (hacky) fix just
                # branches on is_grad_partitioned so we don't filter out the
                # metadata tensor.
                if self.is_grad_partitioned:
                    sizes_and_dtypes = [(list(t.size()), t.dtype)
                                        for t in outputs[:2]] + [(list(t.size()), t.dtype)
                                                                 for t in outputs[2:] if t.is_floating_point()]
                else:
                    sizes_and_dtypes = [(list(t.size()), t.dtype) for t in outputs if t.is_floating_point()]

                self.grad_layer = [
                    self._allocate_or_extend_buffers(i, size, dtype)
                    for i, (size, dtype) in enumerate(sizes_and_dtypes)
                ]

        if isinstance(self.grad_layer, torch.Tensor):
            p2p.recv(self.grad_layer, self.next_stage)
        else:
            assert isinstance(outputs, tuple)
            for idx, buffer in enumerate(self.grad_layer):
                # XXX GPT-2 hack
                if self.is_grad_partitioned and idx == 0 and buffer.dtype != torch.long:
                    buffer.data = torch.zeros(buffer.size(), dtype=torch.long, device=self.device)
                p2p.recv(buffer, self.next_stage)

        if self.wall_clock_breakdown():
            self.timers(PIPE_RECV_GRAD_TIMER).stop()

    def _exec_optimizer_step(self, lr_kwargs=None):
        if self.wall_clock_breakdown():
            self.timers(STEP_MICRO_TIMER).start()
            self.timers(STEP_GLOBAL_TIMER).start()
        self.mem_status('BEFORE STEP', reset_max=True)

        self._force_grad_boundary = True
        self._take_model_step(lr_kwargs)
        self._force_grad_boundary = False

        self.mem_status('AFTER STEP')

        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [(f'Train/Samples/lr', self.get_lr()[0], self.global_samples)]
            if self.fp16_enabled() and hasattr(self.optimizer, 'cur_scale'):
                self.summary_events.append(
                    (f'Train/Samples/loss_scale', self.optimizer.cur_scale, self.global_samples))
            self.monitor.write_events(self.summary_events)

        if self.wall_clock_breakdown():
            self.timers(STEP_MICRO_TIMER).stop()
            self.timers(STEP_GLOBAL_TIMER).stop()
            if self.global_steps % self.steps_per_print() == 0:
                self.timers.log([
                    BATCH_INPUT_TIMER,
                    FORWARD_MICRO_TIMER,
                    BACKWARD_MICRO_TIMER,
                    BACKWARD_INNER_MICRO_TIMER,
                    BACKWARD_REDUCE_MICRO_TIMER,
                    STEP_MICRO_TIMER,
                ])
            if self.global_steps % self.steps_per_print() == 0:
                self.timers.log([
                    FORWARD_GLOBAL_TIMER,
                    BACKWARD_GLOBAL_TIMER,
                    BACKWARD_INNER_GLOBAL_TIMER,
                    BACKWARD_REDUCE_GLOBAL_TIMER,
                    STEP_GLOBAL_TIMER,
                ])

    def _allocate_zeros(self, shape, **kwargs):
        """ Allocate a tensor of zeros on the engine's device.

        Arguments:
            shape: the shape of the tensor to allocate
            kwargs: passed to torch.zeros()

        Returns:
            A tensor from torch.zeros() allocated on self.device.
        """
        if "dtype" not in kwargs:
            if self.fp16_enabled():
                kwargs["dtype"] = torch.half
            if self.bfloat16_enabled():
                kwargs["dtype"] = torch.bfloat16

        return torch.zeros(shape, device=self.device, **kwargs)

    def _allocate_buffer(self, shape, num_buffers=-1, **kwargs):
        buffers = []
        if num_buffers == -1:
            num_buffers = self.num_pipe_buffers
        for count in range(num_buffers):
            buffers.append(self._allocate_zeros(shape, **kwargs))
        return buffers

    def _allocate_or_extend_buffers(self, idx, shape, dtype):
        numel = reduce(mul, shape) if len(shape) > 0 else 1
        if len(self._grad_layer_buf) <= idx or self._grad_layer_buf[idx].numel() < numel:
            new_buf = self._allocate_buffer(shape, dtype=dtype, num_buffers=1)[0]
            if len(self._grad_layer_buf) <= idx:
                self._grad_layer_buf.append(new_buf)
            else:
                self._grad_layer_buf[idx] = new_buf
            return self._grad_layer_buf[idx]
        else:
            return self._grad_layer_buf[idx].flatten()[:numel].view(shape)

    def forward(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def backward(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def step(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def mem_status(self, msg, print_rank=-1, reset_max=False):
        return
        global mem_alloced, mem_cached
        if not self.global_steps == 0 or not self.global_steps == 9:
            #return
            pass
        if self.mpu.get_data_parallel_rank() != 0:
            return

        if self.global_rank != 0:
            return

        rank = self.global_rank
        if print_rank != -1 and rank != print_rank:
            return

        get_accelerator().synchronize()

        if reset_max:
            get_accelerator().reset_max_memory_cached()
            get_accelerator().reset_max_memory_allocated()

        new_alloced = get_accelerator().memory_allocated()
        new_cached = get_accelerator().memory_cached()

        delta_alloced = new_alloced - mem_alloced
        delta_cached = new_cached - mem_cached

        mem_cached = new_cached
        mem_alloced = new_alloced

        max_alloced = get_accelerator().max_memory_allocated()
        max_cached = get_accelerator().max_memory_cached()

        # convert to GB for printing
        new_alloced /= 1024**3
        new_cached /= 1024**3
        delta_alloced /= 1024**3
        delta_cached /= 1024**3
        max_alloced /= 1024**3
        max_cached /= 1024**3

        print(
            f'RANK={rank} STAGE={self.stage_id} STEP={self.global_steps} MEMSTATS', msg,
            f'current alloc={new_alloced:0.4f}GB (delta={delta_alloced:0.4f}GB max={max_alloced:0.4f}GB) '
            f'current cache={new_cached:0.4f}GB (delta={delta_cached:0.4f}GB max={max_cached:0.4f}GB)')

    def module_state_dict(self, exclude_frozen_parameters=False):
        """Override hack to save a pipe model and return the directory path of the save.

        This method should only be called by DeepSpeed's ``save_checkpoint()``. The
        recommended way of saving a ``PipelineModule`` outside of ``save_checkpoint()``
        is ``save_state_dict()``.

        Returns:
            None
        """
        assert isinstance(self.module, PipelineModule)
        assert self._curr_ckpt_path is not None, \
            "PipelineEngine expects module_state_dict() to be called from save_checkpoint()"

        self.module.save_state_dict(self._curr_ckpt_path,
                                    checkpoint_engine=self.checkpoint_engine,
                                    exclude_frozen_params=exclude_frozen_parameters)
        return None

    def load_module_state_dict(self, checkpoint, strict=True, custom_load_fn=None, fetch_z3_params=False):
        """Override hack to instead use a directory path.

        This is important because pipeline models checkpoint by layer instead of rank.

        If ``state_dict`` is not ``None`` or a ``str``, we revert to ``super()`` expecting a ``dict``.

        Args:
            state_dict (str, None): unused
            strict (bool, optional): Strict state loading. Defaults to True.
        """
        assert custom_load_fn is None, "custom_load_fn not supported w. pipeline parallelism"
        state_dict = checkpoint if self.has_moe_layers else checkpoint['module']
        if (state_dict is not None) and (not isinstance(state_dict, str)):
            super().load_module_state_dict(state_dict, strict)
            return

        self.module.load_state_dir(load_dir=self._curr_ckpt_path,
                                   strict=strict,
                                   checkpoint_engine=self.checkpoint_engine)

    # A map of PipeInstruction types to methods. Each method will be executed with the
    # kwargs provided to the PipeInstruction from the scheduler.
    _INSTRUCTION_MAP = {
        schedule.OptimizerStep: _exec_optimizer_step,
        schedule.ReduceGrads: _exec_reduce_grads,
        schedule.ReduceTiedGrads: _exec_reduce_tied_grads,
        schedule.LoadMicroBatch: _exec_load_micro_batch,
        schedule.ForwardPass: _exec_forward_pass,
        schedule.BackwardPass: _exec_backward_pass,
        schedule.SendActivation: _exec_send_activations,
        schedule.RecvActivation: _exec_recv_activations,
        schedule.SendGrad: _exec_send_grads,
        schedule.RecvGrad: _exec_recv_grads,
    }

    def _exec_schedule(self, pipe_schedule):
        # Reserve and reset buffers.
        self._reserve_pipe_buffers(pipe_schedule.num_pipe_buffers())
        self.fwd_outputs = []

        # For each step in the schedule
        for step_cmds in pipe_schedule:
            # For each instruction in the step
            for cmd in step_cmds:
                if type(cmd) not in self._INSTRUCTION_MAP:
                    raise RuntimeError(f'{self.__class__.__name__} does not understand instruction {repr(cmd)}')

                # Equivalent to: self._exec_forward_pass(buffer_id=0)
                self._exec_instr = MethodType(self._INSTRUCTION_MAP[type(cmd)], self)
                self._exec_instr(**cmd.kwargs)

    def get_additional_losses(self):
        return self.agg_additional_losses
