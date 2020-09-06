'''
Copyright 2019 The Microsoft DeepSpeed Team
'''
import time
import logging
import copy
import os

from numpy import prod

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from deepspeed.utils.logging import logger

from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.engine import MEMORY_OPT_ALLREDUCE_SIZE
from deepspeed.utils.timer import ThroughputTimer

from deepspeed.runtime.utils import PartitionedTensor, ensure_directory_exists

from .module import PipelineModule, PipelineError, TiedLayerSpec
from . import p2p

TARGET_ID = -2
LOG_STAGE = -2
DATA_PARALLEL_ID = -2


def is_even(number):
    return number % 2 == 0


mem_alloced = 0
mem_cached = 0


def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


class PipelineEngine(DeepSpeedEngine):
    """ A model wrapper for pipeline-parallel execution.

    Parallelism is achieved by executing micro-batches in a pipelined fashion with
    gradient accumulation.
    """
    def __init__(self, *super_args, **super_kwargs):
        super().__init__(*super_args, **super_kwargs)
        assert isinstance(self.module, PipelineModule), "model must base PipelineModule"

        # pipeline step for logging
        self.log_batch_step_id = -1

        self.micro_batch_size = self.train_micro_batch_size_per_gpu()
        self.micro_batches = self.gradient_accumulation_steps()
        '''---------Set Grid and Communication Groups------'''
        self.grid = self.module._grid
        if self.grid.get_global_rank() == 0:
            logging.info(f'CONFIG: micro_batches={self.micro_batches} '
                         f'micro_batch_size={self.micro_batch_size}')

        self.global_rank = self.grid.get_global_rank()

        assert self.dp_world_size == self.grid.data_parallel_size
        assert self.train_batch_size() == \
            self.micro_batch_size * self.micro_batches * self.grid.data_parallel_size
        '''---------------------Set Stage Info------------------'''
        self.num_stages = self.grid.pipe_parallel_size
        self.stage_id = self.grid.get_stage_id()
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1

        self.first_stage = 0
        self.last_stage = self.num_stages - 1

        self.is_first_stage = self.stage_id == self.first_stage
        self.is_last_stage = self.stage_id == self.last_stage

        # Later stages need fewer buffers because less time between fwd/bwd passes
        self.num_sim_stages = 1
        self.num_sim_buffers = max(self.num_sim_stages - self.stage_id + 1, 0)
        if self.is_last_stage:
            self.num_sim_buffers = 2
        self.num_buffers = max(
            min(self.num_stages - self.stage_id + 1,
                self.micro_batches),
            2)
        self.num_buffers = max(self.num_buffers, self.num_sim_buffers)
        if self.grid.data_parallel_id == 0 and self.num_sim_stages > self.num_stages:
            print(
                f'RANK={self.global_rank} STAGE={self.stage_id} num_buffers={self.num_buffers} num_sim_buffers={self.num_sim_buffers}'
            )
        self.alloced_sim_buffers = False

        self.data_iterator = None
        self.batch_fn = None

        self.batch_timer = ThroughputTimer(batch_size=self.micro_batch_size *
                                           self.micro_batches,
                                           num_workers=self.dp_world_size,
                                           logging_fn=self.tput_log,
                                           monitor_memory=False,
                                           steps_per_output=self.steps_per_print())

        # PipelineEngine needs to handle data loading specially due to only the first
        # and last stages loading inputs/labels. We construct a sampler that uses
        if self.training_data:
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.training_data,
                num_replicas=self.dp_world_size,
                rank=self.mpu.get_data_parallel_rank(),
                shuffle=False)
            pipe_dataloader = self.deepspeed_io(self.training_data, data_sampler=sampler)
            self.set_dataloader(pipe_dataloader)

        self.is_pipe_parallel = self.grid.pipe_parallel_size > 1
        self.is_data_parallel = self.grid.data_parallel_size > 1
        self.is_model_parallel = self.grid.model_parallel_size > 1

        # Partition input/output buffers
        self.is_pipe_partitioned = self.is_model_parallel
        self.is_grad_partitioned = False  #self.is_model_parallel
        if self.global_rank == 0:
            print(f'RANK={self.global_rank} '
                  f'PIPE_PARTITION={self.is_pipe_partitioned} '
                  f'GRAD_PARTITION={self.is_grad_partitioned}')

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
        params_tensor = torch.LongTensor(data=[num_params,
                                               unique_params]).to(self.device)
        dist.all_reduce(params_tensor, group=self.grid.get_model_parallel_group())
        params_tensor = params_tensor.tolist()
        total_params = params_tensor[0]
        unique_params = params_tensor[1]
        print(f'RANK={self.global_rank} '
              f'STAGE={self.stage_id} '
              f'LAYERS={self.module._local_stop - self.module._local_start} '
              f'[{self.module._local_start}, {self.module._local_stop}) '
              f'STAGE_PARAMS={num_params} ({num_params/1e6:0.3f}M) '
              f'TOTAL_PARAMS={total_params} ({total_params/1e6:0.3f}M) '
              f'UNIQUE_PARAMS={unique_params} ({unique_params/1e6:0.3f}M)')

        #intialize peer-2-peer communication and allreduce groups
        if self.is_pipe_parallel:
            p2p.init_process_groups(self.grid)

        self.reduce_and_update = False
        '''---------Input, Output and Grads for Communication------'''
        self.pipe_recv_buf = None
        self.in_layers = [None] * self.num_buffers
        self.grad_layer = None
        self.out_layers = [None] * self.num_buffers
        self.out_layer_saved_tensors = [None
                                        ] * self.num_buffers  # for saving comp graphs
        self.labels = {}

        self.meta_buffer = None

        self.first_output_send = True
        self.first_gradient_send = True

        #stores the loss for the current micro batch being processed
        self.loss = torch.tensor(0.0).to(self.device)

        #stores the loss for the entire batch
        self.total_loss = torch.tensor(0.0, requires_grad=False).to(self.device)
        self.agg_loss = torch.tensor(0.0, requires_grad=False).to(self.device)
        self.dp_group_loss = torch.tensor(0.0, requires_grad=False).to(self.device)

        if self._config.pipeline['activation_checkpoint_interval'] > 0:
            self.module.activation_checkpoint_interval = self._config.pipeline[
                'activation_checkpoint_interval']

        if self.is_last_stage:
            self.loss_model = self.module.loss_fn
            self.loss_input = None
            self._allocate_labels()

        self.prev_sample_id = None

        # Initialize pipeline communicators. Just send total_loss
        if is_even(self.stage_id):
            if not self.is_last_stage:
                p2p.send(self.total_loss, self.next_stage)
            if not self.is_first_stage:
                p2p.recv(self.total_loss, self.prev_stage)
        else:
            if not self.is_first_stage:
                p2p.recv(self.total_loss, self.prev_stage)
            if not self.is_last_stage:
                p2p.send(self.total_loss, self.next_stage)

        # XXX look into timer reporting timing
        # Initialize some timers because of early weirdness.
        if self.wall_clock_breakdown():
            self.timers('forward_microstep').start()
            self.timers('forward_microstep').stop()
            self.timers('backward_microstep').start()
            self.timers('backward_microstep').stop()
            self.timers('backward_inner_microstep').start()
            self.timers('backward_inner_microstep').stop()
            self.timers('backward_allreduce_microstep').start()
            self.timers('backward_allreduce_microstep').stop()
            self.timers('backward_allreduce').start()
            self.timers('backward_allreduce').stop()
            self.timers('step_microstep').start()
            self.timers('step_microstep').stop()

    def train_batch(self):
        """Progress the pipeline to train the next batch of data.

        Returns:
            All ranks return the loss from
        """
        total_steps = 2 * (self.micro_batches + self.num_stages - 1)
        self.prev_sample_id = -1

        self.total_loss.zero_()

        self.batch_timer.start()

        #load the first micro batch
        self._load_input_or_label(0, True)
        for step_id in range(total_steps):
            self.log_batch_step_id = step_id  # Just for logging
            if self.global_rank == 0:
                self.log_for_device(" ")
                self.log_for_device("--------------Step ID", step_id, "-------------")
                self.log_for_device(" ")

            is_last_step = step_id == (total_steps - 1)
            self.run_step(step_id, is_last_step=is_last_step)

        self.batch_timer.stop()

        self.sample_count = self.train_batch_size() * self.global_steps

        # Scale loss, average among DP ranks, and bcast loss to the rest of my DP group
        if self.is_last_stage:
            loss = self._scale_loss(self.total_loss)
            self.dp_group_loss = loss.clone().detach()

            ## Average loss across all data-parallel groups
            self.agg_loss = self.dp_group_loss.clone().detach()
            #print(f'RANK={self.global_rank} bcast SENDER src={self.global_rank} group={self.grid.pp_group}', flush=True)
            if self.is_data_parallel:
                dist.all_reduce(self.agg_loss, group=self.mpu.get_data_parallel_group())
                self.agg_loss /= self.dp_world_size

            assert self.global_rank in self.grid.pp_group
            losses = torch.Tensor([self.dp_group_loss, self.agg_loss]).to(self.device)
            dist.broadcast(tensor=losses,
                           src=self.global_rank,
                           group=self.mpu.get_pipe_parallel_group())

        else:
            # Get loss from last stage
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
            assert src_rank in self.grid.pp_group
            losses = torch.Tensor([0., 0.]).to(self.device)
            dist.broadcast(tensor=losses,
                           src=src_rank,
                           group=self.grid.get_pipe_parallel_group())
            self.dp_group_loss = losses[0].clone().detach()
            self.agg_loss = losses[1].clone().detach()

        # Tensorboard
        if self.tensorboard_enabled():
            if self.global_rank == 0:
                self.summary_events = [(f'Train/Samples/train_loss',
                                        self.agg_loss.mean().item(),
                                        self.global_samples)]
                for event in self.summary_events:  # write_summary_events
                    self.summary_writer.add_scalar(event[0], event[1], event[2])
                if self.global_steps % self.steps_per_print() == 0:
                    self.summary_writer.flush()

        if self.wall_clock_breakdown(
        ) and self.global_steps % self.steps_per_print() == 0:
            self.timers.log([
                'pipe_send_output',
                'pipe_send_grad',
                'pipe_recv_input',
                'pipe_recv_grad'
            ])

        # TODO: should return precisely what loss returned and allow others to be queried?
        return self.agg_loss

    def set_dataloader(self, loader):
        """ Store a DataLoader for the first and last stages of the pipeline. """
        if self.is_first_stage or self.is_last_stage:
            self.training_dataloader = loader
            self.data_iterator = iter(self.training_dataloader)

    def set_dataiterator(self, iterator):
        """ Store a DataLoader for the first and last stages of the pipeline. """
        if self.is_first_stage or self.is_last_stage:
            self.training_dataloader = None
            self.data_iterator = iterator

    def set_batch_fn(self, fn):
        self.batch_fn = fn

    def is_gradient_accumulation_boundary(self):
        """ Override to add pipeline logic. """
        if not self.valid_sample(self.curr_sample_id):
            return False

        return (self.curr_sample_id + 1) % \
            self.gradient_accumulation_steps() == 0

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
        if self.is_model_parallel:
            mp_rank = self.grid.get_slice_parallel_rank()
        else:
            mp_rank = 0

        batch = None

        # Only MP rank 0 loads the data.
        if mp_rank == 0:
            if self.data_iterator is None:
                raise ValueError(
                    f"RANK={self.global_rank} First and last stages must call set_dataiterator() "
                    "before training.")
            try:
                batch = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.training_dataloader)
                return self._next_batch()
            '''
            if self.global_rank == 0:
                print(f'STEP={self.global_steps} SAMPLES={self.global_samples} BATCH={batch}')
            '''

        # All MP ranks participate in batch_fn, where they might broadcast the data.
        if self.batch_fn:
            batch = self.batch_fn(batch)

        # Sanity check dimensions.
        # XXX: the last minibatch with size < micro_batch_size kills us
        if torch.is_tensor(batch[0]):
            if batch[0].size(0) != self.micro_batch_size:
                self.data_iterator = iter(self.training_dataloader)
                return self._next_batch()
        else:
            assert torch.is_tensor(batch[0][0])
            if batch[0][0].size(0) != self.micro_batch_size:
                self.data_iterator = iter(self.training_dataloader)
                return self._next_batch()

        #assert batch[0].size(0) == self.micro_batch_size
        #assert batch[1].size(0) == self.micro_batch_size

        return batch

    def _load_input_or_label(self, sample_id, is_forward):
        if not self.valid_sample(sample_id) or not is_forward:
            return

        if not self.is_first_stage and not self.is_last_stage:
            return

        if self.wall_clock_breakdown():
            self.timers('batch_input').start()

        batch = self._next_batch()

        buf_idx = self._buffer_idx(sample_id)

        if self.is_first_stage:
            if torch.is_tensor(batch[0]):
                self.in_layers[buf_idx] = batch[0].to(self.device).detach()
                self.in_layers[buf_idx].requires_grad = self.in_layers[
                    buf_idx].is_floating_point()
            else:
                assert isinstance(batch[0], list) or isinstance(batch[0], tuple)
                # Assume list or tuple
                tmp = []
                for x in batch[0]:
                    assert torch.is_tensor(x)
                    mine = x.clone().detach().to(self.device)
                    mine.requires_grad = mine.is_floating_point()
                    tmp.append(mine)
                if isinstance(batch[0], tuple):
                    self.in_layers[buf_idx] = tuple(tmp)

        if self.is_last_stage:
            if torch.is_tensor(batch[1]):
                self.labels[buf_idx] = batch[1].to(self.device)
            else:
                tmp = []
                for x in batch[1]:
                    assert torch.is_tensor(x)
                    x = x.to(self.device).detach()
                    tmp.append(x)
                if isinstance(batch[1], tuple):
                    self.labels[buf_idx] = tuple(tmp)

        if self.wall_clock_breakdown():
            self.timers('batch_input').stop()

    def run_step(self, step_id, is_last_step=False):
        step_id = int(step_id)
        self.is_forward = True
        '''-----Identify Current Sample and Forward or Backward --'''
        if is_even(step_id) and is_even(self.stage_id):
            self.curr_sample_id = self._even_step_forward_id(step_id)

        if not is_even(step_id) and not is_even(self.stage_id):
            self.curr_sample_id = self._odd_step_forward_id(step_id)

        if is_even(step_id) and not is_even(self.stage_id):
            self.curr_sample_id = self._even_step_backward_id(step_id)
            self.is_forward = False

        if not is_even(step_id) and is_even(self.stage_id):
            self.curr_sample_id = self._odd_step_backward_id(step_id)
            self.is_forward = False

        #load the sample for next forward pass asynchronously
        self._load_next_input_or_label(self.curr_sample_id, self.is_forward)

        # Exchange forward activations and backward gradients
        self._communicate_activations(self.curr_sample_id,
                                      self.prev_sample_id,
                                      self.is_forward)

        # All Reduce gradients and update the model weights when if
        # self.reduce_and_update is True
        self._communicate_and_update_weights()

        # Run forward or backward on the model on this device
        self.compute(self.is_forward)

        #if its the last step and allreduce has not happened
        #then do the allreduce
        if is_last_step:
            self._communicate_and_update_weights()

        self.prev_sample_id = self.curr_sample_id

    def valid_sample(self, sample_id):
        return (sample_id >= 0) and (sample_id < self.micro_batches)

    def valid_stage(self, stage_id):
        return (stage_id >= self.first_stage) and (stage_id < self.num_stages)

    def _receive_input(self, sample_id, sender_stage):
        if not self.valid_sample(sample_id) or not self.valid_stage(sender_stage):
            return
        self.log_for_device(f'RECV INPUT sample={sample_id}')
        if self.wall_clock_breakdown():
            self.timers('pipe_recv_input').start()

        # Allocate the buffer if necessary
        if self.pipe_recv_buf is None:
            self.pipe_recv_buf = self._recv_tensor_meta(sender_stage)

        buf_idx = self._buffer_idx(sample_id)
        if isinstance(self.pipe_recv_buf, torch.Tensor):
            p2p.recv(self.pipe_recv_buf, sender_stage)
            self.in_layers[buf_idx] = self.pipe_recv_buf.clone().detach()
            self.in_layers[buf_idx].requires_grad = True
        else:
            assert isinstance(self.pipe_recv_buf, tuple)
            #for idx, buffer in enumerate(self.in_layers[buf_idx]):
            self.in_layers[buf_idx] = [None] * len(self.pipe_recv_buf)
            #print(f'{self.global_rank} RECV={[b.size() for b in self.pipe_recv_buf]}')
            for idx, buffer in enumerate(self.pipe_recv_buf):
                assert torch.is_tensor(buffer)
                # XXX hardcode meta type
                if self.is_pipe_partitioned and idx == 0 and buffer.dtype != torch.long:
                    if self.meta_buffer is None:
                        self.meta_buffer = torch.zeros(buffer.size(),
                                                       dtype=torch.long,
                                                       device=self.device)
                    buffer = self.meta_buffer

                p2p.recv(buffer, sender_stage)
                self.in_layers[buf_idx][idx] = buffer.clone().detach()

            # NCCL does not like to send torch.BoolTensor types, so un-cast the
            # attention mask
            if self.module.__class__.__name__ == 'GPT2ModelPipe':
                self.in_layers[buf_idx][-1] = self.in_layers[buf_idx][-1].bool()

            for idx, buffer in enumerate(self.in_layers[buf_idx]):
                buffer.requires_grad = buffer.is_floating_point()

            self.in_layers[buf_idx] = tuple(self.in_layers[buf_idx])

        if self.wall_clock_breakdown():
            self.timers('pipe_recv_input').stop()

    def _receive_grad(self, sample_id, sender_stage):
        if not self.valid_sample(sample_id) or not self.valid_stage(sender_stage):
            return
        self.log_for_device(f'RECV GRAD sample={sample_id}')
        if self.wall_clock_breakdown():
            self.timers('pipe_recv_grad').start()

        buf_idx = self._buffer_idx(sample_id)

        # XXX these shapes are hardcoded for Megatron
        # Restore partitioned output if it was partitioned and we are sending full gradients
        if self.is_pipe_partitioned and not self.is_grad_partitioned:
            part_output = PartitionedTensor.from_meta(
                meta=self.out_layers[buf_idx][0],
                local_part=self.out_layers[buf_idx][1],
                group=self.grid.get_slice_parallel_group())
            self.out_layers[buf_idx][0].data = part_output.full()
            self.out_layers[buf_idx] = tuple(
                [self.out_layers[buf_idx][0],
                 self.out_layers[buf_idx][2]])

        # Allocate gradient if necessary
        if self.grad_layer is None:
            if isinstance(self.out_layers[buf_idx], torch.Tensor):
                s = list(self.out_layers[buf_idx].size())
                self.grad_layer = self._allocate_buffer(s, num_buffers=1)[0]
            else:
                sizes = [
                    list(t.size()) for t in self.out_layers[buf_idx]
                    if t.is_floating_point()
                ]
                self.grad_layer = self._allocate_buffers(sizes, num_buffers=1)[0]

            # Count size of buffers
            # Useful for performance debugging.
            '''
            grad_bytes = 0
            if isinstance(self.grad_layer, torch.Tensor):
                grad_bytes += _tensor_bytes(self.grad_layer)
            else:
                for tensor in self.grad_layer:
                    grad_bytes += _tensor_bytes(tensor)
            grad_bytes /= 1024**2
            if self.grid.data_parallel_id == 0:
                print(
                    f'RANK={self.global_rank} STAGE={self.stage_id} grad_buffer_size={grad_bytes:0.2f}MB'
                )
            '''

        if isinstance(self.grad_layer, torch.Tensor):
            p2p.recv(self.grad_layer, sender_stage)
        else:
            assert isinstance(self.out_layers[buf_idx], tuple)
            for idx, buffer in enumerate(self.grad_layer):
                # XXX GPT-2 hack
                if self.is_grad_partitioned and idx == 0 and buffer.dtype != torch.long:
                    buffer.data = torch.zeros(buffer.size(),
                                              dtype=torch.long,
                                              device=self.device)
                p2p.recv(buffer, sender_stage)

        if self.wall_clock_breakdown():
            self.timers('pipe_recv_grad').stop()

    def _send_tensor_meta(self, buffer, recv_stage):
        """ Communicate metadata about upcoming p2p transfers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list)
            * num_tensors if type=list
            foreach tensor in buffer:
                * ndims
                * shape
        """
        send_bytes = 0
        if isinstance(buffer, torch.Tensor):
            type_tensor = torch.LongTensor(data=[0]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            send_shape = torch.LongTensor(data=buffer.size()).to(self.device)
            send_ndims = torch.LongTensor(data=[len(buffer.size())]).to(self.device)
            p2p.send(send_ndims, recv_stage)
            p2p.send(send_shape, recv_stage)
            send_bytes += _tensor_bytes(buffer)
        elif isinstance(buffer, list):
            assert (False)
            type_tensor = torch.LongTensor(data=[1]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            count_tensor = torch.LongTensor(data=[len(buffer)]).to(self.device)
            p2p.send(count_tensor, recv_stage)
            for tensor in buffer:
                assert isinstance(tensor, torch.Tensor)
                send_shape = torch.LongTensor(data=tensor.size()).to(self.device)
                send_ndims = torch.LongTensor(data=[len(tensor.size())]).to(self.device)
                p2p.send(send_ndims, recv_stage)
                p2p.send(send_shape, recv_stage)
                send_bytes += _tensor_bytes(tensor)
        elif isinstance(buffer, tuple):
            type_tensor = torch.LongTensor(data=[2]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            count_tensor = torch.LongTensor(data=[len(buffer)]).to(self.device)
            p2p.send(count_tensor, recv_stage)
            for idx, tensor in enumerate(buffer):
                assert isinstance(tensor, torch.Tensor)
                send_shape = torch.LongTensor(data=tensor.size()).to(self.device)
                send_ndims = torch.LongTensor(data=[len(tensor.size())]).to(self.device)
                p2p.send(send_ndims, recv_stage)
                p2p.send(send_shape, recv_stage)
                # Useful for performance debugging.
                '''
                new_bytes = _tensor_bytes(tensor)
                send_bytes += _tensor_bytes(tensor)
                # Useful for performance debugging.
                if self.grid.data_parallel_id == 0:
                    print(
                        f'STAGE={self.stage_id} pipe-send-volume[{idx}]: shape={send_shape} {new_bytes/1024**2:0.2f}MB'
                    )
                '''
        else:
            raise NotImplementedError(f'Could not send meta type {type(buffer)}')

        # Useful for performance debugging.
        '''
        if self.grid.data_parallel_id == 0:
            print(f'STAGE={self.stage_id} pipe-send-volume: {send_bytes/1024**2:0.2f}MB')
        '''

    def _recv_tensor_meta(self, send_stage):
        """Receive metadata about upcoming p2p transfers and return allocated buffers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list)
            * num_tensors if type=list
            foreach tensor in buffer:
                * ndims
                * shape

        Returns:
            Allocated buffer for receiving from send_stage.
        """

        type_tensor = torch.LongTensor(data=[0]).to(self.device)
        p2p.recv(type_tensor, send_stage)
        recv_type = type_tensor.item()

        # A single tensor will be sent.
        if recv_type == 0:
            recv_ndims = torch.LongTensor(data=[0]).to(self.device)
            p2p.recv(recv_ndims, send_stage)
            recv_ndims = recv_ndims.item()
            recv_shape = torch.LongTensor([1] * recv_ndims).to(self.device)
            p2p.recv(recv_shape, send_stage)
            recv_shape = recv_shape.tolist()
            return self._allocate_buffer(recv_shape, num_buffers=1)[0]

        # List or tuple of tensors
        elif recv_type == 1 or recv_type == 2:
            count_tensor = torch.LongTensor(data=[0]).to(self.device)
            p2p.recv(count_tensor, send_stage)
            num_tensors = count_tensor.item()
            recv_shapes = []
            for idx in range(num_tensors):
                recv_ndims = torch.LongTensor(data=[0]).to(self.device)
                p2p.recv(recv_ndims, send_stage)
                recv_ndims = recv_ndims.item()
                recv_shape = torch.LongTensor([1] * recv_ndims).to(self.device)
                p2p.recv(recv_shape, send_stage)
                recv_shapes.append(recv_shape.tolist())

            buffers = self._allocate_buffers(recv_shapes, num_buffers=1)[0]
            # Convert to tuples if requested.
            if recv_type == 2:
                buffers = tuple(buffers)
            return buffers

        else:
            raise NotImplementedError(f'Could not receive type {type(recv_type)}')

    def _send_output(self, sample_id, receiver_stage):
        if not self.valid_sample(sample_id) or not self.valid_stage(receiver_stage):
            return
        self.log_for_device(f'SEND OUTPUT sample={sample_id}')

        if self.wall_clock_breakdown():
            self.timers('pipe_send_output').start()

        buf_idx = self._buffer_idx(sample_id)

        # NCCL does not like to send torch.BoolTensor types, so cast the mask to half().
        # We could do char, but with half() we can eventually flatten with other fp16
        # messages (TODO)
        if self.module.__class__.__name__ == 'GPT2ModelPipe':
            self.out_layers[buf_idx] = list(self.out_layers[buf_idx])
            self.out_layers[buf_idx][-1] = self.out_layers[buf_idx][-1].half()
            self.out_layers[buf_idx] = tuple(self.out_layers[buf_idx])

        if self.first_output_send:
            self.first_output_send = False
            self._send_tensor_meta(self.out_layers[buf_idx], receiver_stage)

        if isinstance(self.out_layers[buf_idx], torch.Tensor):
            p2p.send(self.out_layers[buf_idx], receiver_stage)
        elif isinstance(self.out_layers[buf_idx], tuple):
            for idx, buffer in enumerate(self.out_layers[buf_idx]):
                p2p.send(buffer, receiver_stage)
        else:
            raise NotImplementedError('Could not send output of type '
                                      f'{type(self.out_layers[buf_idx])}')

        # Restore the boolean tensor
        if self.module.__class__.__name__ == 'GPT2ModelPipe':
            self.out_layers[buf_idx] = list(self.out_layers[buf_idx])
            self.out_layers[buf_idx][-1] = self.out_layers[buf_idx][-1].bool()
            self.out_layers[buf_idx] = tuple(self.out_layers[buf_idx])

        if self.wall_clock_breakdown():
            self.timers('pipe_send_output').stop()

    def _send_grad(self, sample_id, receiver_stage):
        if not self.valid_sample(sample_id) or not self.valid_stage(receiver_stage):
            return
        self.log_for_device(f'SEND GRAD sample={sample_id}')

        if self.wall_clock_breakdown():
            self.timers('pipe_send_grad').start()

        buf_idx = self._buffer_idx(sample_id)

        # Partition the gradient
        if self.is_grad_partitioned:
            part = PartitionedTensor(tensor=self.in_layers[buf_idx][0].grad,
                                     group=self.grid.get_slice_parallel_group())
            # Clear the large output data, but save the computation graph
            # Inject the partitoned tensor into the output before sending

            # XXX Hack
            self.in_layers[buf_idx] = tuple(
                [part.to_meta(),
                 part.data(),
                 self.in_layers[buf_idx][1]])

        # XXX Terrible hack
        # Drop the attention mask from the input buffer here. It does not have
        # a grad that needs to be communicated. We free the buffer immediately
        # after, so no need to restore it. The receiver also has a hack that skips
        # the recv. This is because NCCL does not let us send torch.BoolTensor :-(.
        if self.module.__class__.__name__ == 'GPT2ModelPipe':
            self.in_layers[buf_idx] = list(self.in_layers[buf_idx])
            self.in_layers[buf_idx].pop()
            self.in_layers[buf_idx] = tuple(self.in_layers[buf_idx])

        if isinstance(self.in_layers[buf_idx], torch.Tensor):
            assert self.in_layers[buf_idx].grad is not None
            p2p.send(self.in_layers[buf_idx].grad, receiver_stage)
        else:
            # XXX terrible hacky branch
            if self.is_grad_partitioned:
                # First two sends are partitioned gradient
                p2p.send(self.in_layers[buf_idx][0], receiver_stage)
                p2p.send(self.in_layers[buf_idx][1], receiver_stage)
                # XXX hack hack hack
                #p2p.send(self.in_layers[buf_idx][2].grad, receiver_stage)
            else:
                for idx, buffer in enumerate(self.in_layers[buf_idx]):
                    # Skip tensors that will not produce a grad
                    if not buffer.is_floating_point():
                        assert buffer.grad is None
                        continue
                    assert buffer.grad is not None
                    p2p.send(buffer.grad, receiver_stage)

        # We can free up the input buffer now
        self.in_layers[buf_idx] = None

        if self.wall_clock_breakdown():
            self.timers('pipe_send_grad').stop()

    '''Send Activations or gradient from previous step,
    and receive activation or gradient for the current step'''

    def _communicate_activations(self, sample_id, prev_sample_id, is_forward):
        '''make sure to communicate the activations first and gradients second
        or vice versa.
        Interleaving them will cause a deadlock'''
        if not self.is_pipe_parallel:
            return

        if is_forward:
            self._receive_input(sample_id, self.prev_stage)
            self._send_grad(prev_sample_id, self.prev_stage)
        else:
            self._send_output(prev_sample_id, self.next_stage)
            self._receive_grad(sample_id, self.next_stage)

    def _communicate_and_update_weights(self):
        if not self.reduce_and_update:
            return

        if self.wall_clock_breakdown():
            self.timers('step_microstep').start()
            self.timers('step').start()

        self.log_for_device(
            f'STEP pipe-sample={self.curr_sample_id} micro_steps={self.micro_steps} global_steps={self.global_steps+1}'
        )

        # Aggregate gradients
        if self.wall_clock_breakdown():
            self.timers('backward_tied_allreduce_microstep').start()
            self.timers('backward_tied_allreduce').start()

        self.module.allreduce_tied_weight_gradients()
        if self.is_data_parallel:
            self.log_for_device('ALLREDUCE GRADS')
            self.buffered_allreduce_fallback(
                elements_per_buffer=MEMORY_OPT_ALLREDUCE_SIZE)

        if self.wall_clock_breakdown():
            self.timers('backward_tied_allreduce').stop()
            self.timers('backward_tied_allreduce_microstep').stop()

        self.mem_status('BEFORE STEP', reset_max=True)
        '''
        # Just some debugging
        torch.cuda.synchronize()
        for rank in range(dist.get_world_size()):
            if rank == self.global_rank:
                for idx, layer in enumerate(self.module.forward_funcs):
                    if hasattr(layer, 'parameters'):
                        curr_layer = idx + self.module._local_start
                        if curr_layer != 2:
                            #continue
                            pass
                        grads = [p.grad.flatten()[0:2].tolist() for name, p in layer.named_parameters()]
                        #grads = [p.grad for name, p in layer.named_parameters()]
                        #print(f'RANK={dist.get_rank()} BEFORE STEP layer={curr_layer} grads={grads}')

            dist.barrier()


        for rank in range(dist.get_world_size()):
            if rank == self.global_rank:
                for idx, layer in enumerate(self.module.forward_funcs):
                    if hasattr(layer, 'parameters'):
                        curr_layer = idx + self.module._local_start
                        params = [p.flatten()[0:2].tolist() for name, p in layer.named_parameters()]
                        #print(f'RANK={dist.get_rank()} BEFORE STEP layer={curr_layer} params={params}')
            dist.barrier()
        '''

        #print(f'RANK={dist.get_rank()} layer= lr={self.get_lr()} mom={self.get_mom()}')
        self._take_model_step()
        '''
        torch.cuda.synchronize()
        for rank in range(dist.get_world_size()):
            if rank == self.global_rank:
                for idx, layer in enumerate(self.module.forward_funcs):
                    if hasattr(layer, 'parameters'):
                        curr_layer = idx + self.module._local_start
                        params = [p.flatten()[0:2].tolist() for name, p in layer.named_parameters()]
                        #print(f'RANK={dist.get_rank()} AFTER STEP layer={curr_layer} params={params}')
            dist.barrier()
        '''

        self.mem_status('AFTER STEP')

        if self.tensorboard_enabled():
            if self.global_rank == 0:
                self.summary_events = [(f'Train/Samples/lr',
                                        self.get_lr()[0],
                                        self.global_samples)]
                for event in self.summary_events:  # write_summary_events
                    self.summary_writer.add_scalar(event[0], event[1], event[2])

        self.reduce_and_update = False

        if self.wall_clock_breakdown():
            self.timers('step_microstep').stop()
            self.timers('step').stop()
            if self.global_steps % self.steps_per_print() == 0:
                self.timers.log([
                    'batch_input',
                    'forward_microstep',
                    'backward_microstep',
                    'backward_inner_microstep',
                    'backward_allreduce_microstep',
                    'backward_tied_allreduce_microstep',
                    'step_microstep'
                ])
            # Log timing
            if self.tensorboard_enabled():
                pass
                '''
                if self.global_rank == 0:
                    self.summary_events = [(f'Train/Samples/stage-{self.stage_id}/elapsed_time_ms_forward', self.timers('forward').elapsed(reset=False) * 1000.0, self.sample_count), \
                                            (f'Train/Samples/stage-{self.stage_id}/elapsed_time_ms_backward', self.timers('backward').elapsed(reset=False) * 1000.0, self.sample_count), \
                                            (f'Train/Samples/stage-{self.stage_id}/elapsed_time_ms_backward_inner', self.timers('backward_inner').elapsed(reset=False) * 1000.0, self.sample_count), \
                                            (f'Train/Samples/stage-{self.stage_id}/elapsed_time_ms_backward_allreduce', self.timers('backward_allreduce').elapsed(reset=False) * 1000.0, self.sample_count), \
                                            (f'Train/Samples/stage-{self.stage_id}/elapsed_time_ms_step', self.timers('step').elapsed(reset=False) * 1000.0, self.sample_count)
                                            ]
                    for event in self.summary_events:  # write_summary_events
                        self.summary_writer.add_scalar(event[0], event[1], event[2])
                '''
            if self.global_steps % self.steps_per_print() == 0:
                self.timers.log([
                    'forward',
                    'backward',
                    'backward_inner',
                    'backward_allreduce',
                    'step'
                ])

    def compute(self, is_forward):
        if is_forward:
            self.do_forward(self.curr_sample_id)
        else:
            self.do_backward(self.curr_sample_id)
            self.reduce_and_update = self.is_gradient_accumulation_boundary()

    def _even_step_forward_id(self, step_id):
        base = step_id // 2
        sample_id = int(base - self.stage_id // 2)
        return sample_id

    def _odd_step_forward_id(self, step_id):
        base = (step_id - 1) // 2
        sample_id = int(base - self.stage_id // 2)
        return sample_id

    def _even_step_backward_id(self, step_id):
        base = step_id // 2
        sample_id = int(base - self.num_stages + (self.stage_id + 1) // 2)
        return sample_id

    def _odd_step_backward_id(self, step_id):
        base = ((step_id - 1) // 2) - self.num_stages + 1
        sample_id = int(base + self.stage_id // 2)
        return sample_id

    def zero_grad_input(self, sample_id):
        buf_idx = self._buffer_idx(sample_id)
        if isinstance(self.in_layers[buf_idx], torch.Tensor):
            if self.in_layers[buf_idx].grad is not None:
                self.in_layers[buf_idx].grad.data.zero_()
        else:
            for t in self.in_layers[buf_idx]:
                if t.grad is not None:
                    t.grad.data.zero_()

    def do_forward(self, sample_id):
        if not self.valid_sample(sample_id):
            return

        self.tput_timer.start()
        self.log_for_device(f'FORWARD sample={sample_id}')
        self.mem_status('BEFORE FWD', reset_max=True)

        buf_idx = self._buffer_idx(sample_id)

        # collect the partitioned input from the previous stage
        if self.is_pipe_partitioned and not self.is_first_stage:
            #print(f'RANK={self.global_rank} FWD restoring {self.in_layers[buf_idx]}')
            part_input = PartitionedTensor.from_meta(
                meta=self.in_layers[buf_idx][0],
                local_part=self.in_layers[buf_idx][1],
                group=self.grid.get_slice_parallel_group())

            self.in_layers[buf_idx] = tuple(
                [part_input.full(),
                 self.in_layers[buf_idx][2]])
            self.in_layers[buf_idx][0].requires_grad = True
            # skip mask
            #self.in_layers[buf_idx][1].requires_grad = True
            part_input = None

        # Zero out the gradients each time we use the tensor because only the data in
        # tensor changes across batches
        self.zero_grad_input(sample_id)

        self.out_layers[buf_idx] = super().forward(self.in_layers[buf_idx])

        # Partition the outputs if we are not the last stage
        if self.is_pipe_partitioned and not self.is_last_stage:
            part = PartitionedTensor(tensor=self.out_layers[buf_idx][0],
                                     group=self.grid.get_slice_parallel_group())
            # Clear the large output data, but save the computation graph
            self.out_layers[buf_idx][0].data = torch.zeros(1)
            self.out_layer_saved_tensors[buf_idx] = self.out_layers[buf_idx][0]
            # Inject the partitioned tensor into the output before sending
            self.out_layers[buf_idx] = tuple(
                [part.to_meta(),
                 part.data(),
                 self.out_layers[buf_idx][1]])
            part = None

        # Allocate some unused buffers to simulate longer pipelines. deepcopy
        # preserves devices
        if (self.num_sim_buffers > self.num_buffers) and not self.alloced_sim_buffers:
            alloc = torch.cuda.memory_allocated()
            for sim_idx in range(self.num_buffers, self.num_sim_buffers):
                self.in_layers[sim_idx] = copy.deepcopy(self.in_layers[buf_idx])
            torch.cuda.synchronize()
            alloc = torch.cuda.memory_allocated() - alloc
            alloc = alloc / 1024**2  # MB
            num_alloced = self.num_sim_buffers - self.num_buffers
            print(
                f'RANK={self.global_rank} STAGE={self.stage_id} allocated {num_alloced} simulated buffers, total size {alloc:0.2f}MB'
            )
            self.alloced_sim_buffers = True

        self.mem_status('AFTER FWD')

        # Optionally compute loss on the last device
        if self.is_last_stage:
            if self.loss_model is not None:
                self.mem_status('BEFORE LOSS', reset_max=True)
                self.loss = self.loss_model(self.out_layers[buf_idx],
                                            self.labels[buf_idx])
                self.mem_status('AFTER LOSS')
            else:
                # Some models just return loss from forward()
                self.loss = self.out_layers[buf_idx]
            self.total_loss += self.loss.detach()

    def do_backward(self, sample_id):
        if not self.valid_sample(sample_id):
            return

        self.log_for_device(f'BACKWARD sample={sample_id}')

        assert self.optimizer is not None, "must provide optimizer during " \
                                           "init in order to use backward"

        buf_idx = self._buffer_idx(sample_id)

        self.mem_status('BEFORE BWD', reset_max=True)

        # The last stage just runs backward on the loss using DeepSpeed's typical
        # mechanisms.
        if self.is_last_stage:
            super().backward(self.loss, allreduce_gradients=False)
            self.mem_status('AFTER BWD')
            return

        if self.wall_clock_breakdown():
            self.timers('backward_microstep').start()
            self.timers('backward').start()
            self.timers('backward_inner_microstep').start()
            self.timers('backward_inner').start()

        # Reconstruct out_layers if we previously partitioned the output. We must be
        # careful to also restore the computational graph of the tensors we partitioned.
        if self.is_pipe_partitioned:
            if self.is_grad_partitioned:
                #print(f'RANK={self.global_rank} BEFORE-BWD restoring out_layers={self.out_layers}')
                part_output = PartitionedTensor.from_meta(
                    meta=self.out_layers[buf_idx][0],
                    local_part=self.out_layers[buf_idx][1],
                    group=self.grid.get_slice_parallel_group())
                self.out_layer_saved_tensors[buf_idx].data = part_output.full()
                self.out_layers[buf_idx] = tuple(
                    [self.out_layer_saved_tensors[buf_idx],
                     self.out_layers[buf_idx][2]])
            else:
                # Already restored from partition
                self.out_layer_saved_tensors[buf_idx].data = self.out_layers[buf_idx][0]
                self.out_layers[buf_idx] = tuple(
                    [self.out_layer_saved_tensors[buf_idx],
                     self.out_layers[buf_idx][1]])

        grad_tensors = self.grad_layer
        if self.is_grad_partitioned:
            #print(f'RANK={self.global_rank} BEFORE-BWD restoring grad={self.grad_layer[0].size()} {self.grad_layer[1].size()}')
            part_grad = PartitionedTensor.from_meta(
                meta=self.grad_layer[0],
                local_part=self.grad_layer[1],
                group=self.grid.get_slice_parallel_group())
            grad_tensors = tuple([part_grad.full(), self.grad_layer[2]])
            part_grad = None
            #print(f'RANK={self.global_rank} BEFORE-BWD restored grad={self.grad_layer[0].size()} {self.grad_layer[1].size()}')

        # This handles either a single tensor or tuple of tensors.
        if isinstance(self.out_layers[buf_idx], tuple):
            out_tensors = [t for t in self.out_layers[buf_idx] if t.is_floating_point()]
            #torch.autograd.backward(tensors=self.out_layers[buf_idx],
            assert len(out_tensors) == len(grad_tensors)
            torch.autograd.backward(tensors=out_tensors, grad_tensors=grad_tensors)
        else:
            torch.autograd.backward(tensors=(self.out_layers[buf_idx],
                                             ),
                                    grad_tensors=(grad_tensors,
                                                  ))

        # Free up the memory from the output of forward()
        self.out_layer_saved_tensors[buf_idx] = None
        self.out_layers[buf_idx] = None
        grad_tensors = None

        if self.wall_clock_breakdown():
            self.timers('backward_inner').stop()
            self.timers('backward_inner_microstep').stop()
            self.timers('backward').stop()
            self.timers('backward_microstep').stop()

        self.mem_status('AFTER BWD')

    def _allocate_zeros(self, shape, fp16=None, **kwargs):
        """ Allocate a tensor of zeros on the engine's device.

        Arguments:
            shape: the shape of the tensor to allocate
            fp16 (bool): whether to use FP16. default: defer to self.fp16_enabled()
            kwargs: passed to torch.zeros()

        Returns:
            A tensor from torch.zeros() allocated on self.device.
        """

        if fp16 is None:
            fp16 = self.fp16_enabled()

        if fp16:
            return torch.zeros(shape, dtype=torch.half, device=self.device, **kwargs)
        else:
            return torch.zeros(shape, device=self.device, **kwargs)

    def _buffer_idx(self, sample_id):
        """ Map a microbatch ID to a buffer. """
        assert self.valid_sample(sample_id)
        return sample_id % self.num_buffers

    def _allocate_buffer(self, shape, num_buffers=-1, **kwargs):
        buffers = []
        if num_buffers == -1:
            num_buffers = self.num_buffers
        for count in range(num_buffers):
            buffers.append(self._allocate_zeros(shape, **kwargs))
        return buffers

    def _allocate_buffers(self, shapes, requires_grad=False, num_buffers=-1):
        buffers = []
        if num_buffers == -1:
            num_buffers = self.num_buffers
        for count in range(num_buffers):
            buffer = []
            for shape in shapes:
                buffer.append(self._allocate_zeros(shape, requires_grad=requires_grad))
            buffers.append(buffer)
        return buffers

    def _allocate_labels(self):
        """ Allocate labels on the last pipeline stage. """
        if not self.is_last_stage:
            return
        for count in range(self.num_buffers):
            self.labels[count] = self._allocate_zeros(shape=[self.micro_batch_size],
                                                      fp16=False)

    def _load_next_input_or_label(self, sample_id, is_forward):
        if self.valid_sample(sample_id):
            self._load_input_or_label(sample_id + 1, is_forward)

    def forward(self, *args, **kwargs):
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def backward(self, *args, **kwargs):
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def step(self, *args, **kwargs):
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

        torch.cuda.synchronize()

        if reset_max:
            torch.cuda.reset_max_memory_cached()
            torch.cuda.reset_max_memory_allocated()

        new_alloced = torch.cuda.memory_allocated()
        new_cached = torch.cuda.memory_cached()

        delta_alloced = new_alloced - mem_alloced
        delta_cached = new_cached - mem_cached

        mem_cached = new_cached
        mem_alloced = new_alloced

        max_alloced = torch.cuda.max_memory_allocated()
        max_cached = torch.cuda.max_memory_cached()

        # convert to GB for printing
        new_alloced /= 1024**3
        new_cached /= 1024**3
        delta_alloced /= 1024**3
        delta_cached /= 1024**3
        max_alloced /= 1024**3
        max_cached /= 1024**3

        print(
            f'RANK={rank} STAGE={self.stage_id} STEP={self.global_steps} MEMSTATS',
            msg,
            f'current alloc={new_alloced:0.4f}GB (delta={delta_alloced:0.4f}GB max={max_alloced:0.4f}GB) '
            f'current cache={new_cached:0.4f}GB (delta={delta_cached:0.4f}GB max={max_cached:0.4f}GB)'
        )

    def module_state_dict(self):
        """Override hack to save a pipe model and return the directory path of the save.

        This method should only be called by DeepSpeed's ``save_checkpoint()``. The
        recommended way of saving a ``PipelineModule`` outside of ``save_checkpoint()``
        is ``save_state_dict()``.

        Returns:
            str: The directory path where the checkpoint was saved.
        """
        assert isinstance(self.module, PipelineModule)
        assert self._curr_save_path is not None, \
            "PipelineEngine expects module_state_dict() to be called from save_checkpoint()"

        self.module.save_state_dict(self._curr_save_path)
        return self._curr_save_path

    def load_module_state_dict(self, state_dict, strict=True):
        """Override hack to instead use a directory path.

        This is important because pipeline models checkpoint by layer instead of rank.

        If ``state_dict`` is not a ``str``, we revert to ``super()`` expecting a ``dict``.

        Args:
            state_dict (str): Path to the directory for checkpoint.
            strict (bool, optional): Strict state loading. Defaults to True.
        """
        if not isinstance(state_dict, str):
            super().load_module_state_dict(state_dict, strict)
            return

        self.module.load_state_dir(state_dict, strict=strict)
