"""
"Copyright 2020 The Microsoft DeepSpeed Team.
Licensed under the MIT license.
"""

import sys
import torch

from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from deepspeed.utils import log_dist
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.ops.op_builder import UtilsBuilder

from torch import Tensor
from torch.nn import Parameter
from typing import Deque, Dict, Tuple
from torch.cuda import Stream, Event
from deepspeed.runtime.zero.parameter_offload import DeepSpeedZeRoOffload
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.runtime.zero.stage3 import print_rank_0
from typing import List
import collections

from deepspeed import comm as dist
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.runtime.fp16.loss_scaler import DynamicLossScaler, LossScaler
from deepspeed.accelerator import get_accelerator

class MiCS_Optimizer(DeepSpeedZeroOptimizer_Stage3):
    """
    MiCS Optimizer
    """
    def __init__(self,
                 module,
                 init_optimizer,
                 timers,
                 ds_config,
                 static_loss_scale=1.0,
                 dynamic_loss_scale=False,
                 dynamic_loss_args=None,
                 verbose=True,
                 contiguous_gradients=True,
                 reduce_bucket_size=500000000,
                 prefetch_bucket_size=50000000,
                 max_reuse_distance=1000000000,
                 max_live_parameters=1000000000,
                 param_persistence_threshold=100000,
                 model_persistence_threshold=sys.maxsize,
                 dp_process_group=None,
                 reduce_scatter=True,
                 overlap_comm=False,
                 offload_optimizer_config=None,
                 offload_param_config=None,
                 sub_group_size=1000000000000,
                 mpu=None,
                 clip_grad=0.0,
                 communication_data_type=torch.float16,
                 postscale_gradients=True,
                 gradient_predivide_factor=1.0,
                 gradient_accumulation_steps=1,
                 elastic_checkpoint=False,
                 aio_config=None):

        log_dist(f'Initialize MiCS Optimizer', ranks=[0])

        # adopt from stage3 but calls wrapped MiCSOffload
        # which uses MiCS_Init context manager, which is derived from 
        # partition_parameters.py:Init
        see_memory_usage("Stage 3 initialize beginning", force=True)

        log_dist(f"initialized {__class__.__name__} with args: {locals()}", ranks=[0])
        log_dist(f"Reduce bucket size {reduce_bucket_size}", ranks=[0])
        log_dist(f"Prefetch bucket size {prefetch_bucket_size}", ranks=[0])

        # The fused optimizer does all the work. We need this layer for two reason:
        # 1. maintain same user API from apex.fp16_utils
        # 2. keep common stuff here in case we need to add ne552w fused optimizer later

        # differences from apex.fp16_utils:
        # - assume all model params in fp16
        # - assume all params requires grad
        # - flat by groups, not keeping state. TODO: remove state explicitly?
        # - master grad and unflat master weight never exist. TODO: a way to save out unflat master?
        if not get_accelerator().is_available():
            raise SystemError("Cannot use fp16 without accelerator.")

        self.optimizer = init_optimizer

        # Load pre-built or JIT compile (un)flatten ops
        util_ops = UtilsBuilder().load()
        self.flatten = util_ops.flatten
        self.unflatten = util_ops.unflatten
        self.dtype = self.optimizer.param_groups[0]['params'][0].dtype
        self._global_grad_norm = 0.

        self.custom_loss_scaler = False
        self.external_loss_scale = None

        self.optimizer_swapper = None
        self.swap_optimizer = False

        self.offload_optimizer = False
        self.offload_optimizer_pin_memory = False
        self.offload_optimizer_fast_init = False
        self.offload_param = False
        self.offload_param_pin_memory = False
        self.params_in_nvme_and_cpu = False
        self.max_params_in_cpu = 0

        self.parameter_offload = DeepSpeedZeRoOffload(
            module=module,
            timers=timers,
            ds_config=ds_config,
            overlap_comm=overlap_comm,
            prefetch_bucket_size=prefetch_bucket_size,
            max_reuse_distance=max_reuse_distance,
            max_live_parameters=max_live_parameters,
            param_persistence_threshold=param_persistence_threshold,
            model_persistence_threshold=model_persistence_threshold,
            offload_param_config=offload_optimizer_config,
            mpu=mpu)

        self.persistent_parameters = self.parameter_offload.persistent_parameters
        self._configure_offloading(offload_optimizer_config, offload_param_config)

        self.module = module
        self.elastic_checkpoint = elastic_checkpoint

        self.inf_or_nan_tracker: Tensor = torch.zeros(
            1,
            dtype=torch.bool,
            device=get_accelerator().current_device_name(),
            requires_grad=False)

        self.deepspeed_adam_offload = (self.offload_optimizer
                                       and type(init_optimizer) == DeepSpeedCPUAdam)

        self.device = get_accelerator().current_device_name(
        ) if not self.offload_optimizer else OffloadDeviceEnum.cpu
        ### streams used for overlapping computation with communication
        self.reduce_and_partition_stream = get_accelerator().Stream(
        ) if overlap_comm else get_accelerator().default_stream()

        ############################################################################

        self.n_caching_allocator_flushes = 0

        #-------------Stage 3 Setup-------------------#

        self.timers = timers

        self.reduce_scatter = reduce_scatter

        self.dp_process_group = dp_process_group

        self.partition_count = dist.get_world_size(group=self.dp_process_group)

        if mpu is None:
            self.model_parallel_group = None
            self.model_parallel_rank = 0
        else:
            self.model_parallel_group = mpu.get_model_parallel_group()
            self.model_parallel_rank = mpu.get_model_parallel_rank()

        self.overflow = False
        self.clip_grad = clip_grad
        self.communication_data_type = communication_data_type
        self.gradient_predivide_factor = gradient_predivide_factor
        self.postscale_gradients = postscale_gradients
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.micro_step_id = 0
        self.reduce_bucket_size = int(reduce_bucket_size)

        if self.reduce_scatter:
            assert self.communication_data_type in (torch.float16, torch.bfloat16), f"ZeRO-3 supports only float16 or bfloat16 communication_data_type with reduce scatter enabled. Got: '{self.communication_data_type}'"
            assert self.gradient_predivide_factor == 1.0, "gradient_predivide_factor != 1.0 is not yet supported with ZeRO-3 with reduce scatter enabled"
            assert self.postscale_gradients, "pre-scale gradients is not yet supported with ZeRO-3 with reduce scatter enabled"

        # Holds the mode parameter
        # The param.data may not hold any meaningful data
        # when param's status is NOT_AVAILABLE or IN_FLGHT
        self.fp16_groups = []

        # Hold partitioned parameters
        self.fp16_partitioned_groups = []

        # Holds a fused and flattened copy of the parameters
        self.fp16_partitioned_groups_flat = []
        self.fp16_partitioned_groups_flat_numel = []

        #defragmented pinned memory
        self.param_groups_fp16_flat_cpu_memory = []

        #a single 32-bit partition of the parallel partitioned parameters
        #that this process will update
        self.fp32_partitioned_groups_flat = []
        self.next_swappable_fp32_partitioned_groups = []

        # number of elements per partition in each group
        self.partition_size = []

        self.all_reduce_print = False

        self.prefetch_elements = int(prefetch_bucket_size)

        self.contiguous_gradients = contiguous_gradients

        # padding on each partition for alignment purposes
        self.groups_padding = []

        self.sub_group_size = sub_group_size

        self.sub_group_to_group_id = {}

        # Trainable parameters
        self.trainable_param_groups = self._get_trainable_parameter_groups()

        see_memory_usage("Before creating fp16 partitions", force=True)
        self._create_fp16_partitions_with_defragmentation(self.trainable_param_groups)
        num_fp16_subgroups = len(self.fp16_partitioned_groups_flat)
        see_memory_usage(f"After creating fp16 partitions: {num_fp16_subgroups}",
                         force=True)

        # Optimizer tensor swapping
        if self.swap_optimizer:
            self._configure_tensor_swapping(offload_optimizer_config, aio_config)

        self.params_in_ipg_bucket = []
        self.is_gradient_accumulation_boundary: bool = True

        self.param_reduce_events: Deque[get_accelerator().Event] = collections.deque()
        # TODO. make this configurable via JSON
        self.max_param_reduce_events: int = 2

        self.param_dict = {}

        # map between param_id and bool to specify if a param is in this partition
        self.is_param_in_current_partition = {}

        self.extra_large_param_to_reduce = None
        self.grads_in_ipg_bucket = []
        self.params_in_ipg_bucket = []

        self.params_already_reduced = []
        self.is_gradient_accumulation_boundary = True
        self._release_ipg_buffers()
        self.previous_reduced_grads = None

        # simplified param id
        self.param_id = {}

        count = 0
        for i, params_group in enumerate(self.fp16_groups):
            for param in params_group:
                unique_id = id(param)
                self.param_id[unique_id] = count
                self.param_dict[count] = param
                self.params_already_reduced.append(False)
                count = count + 1

        #Largest partitioned param
        largest_partitioned_param_numel = max([
            max([
                max(tensor.numel(),
                    tensor.ds_numel) for tensor in fp16_partitioned_group
            ]) for fp16_partitioned_group in self.fp16_partitioned_groups
        ])
        print_rank_0(
            f'Largest partitioned param numel = {largest_partitioned_param_numel}',
            force=False)

        self._setup_for_real_optimizer()
        self.grad_position = {}
        self.set_grad_positions()

        if self.offload_optimizer:
            self.norm_for_param_grads = {}
            self.local_overflow = False

        # stores if a partition has been reduced in this step
        self.is_partition_reduced = {}

        # stores if a grad in a partition has been computed or not
        self.is_grad_computed = {}

        # will store the averaged gradients required by this partition
        self.averaged_gradients = {}

        #creates backward hooks for gradient partitioning
        self.create_reduce_and_remove_grad_hooks()

        #exit(0)

        # we may have a way of fusing dynamic scale. Do not support for now
        if self.dtype == torch.float or not dynamic_loss_scale:
            loss_scale_value = 1.0 if self.dtype == torch.float else static_loss_scale

            self.dynamic_loss_scale = False
            self.loss_scaler = LossScaler(scale=loss_scale_value)
        else:
            if dynamic_loss_args is None:
                self.loss_scaler = DynamicLossScaler()
            else:
                self.loss_scaler = DynamicLossScaler(**dynamic_loss_args)

            self.dynamic_loss_scale = True

        self.debug_fp16_grads = [{} for _ in self.fp16_groups]

        if dist.get_rank(group=self.dp_process_group) == 0:
            see_memory_usage(f"After initializing ZeRO optimizer", force=True)


    # TODO: Support different/changing load/save DP degree.
    def load_state_dict(self,
                        state_dict_list,
                        load_optimizer_states=True,
                        load_from_fp32_weights=False,
                        checkpoint_folder=None):
        r""" Loading the MiCS checkpoints

        TODO: move the implementation from zhen/merged_ds_master branch
        """
        raise NotImplementedError("Not implemented for loading MiCS checkpoints")
    
