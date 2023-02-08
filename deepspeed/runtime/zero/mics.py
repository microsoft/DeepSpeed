"""
"Copyright 2020 The Microsoft DeepSpeed Team.
Licensed under the MIT license.
"""

import sys
from typing import List

import torch
from deepspeed import comm as dist
from deepspeed.runtime.config import DeepSpeedConfig
from deepspeed.runtime.zero.mics_utils import (MiCS_CommGroups,
                                               create_mics_comm_groups,
                                               scale_tensors)
from deepspeed.runtime.zero.parameter_offload import (DeepSpeedZeRoOffload,
                                                      is_zero_param)
from deepspeed.runtime.zero.partition_parameters import Init
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from deepspeed.utils import instrument_w_nvtx, log_dist
from torch import Tensor
from torch.nn import Parameter


class MiCS_Init(Init):
    """"""
    def __init__(self,
                 module=None,
                 data_parallel_group=None,
                 mem_efficient_linear=True,
                 remote_device=None,
                 pin_memory=False,
                 config_dict_or_path=None,
                 config=None,
                 enabled=True,
                 dtype=None,
                 mpu=None):
        assert config_dict_or_path is not None, "Must provide configuration for MiCS Initialization"
        _ds_config = DeepSpeedConfig(config_dict_or_path, mpu)
        if not dist.is_initialized():
            dist.init_distributed()
            assert dist.is_initialized(), "Parameters cannot be scattered without initializing deepspeed.comm"
        self.mics_comm_groups = create_mics_comm_groups(
            _ds_config.mics_shard_size,
            data_parallel_group,
            hierarchical_allgather=_ds_config.mics_hierarchial_params_gather,
            mpu=mpu)

        super().__init__(module,
                         data_parallel_group,
                         mem_efficient_linear,
                         remote_device,
                         pin_memory,
                         config_dict_or_path,
                         config,
                         enabled,
                         dtype,
                         mpu)

    def _convert_to_deepspeed_param(self, param):
        super()._convert_to_deepspeed_param(param)
        param.comm = self.mics_comm_groups

    def get_partition_dp_group(self, param):
        return param.comm.param_shard_group

    def get_partition_rank(self):
        return self.mics_comm_groups.param_shard_rank

    @property
    def num_partitions(self):
        return self.mics_comm_groups.param_shard_size


class MiCS_Offload(DeepSpeedZeRoOffload):
    """ Wrapper to change the behavior for parameter sharding
    """
    def __init__(self,
                 module,
                 timers,
                 ds_config,
                 overlap_comm=True,
                 prefetch_bucket_size=50000000,
                 max_reuse_distance=1000000000,
                 max_live_parameters=1000000000,
                 param_persistence_threshold=100000,
                 model_persistence_threshold=sys.maxsize,
                 offload_param_config=None,
                 mpu=None):
        super().__init__(module,
                         timers,
                         ds_config,
                         overlap_comm,
                         prefetch_bucket_size,
                         max_reuse_distance,
                         max_live_parameters,
                         param_persistence_threshold,
                         model_persistence_threshold,
                         offload_param_config,
                         mpu)

    def _convert_to_zero_parameters(self, ds_config, module, mpu):
        """ overload the parent class function for convert the parameters

        """
        log_dist(f'Convert to zero parameters from MiCS Offload manager', ranks=[0])
        non_zero_params = [p for p in module.parameters() if not is_zero_param(p)]
        if non_zero_params:
            zero_params = [p for p in module.parameters() if is_zero_param(p)]
            if zero_params:
                zero_params[0].convert_to_zero_parameters(param_list=non_zero_params)
            else:
                group = None
                if mpu:
                    group = mpu.get_data_parallel_group()

                MiCS_Init(module=module,
                          data_parallel_group=group,
                          dtype=self.dtype,
                          config_dict_or_path=ds_config,
                          remote_device=self.offload_device,
                          pin_memory=self.offload_param_pin_memory,
                          mpu=mpu)


class MiCS_Optimizer(DeepSpeedZeroOptimizer_Stage3):
    """
    MiCS Optimizer
    """
    def __init__(self,
                 module,
                 init_optimizer,
                 timers,
                 ds_config,
                 static_loss_scale=1,
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
                 clip_grad=0,
                 communication_data_type=torch.float16,
                 postscale_gradients=True,
                 gradient_predivide_factor=1,
                 gradient_accumulation_steps=1,
                 elastic_checkpoint=False,
                 aio_config=None):

        log_dist("Init MiCS optimizer", ranks=[0])
        super().__init__(module,
                         init_optimizer,
                         timers,
                         ds_config,
                         static_loss_scale,
                         dynamic_loss_scale,
                         dynamic_loss_args,
                         verbose,
                         contiguous_gradients,
                         reduce_bucket_size,
                         prefetch_bucket_size,
                         max_reuse_distance,
                         max_live_parameters,
                         param_persistence_threshold,
                         model_persistence_threshold,
                         dp_process_group,
                         reduce_scatter,
                         overlap_comm,
                         offload_optimizer_config,
                         offload_param_config,
                         sub_group_size,
                         mpu,
                         clip_grad,
                         communication_data_type,
                         postscale_gradients,
                         gradient_predivide_factor,
                         gradient_accumulation_steps,
                         elastic_checkpoint,
                         aio_config)
        first_param = next(module.parameters())
        # overload the dp_process_group and partition_count
        self.dp_process_group = first_param.comm.param_shard_group
        self.partition_count = first_param.comm.param_shard_size

    def initialize_ds_offload(self,
                              module,
                              timers,
                              ds_config,
                              overlap_comm,
                              prefetch_bucket_size,
                              max_reuse_distance,
                              max_live_parameters,
                              param_persistence_threshold,
                              model_persistence_threshold,
                              offload_optimizer_config,
                              mpu):
        return MiCS_Offload(module,
                            timers,
                            ds_config,
                            overlap_comm,
                            prefetch_bucket_size,
                            max_reuse_distance,
                            max_live_parameters,
                            param_persistence_threshold,
                            model_persistence_threshold,
                            offload_optimizer_config,
                            mpu)

    def partition_grads(self,
                        params_to_release: List[Parameter],
                        grad_partitions: List[Tensor]) -> None:
        grad_buffers = super().partition_grads(params_to_release, grad_partitions)
        # perform all-reduce among replication groups
        # the function will perform accumulation boundary check
        self.allreduce_mics_shard_grads(params_to_release, grad_buffers)

    @instrument_w_nvtx
    def allreduce_mics_shard_grads(self,
                                   params,
                                   partitioned_grads_buffers: List[Tensor]):
        """
        """
        # TODO: improve the condition check
        if not self.is_gradient_accumulation_boundary or \
            len(partitioned_grads_buffers) == 0:
            return

        mics_comm_groups: MiCS_CommGroups = params[0].comm
        param_repli_group = mics_comm_groups.param_repli_group
        param_repli_size = mics_comm_groups.param_repli_size

        if param_repli_size is None or param_repli_size <= 1:
            return
        if not partitioned_grads_buffers[0].is_cuda:
            raise RuntimeError("Local sharding has no support for CPU offloading")

        if dist.has_all_reduce_coalesced():
            scale_tensors(partitioned_grads_buffers, param_repli_size)
            dist.all_reduce_coalesced(tensors=partitioned_grads_buffers,
                                      group=param_repli_group)
        else:
            # manually coalescing all-reduce
            aggregated_buffer: Tensor = torch.cat(partitioned_grads_buffers)
            aggregated_buffer.div_(param_repli_size)
            dist.all_reduce(aggregated_buffer, group=param_repli_group)
            offset = 0
            for grad_buff in partitioned_grads_buffers:
                grad_buff.view(-1).copy_(
                    aggregated_buffer.narrow(0,
                                             offset,
                                             grad_buff.numel()))
                offset += grad_buff.numel()

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
