"""
"Copyright 2020 The Microsoft DeepSpeed Team.
Licensed under the MIT license.
"""

import sys
import torch

from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from deepspeed.utils import log_dist


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
