'''Copyright The Microsoft DeepSpeed Team'''
"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from pydantic import Field
from deepspeed.runtime.config_utils import DeepSpeedConfigModel

ACT_CHKPT = 'activation_checkpointing'


def get_activation_checkpointing_config(param_dict):
    act_chkpt_config_dict = param_dict.get(ACT_CHKPT, {})
    return DeepSpeedActivationCheckpointingConfig(**act_chkpt_config_dict)


class DeepSpeedActivationCheckpointingConfig(DeepSpeedConfigModel):
    partition_activations: bool = False
    contiguous_memory_optimization: bool = False
    cpu_checkpointing: bool = False
    number_checkpoints: int = Field(None, gt=0)
    synchronize_checkpoint_boundary: bool = False
    profile: bool = False
