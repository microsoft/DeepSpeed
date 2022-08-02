"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from pydantic import Field
from deepspeed.runtime.config_utils import get_scalar_param, DeepSpeedConfigModel


class DeepSpeedActivationCheckpointingConfig(DeepSpeedConfigModel):
    partition_activations: bool = False
    contiguous_memory_optimization: bool = False
    cpu_checkpointing: bool = False
    number_checkpoints: int = Field(None, gt=0)
    synchronize_checkpoint_boundary: bool = False
    profile: bool = False
