"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from pydantic import Field
from deepspeed.runtime.config_utils import DeepSpeedConfigModel

NEBULA = "nebula"


def get_nebula_config(param_dict):
    nebula_config_dict = param_dict.get(NEBULA, {})
    return DeepSpeedNebulaConfig(**nebula_config_dict)


class DeepSpeedNebulaConfig(DeepSpeedConfigModel):
    enabled: bool = False
    load_path: str = None
    enable_nebula_load: bool = True
    persistent_storage_path: str = None  # Should this be Path dtype?
    persistent_time_interval: int = Field(100, gt=0)
    num_of_version_in_retention: int = Field(2, ge=0)
