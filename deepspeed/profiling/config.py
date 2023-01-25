"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from pydantic import Field
from deepspeed.runtime.config_utils import DeepSpeedConfigModel

FLOPS_PROFILER = "flops_profiler"


def get_flops_profiler_config(param_dict):
    flops_profiler_config_dict = param_dict.get(FLOPS_PROFILER, {})
    return DeepSpeedFlopsProfilerConfig(**flops_profiler_config_dict)


class DeepSpeedFlopsProfilerConfig(DeepSpeedConfigModel):
    enabled: bool = False
    profile_step: int = Field(1, ge=1)
    module_depth: int = -1
    top_modules: int = 1
    detailed: bool = True
    output_file: str = None
