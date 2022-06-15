"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional
from enum import Enum
from pathlib import Path
from deepspeed.runtime.config_utils import get_scalar_param, DeepSpeedConfigModel
from deepspeed.utils import logger

# ZeRO optimization. By default, this optimization is not enabled.
# Users have to configure the desired optimization (0 means disabled) in params.json as below example:
ZERO_FORMAT = '''
ZeRO optimization should be enabled as:
"session_params": {
  "zero_optimization": {
    "stage": [0|1|2],
    "stage3_max_live_parameters" : 1000000000,
    "stage3_max_reuse_distance" : 1000000000,
    "allgather_partitions": [true|false],
    "allgather_bucket_size": 500000000,
    "reduce_scatter": [true|false],
    "contiguous_gradients" : [true|false]
    "overlap_comm": [true|false],
    "reduce_bucket_size": 500000000,
    "load_from_fp32_weights": [true|false],
    "cpu_offload": [true|false] (deprecated),
    "cpu_offload_params" : [true|false] (deprecated),
    "cpu_offload_use_pin_memory": [true|false] (deprecated),
    "sub_group_size" : 1000000000000,
    "offload_param": {...},
    "offload_optimizer": {...},
    "ignore_unused_parameters": [true|false],
    "round_robin_gradients": [true|false]
    }
}
'''

ZERO_OPTIMIZATION = 'zero_optimization'
ZERO_OPTIMIZATION_DISABLED = 0
ZERO_OPTIMIZATION_OPTIMIZER_STATES = 1
ZERO_OPTIMIZATION_GRADIENTS = 2
ZERO_OPTIMIZATION_WEIGHTS = 3
MAX_STAGE_ZERO_OPTIMIZATION = ZERO_OPTIMIZATION_WEIGHTS


def read_zero_config_deprecated(self, param_dict):
    zero_config_dict = {}
    zero_config_dict['stage'] = (1 if param_dict[ZERO_OPTIMIZATION] else 0)
    if zero_config_dict['stage'] > 0:
        zero_config_dict['allgather_bucket_size'] = get_scalar_param(
            param_dict,
            'allgather_size',
            5e8)
    logger.warning(
        'DeepSpeedConfig: this format of ZeRO optimization setup is deprecated. Please use the following format: {}'
        .format(ZERO_FORMAT))
    return zero_config_dict


class OffloadDeviceEnum(str, Enum):
    none = "none"
    cpu = "cpu"
    nvme = "nvme"


class DeepSpeedZeroOffloadParamConfig(DeepSpeedConfigModel):
    device: OffloadDeviceEnum = OffloadDeviceEnum.none
    nvme_path: Path = None
    buffer_count: int = Field(5, ge=0)
    buffer_size: int = Field(1e8, ge=0)
    max_in_cpu: int = Field(1e9, ge=0)
    pin_memory: bool = False


class DeepSpeedZeroOffloadOptimizerConfig(DeepSpeedConfigModel):
    device: OffloadDeviceEnum = OffloadDeviceEnum.none
    nvme_path: Path = None
    buffer_count: int = Field(4, ge=0)
    pin_memory: bool = False
    pipeline_read: bool = False
    pipeline_write: bool = False
    fast_init: bool = False

    @validator("pipeline_read", "pipeline_write")
    @classmethod
    def set_pipeline(cls, field_value, values):
        if field_value or values.get("pipeline", False):
            values["pipeline"] = True
        return field_value


class DeepSpeedZeroConfig(DeepSpeedConfigModel):
    stage: int = Field(0, le=3, ge=0)
    contiguous_gradients: bool = True
    reduce_scatter: bool = True
    reduce_bucket_size: int = 5e8
    allgather_partitions: bool = True
    allgather_bucket_size: int = 5e8
    overlap_comm: bool = None  # None for dynamic default value
    load_from_fp32_weights: bool = True

    elastic_checkpoint: bool = False

    # Offload Specific Parameters
    offload_param: Optional[DeepSpeedZeroOffloadParamConfig] = None
    offload_optimizer: Optional[DeepSpeedZeroOffloadOptimizerConfig] = None
    sub_group_size: int = 1e9
    cpu_offload_param: bool = Field(None, deprecated=True)
    cpu_offload_use_pin_memory: bool = Field(None, deprecated=True)
    cpu_offload: bool = Field(None, deprecated=True)

    # Stage3 Specific Parameters
    prefetch_bucket_size: int = 5e7
    param_persistence_threshold: int = Field(1e5,
                                             alias='stage3_param_persistence_threshold')
    max_live_parameters: int = 1e9
    max_reuse_distance: int = 1e9
    gather_16bit_weights_on_model_save: bool = False
    stage3_gather_fp16_weights_on_model_save: bool = Field(
        False,
        deprecated=True,
        new_param="gather_16bit_weights_on_model_save")

    ignore_unused_parameters: bool = True
    legacy_stage1: bool = False
    round_robin_gradients: bool = False

    @validator("overlap_comm")
    @classmethod
    def overlap_comm_valid(cls, field_value, values):
        if field_value is None:
            assert "stage" in values, "DeepSpeedZeroConfig: 'stage' must be defined before 'overlap_comm'"
            field_value = values["stage"] == 3
        return field_value
