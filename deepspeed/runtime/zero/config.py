"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional
from enum import Enum
from pathlib import Path
from deepspeed.runtime.config_utils import get_scalar_param, DeepSpeedConfigObject
from deepspeed.utils import logger
from .constants import *
from .offload_constants import *
from .offload_config import get_offload_param_config, get_default_offload_param_config, \
    get_offload_optimizer_config, get_default_offload_optimizer_config


class DeprecatedParameterError(Exception):
    pass


class DeepSpeedConfigModel(BaseModel):
    def __init__(self, **data):
        super().__init__(**data)
        self.deprecated_fields_check(self.__fields_set__)

    @validator('*')
    @classmethod
    def deprecated_field_check(cls, field_value, values, field, config):
        kwargs = field.field_info.extra
        if 'deprecated' in kwargs and kwargs['deprecated'] == True:
            if 'deprecated_msg' not in kwargs:
                msg = f"Parameter '{field.name}' is deprecated"
                if 'new_param' in kwargs:
                    msg += f", replaced by '{kwargs['new_param']}'"
            else:
                msg = kwargs['deprecated_msg']
            logger.warning('DeepSpeedZeroConfig:' + msg)
            if 'new_param' in kwargs:
                if kwargs['new_param'] not in values:
                    raise DeprecatedParameterError(
                        f"Deprecated parameter '{field.name}' should be defined after the replacing parameter '{kwargs['new_param']}'"
                    )
                values[kwargs['new_param']] = field_value
        return field_value

    @classmethod
    def deprecated_fields_check(self, fields_set):
        for field in self.__fields__.values():
            kwargs = field.field_info.extra
            if 'deprecated' in kwargs and kwargs['deprecated']:
                if field.name in fields_set and kwargs['new_param'] in fields_set:
                    raise DeprecatedParameterError(
                        f"Deprecated param '{field.name}' and new parameter '{kwargs['new_param']}' cannot be provided together"
                    )

    class Config:
        validate_all = True
        validate_assignment = True
        use_enum_values = True
        extra = 'forbid'


class OffloadDeviceEnum(str, Enum):
    none: 'none'
    cpu: 'cpu'
    nvme: 'nvme'


class DeepSpeedZeroOffloadParamConfig(DeepSpeedConfigModel):
    device: OffloadDeviceEnum = OffLoadDeviceEnum.none
    nvme_path: Path = None
    buffer_count: int = Field(5, ge=0)
    buffer_size: int = Field(1e8, ge=0)
    max_in_cpu: int = Field(1e9, ge=0)
    pin_memory: bool = False


class DeepSpeedZeroOffloadOptimizerConfig(DeepSpeedConfigModel):
    device: OffloadDeviceEnum = OffLoadDeviceEnum.none
    nvme_path: Path = None
    buffer_count: int = Field(4, ge=0)
    pin_memory: bool = False
    pipeline_read: bool = False
    pipeline_write: bool = False
    fast_init: bool = False


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

    #Offload Specific Parameters
    offload_param: Optional[DeepSpeedZeroOffloadParamConfig] = None
    offload_optimizer: Optional[DeepSpeedZeroOffloadOptimizerConfig] = None
    sub_group_size: int = 1e9
    cpu_offload_param: bool = Field(None, deprecated=True)
    cpu_offload_use_pin_memory: bool = Field(None, deprecated=True)
    cpu_offload: bool = Field(None, deprecated=True)

    #Stage3 Specific Parameters
    prefect_bucket_size: int = 5e7
    param_persistence_threshold: int = 1e5
    max_live_parameters: int = 1e9
    max_reuse_distance: int = 1e9
    gather_16bit_weights_on_model_save: bool = False
    stage3_gather_fp16_weights_on_model_save: Field(
        False,
        deprecated=True,
        new_param='gather_16bit_weights_on_model_save')

    ignore_unused_parameters: bool = True
    round_robin_gradients: bool = False

    @validate('overlap_comm')
    @classmethod
    def overlap_comm_valid(cls, field_value, values):
        if field_value is None:
            field_value = (values['stage'] == 3)
        return field_value
