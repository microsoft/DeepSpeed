"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from pydantic import Field
from deepspeed.runtime.config_utils import DeepSpeedConfigModel
from enum import Enum

AUTOTUNING = "autotuning"


def get_autotuning_config(param_dict):
    autotuning_config_dict = param_dict.get(AUTOTUNING, {})
    return DeepSpeedAutotuningConfig(**autotuning_config_dict)


class MetricEnum(str, Enum):
    latency = "latency"
    throughput = "throughput"
    flops = "flops"
    forward = "forward"
    backward = "backward"
    steps = "steps"


class TunerTypeEnum(str, Enum):
    gridsearch = "gridsearch"
    random = "random"
    model_based = "model_based"


class ModelInfoConfig(DeepSpeedConfigModel):
    profile: bool = False
    num_params: int = Field(None, ge=0)
    hidden_size: int = Field(None, ge=0)
    num_layers: int = Field(None, ge=0)


class DeepSpeedAutotuningConfig(DeepSpeedConfigModel):
    enabled: bool = False
    fast: bool = True
    results_dir: str = None  # Should this be Path dtype?
    exps_dir: str = None
    overwrite: bool = True
    start_step: int = Field(3, ge=0, alias="start_profile_step")
    end_step: int = Field(5, ge=0, alias="end_profile_step")
    metric: MetricEnum = MetricEnum.throughput
    metric_path: str = None
    tuner_type: TunerTypeEnum = TunerTypeEnum.gridsearch
    tuner_early_stopping: int = Field(5, ge=0)
    tuner_num_trials: int = Field(50, gt=0)
    arg_mapping: str = None
    model_info: ModelInfoConfig = ModelInfoConfig()
    model_info_path: str = None
    mp_size: int = Field(1, gt=0)
    max_train_batch_size: int = Field(None, gt=0)
    min_train_batch_size: int = Field(1, gt=0)
    max_train_micro_batch_size_per_gpu: int = Field(1024, gt=0)
    min_train_micro_batch_size_per_gpu: int = Field(1, gt=0)
    num_tuning_micro_batch_sizes: int = Field(3, gt=0)
