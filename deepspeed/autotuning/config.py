'''Copyright The Microsoft DeepSpeed Team'''
"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from typing import Dict
from pydantic import Field, validator, root_validator
from enum import Enum
from deepspeed.runtime.config_utils import DeepSpeedConfigModel


def get_autotuning_config(param_dict):
    return DeepSpeedAutotuningConfig(**param_dict.get("autotuning", {}))


class AutotuningMetricEnum(str, Enum):
    latency = "latency"
    throughput = "throughput"
    flops = "flops"
    forward = "forward"
    steps = "steps"


class AutotuningTunerEnum(str, Enum):
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
    results_dir: str = "autotuning_results"
    exps_dir: str = "autotuning_exps"
    overwrite: bool = True
    start_profile_step: int = Field(3, ge=0)
    end_profile_step: int = Field(5, ge=0)
    metric: AutotuningMetricEnum = "throughput"
    metric_path: str = None
    tuner_type: AutotuningTunerEnum = "gridsearch"
    tuner_num_trials: int = Field(50, ge=0)
    tuner_early_stopping: int = Field(5, ge=0)
    arg_mappings: Dict[str, str] = None
    model_info: ModelInfoConfig = {}
    model_info_path: str = None
    mp_size: int = Field(1, ge=1)
    max_train_batch_size: int = Field(None, ge=1)
    min_train_batch_size: int = Field(1, ge=1)
    max_train_micro_batch_size_per_gpu: int = Field(1024, ge=1)
    min_train_micro_batch_size_per_gpu: int = Field(1, ge=1)
    num_tuning_micro_batch_sizes: int = Field(3, ge=1)

    @validator("results_dir", "exps_dir")
    def assert_non_empty_str(cls, field_value, values):
        assert field_value != "", "field cannot by empty"
        return field_value

    @root_validator
    def check_profile_start_end(cls, values):
        start_step = values.get("start_profile_step")
        end_step = values.get("end_profile_step")
        assert start_step <= end_step, f"start_profiling_step ({start_step}) cannot be greater than end_profiling_step ({end_step})"
        return values

    @root_validator
    def check_min_max_batch_sizes(cls, values):
        max_batch = values.get("max_train_batch_size")
        min_batch = values.get("min_train_batch_size")
        assert min_batch <= max_batch, f"min_train_batch_size ({min_batch}) cannot be greater than max_train_batch_size ({max_batch})"

        max_micro_batch = values.get("max_train_micro_batch_size")
        min_micro_batch = values.get("min_train_micro_batch_size")
        assert min_micro_batch <= max_micro_batch, f"min_train_micro_batch_size ({min_micro_batch}) cannot be greater than max_train_micro_batch_size ({max_micro_batch})"
        return values
