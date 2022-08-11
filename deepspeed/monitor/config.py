"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from types import SimpleNamespace
from deepspeed.runtime.config_utils import DeepSpeedConfigModel

TENSORBOARD = "tensorboard"
WANDB = "wandb"
CSV_MONITOR = "csv_monitor"


def get_monitor_config(param_dict):
    tensorboard_config_dict = param_dict.get(TENSORBOARD, {})
    wandb_config_dict = param_dict.get(WANDB, {})
    csv_monitor_config_dict = param_dict.get(CSV_MONITOR, {})
    monitor_config = SimpleNamespace(
        tensorboard=TensorBoardConfig(**tensorboard_config_dict),
        wandb=WandbConfig(**wandb_config_dict),
        csv_monitor=CSVMonitorConfig(**csv_monitor_config_dict))
    return monitor_config


class TensorBoardConfig(DeepSpeedConfigModel):
    enabled: bool = False
    output_path: str = ""
    job_name: str = "DeepSpeedJobName"


class WandbConfig(DeepSpeedConfigModel):
    enabled: bool = False
    group: str = None
    team: str = None
    project: str = "deepspeed"


class CSVMonitorConfig(DeepSpeedConfigModel):
    enabled: bool = False
    output_path: str = ""
    job_name: str = "DeepSpeedJobName"
