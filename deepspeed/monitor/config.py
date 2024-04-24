# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Optional, Literal

from deepspeed.pydantic_v1 import root_validator
from deepspeed.runtime.config_utils import DeepSpeedConfigModel


def get_monitor_config(param_dict):
    monitor_dict = {key: param_dict.get(key, {}) for key in ("tensorboard", "wandb", "csv_monitor", "comet")}
    return DeepSpeedMonitorConfig(**monitor_dict)


class TensorBoardConfig(DeepSpeedConfigModel):
    """Sets parameters for TensorBoard monitor."""

    enabled: bool = False
    """ Whether logging to Tensorboard is enabled. Requires `tensorboard` package is installed. """

    output_path: str = ""
    """
    Path to where the Tensorboard logs will be written. If not provided, the
    output path is set under the training script’s launching path.
    """

    job_name: str = "DeepSpeedJobName"
    """ Name for the current job. This will become a new directory inside `output_path`. """


class WandbConfig(DeepSpeedConfigModel):
    """Sets parameters for WandB monitor."""

    enabled: bool = False
    """ Whether logging to WandB is enabled. Requires `wandb` package is installed. """

    group: str = None
    """ Name for the WandB group. This can be used to group together runs. """

    team: str = None
    """ Name for the WandB team. """

    project: str = "deepspeed"
    """ Name for the WandB project. """


class CSVConfig(DeepSpeedConfigModel):
    """Sets parameters for CSV monitor."""

    enabled: bool = False
    """ Whether logging to local CSV files is enabled. """

    output_path: str = ""
    """
    Path to where the csv files will be written. If not provided, the output
    path is set under the training script’s launching path.
    """

    job_name: str = "DeepSpeedJobName"
    """ Name for the current job. This will become a new directory inside `output_path`. """


class CometConfig(DeepSpeedConfigModel):
    """Sets parameters for Comet monitor."""

    enabled: bool = False
    """ Whether logging to Comet is enabled. Requires `comet` package is installed. """
    
    samples_log_interval: int = 100

    project: Optional[str] = None
    """ Comet project name. If no name provided - 'general' will be used """

    workspace: Optional[str] = None
    """ Comet workspace name. If no provided - the default for the user will be used"""

    api_key: Optional[str] = None

    experiment_key: Optional[str] = None

    online: bool = True

    mode: Literal["get_or_create", "create", "get"] = None




class DeepSpeedMonitorConfig(DeepSpeedConfigModel):
    """Sets parameters for various monitoring methods."""

    tensorboard: TensorBoardConfig = {}
    """ TensorBoard monitor, requires `tensorboard` package is installed. """

    comet: CometConfig = {}
    """ Comet monitor, requires comet_ml package is installed """

    wandb: WandbConfig = {}
    """ WandB monitor, requires `wandb` package is installed. """

    csv_monitor: CSVConfig = {}
    """ Local CSV output of monitoring data. """

    @root_validator
    def check_enabled(cls, values):
        values["enabled"] = values.get("tensorboard").enabled or values.get("wandb").enabled or values.get(
            "csv_monitor").enabled or values.get("comet")
        return values
