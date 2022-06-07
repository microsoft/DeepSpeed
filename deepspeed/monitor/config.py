"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from deepspeed.runtime.config_utils import get_scalar_param, DeepSpeedConfigObject
from .constants import *


class DeepSpeedMonitorConfig(DeepSpeedConfigObject):
    def __init__(self, param_dict):
        super(DeepSpeedMonitorConfig, self).__init__()

        self.tensorboard_enabled = self.get_tensorboard_enabled(param_dict)
        self.tensorboard_output_path = self.get_tensorboard_output_path(param_dict)
        self.tensorboard_job_name = self.get_tensorboard_job_name(param_dict)
        self.wandb_enabled = self.get_wandb_enabled(param_dict)
        self.wandb_group = self.get_wandb_group(param_dict)
        self.wandb_team = self.get_wandb_team(param_dict)
        self.wandb_project = self.get_wandb_project(param_dict)
        self.wandb_host = self.get_wandb_host(param_dict)
        self.csv_monitor_enabled = self.get_csv_monitor_enabled(param_dict)
        self.csv_monitor_output_path = self.get_csv_monitor_output_path(param_dict)

    def get_csv_monitor_enabled(self, param_dict):
        if CSV_MONITOR in param_dict.keys():
            return get_scalar_param(param_dict[CSV_MONITOR],
                                    CSV_MONITOR_ENABLED,
                                    CSV_MONITOR_ENABLED_DEFAULT)
        else:
            return False

    def get_tensorboard_enabled(self, param_dict):
        if TENSORBOARD in param_dict.keys():
            return get_scalar_param(param_dict[TENSORBOARD],
                                    TENSORBOARD_ENABLED,
                                    TENSORBOARD_ENABLED_DEFAULT)
        else:
            return False

    def get_wandb_enabled(self, param_dict):
        if WANDB in param_dict.keys():
            return get_scalar_param(param_dict[WANDB],
                                    WANDB_ENABLED,
                                    WANDB_ENABLED_DEFAULT)
        else:
            return False

    def get_tensorboard_output_path(self, param_dict):
        if self.get_tensorboard_enabled(param_dict):
            return get_scalar_param(
                param_dict[TENSORBOARD],
                TENSORBOARD_OUTPUT_PATH,
                TENSORBOARD_OUTPUT_PATH_DEFAULT,
            )
        else:
            return TENSORBOARD_OUTPUT_PATH_DEFAULT

    def get_tensorboard_job_name(self, param_dict):
        if self.get_tensorboard_enabled(param_dict):
            return get_scalar_param(param_dict[TENSORBOARD],
                                    TENSORBOARD_JOB_NAME,
                                    TENSORBOARD_JOB_NAME_DEFAULT)
        else:
            return TENSORBOARD_JOB_NAME_DEFAULT

    def get_wandb_team(self, param_dict):
        if self.get_wandb_enabled(param_dict):
            return get_scalar_param(param_dict[WANDB],
                                    WANDB_TEAM_NAME,
                                    WANDB_TEAM_NAME_DEFAULT)
        else:
            return WANDB_TEAM_NAME_DEFAULT

    def get_wandb_project(self, param_dict):
        if self.get_wandb_enabled(param_dict):
            return get_scalar_param(param_dict[WANDB],
                                    WANDB_PROJECT_NAME,
                                    WANDB_PROJECT_NAME_DEFAULT)
        else:
            return WANDB_PROJECT_NAME_DEFAULT

    def get_wandb_group(self, param_dict):
        if self.get_wandb_enabled(param_dict):
            return get_scalar_param(param_dict[WANDB],
                                    WANDB_GROUP_NAME,
                                    WANDB_GROUP_NAME_DEFAULT)
        else:
            return WANDB_GROUP_NAME_DEFAULT

    def get_wandb_host(self, param_dict):
        if self.get_wandb_enabled(param_dict):
            return get_scalar_param(param_dict[WANDB],
                                    WANDB_HOST_NAME,
                                    WANDB_HOST_NAME_DEFAULT)
        else:
            return WANDB_HOST_NAME_DEFAULT

    def get_csv_monitor_output_path(self, param_dict):
        if self.get_csv_monitor_enabled(param_dict):
            return get_scalar_param(
                param_dict[CSV_MONITOR],
                CSV_MONITOR_OUTPUT_PATH,
                CSV_MONITOR_OUTPUT_PATH_DEFAULT,
            )
        else:
            return CSV_MONITOR_OUTPUT_PATH_DEFAULT
