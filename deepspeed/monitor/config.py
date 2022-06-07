"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from typing import Optional
from deepspeed.runtime.config_utils import get_scalar_param
from pydantic import BaseModel, validator, ValidationError, create_model
from .constants import *


class DeepSpeedMonitorConfig(BaseModel):

    tensorboard: Optional[create_model(
        'TensorBoard',
        enabled=(bool,
                 TENSORBOARD_ENABLED_DEFAULT),
        output_path=(str,
                     TENSORBOARD_OUTPUT_PATH_DEFAULT),
        job_name=(str,
                  TENSORBOARD_JOB_NAME_DEFAULT))]  #, __validators__=tb_validators)]
    wandb: Optional[create_model('Wandb',
                                 enabled=(bool,
                                          WANDB_ENABLED_DEFAULT),
                                 group=(str,
                                        WANDB_GROUP_NAME_DEFAULT),
                                 team=(str,
                                       WANDB_TEAM_NAME_DEFAULT),
                                 project=(str,
                                          WANDB_PROJECT_NAME_DEFAULT),
                                 host=(str,
                                       WANDB_HOST_NAME_DEFAULT))]
    csv_monitor: Optional[create_model('csv_Monitor',
                                       enabled=(bool,
                                                CSV_MONITOR_ENABLED_DEFAULT),
                                       output_path=(str,
                                                    CSV_MONITOR_OUTPUT_PATH_DEFAULT))]

    class Config:
        validate_all = True
        validate_assignment = True
        use_enum_values = True
        #extra = 'forbid'
