"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from deepspeed.runtime.config_utils import DeepSpeedConfigModel


class TensorBoardConfig(DeepSpeedConfigModel):
    enabled: bool = False
    output_path: str = ""
    job_name: str = "DeepSpeedJobName"


class WandbConfig(DeepSpeedConfigModel):
    enabled: bool = False
    group: str = None
    team: str = None
    project: str = "deepspeed"


class CSVConfig(DeepSpeedConfigModel):
    enabled: bool = False
    output_path: str = ""
    job_name: str = "DeepSpeedJobName"
