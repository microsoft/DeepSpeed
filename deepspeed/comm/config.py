"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from deepspeed.runtime.config_utils import DeepSpeedConfigModel


class DeepSpeedCommsConfig(DeepSpeedConfigModel):
    enabled: bool = False
    prof_all: bool = True
    prof_ops: list = []
    verbose: bool = False
    debug: bool = False
