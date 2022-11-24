"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from deepspeed.runtime.config_utils import DeepSpeedConfigModel

COMMS_LOGGER = "comms_logger"


def get_comms_config(param_dict):
    comms_config_dict = param_dict.get(COMMS_LOGGER, {})
    return DeepSpeedCommsConfig(**comms_config_dict)


class DeepSpeedCommsConfig(DeepSpeedConfigModel):
    enabled: bool = False
    prof_all: bool = True
    prof_ops: list = []
    verbose: bool = False
    debug: bool = False
