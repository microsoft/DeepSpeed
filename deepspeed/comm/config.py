"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from deepspeed.runtime.config_utils import DeepSpeedConfigModel


def get_comms_config(param_dict):
    return DeepSpeedCommsConfig(**param_dict.get("comms_logger", {}))


class DeepSpeedCommsConfig(DeepSpeedConfigModel):
    enabled: bool = False
    prof_all: bool = True
    prof_ops: list = []
    verbose: bool = False
    debug: bool = False
