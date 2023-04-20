# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.runtime.config_utils import DeepSpeedConfigModel


def get_comms_config(param_dict):
    return DeepSpeedCommsConfig(**param_dict.get("comms_logger", {}))


class DeepSpeedCommsConfig(DeepSpeedConfigModel):
    enabled: bool = False
    """ Whether communication logging is enabled. """

    prof_all: bool = True
    """ Whether to profile all operations. """

    prof_ops: list = []
    """ A list of communication operations to log (only the specified ops will be profiled). """

    verbose: bool = False
    """ Whether to immediately print every communication operation. """

    debug: bool = False
    """ Appends the caller function to each communication operation's `log_name`. """
