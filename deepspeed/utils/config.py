# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.runtime.config_utils import get_scalar_param, DeepSpeedConfigObject
from deepspeed.utils.constants import *


class DeepSpeedThroughputTimerConfig(DeepSpeedConfigObject):

    def __init__(self, param_dict):
        super(DeepSpeedThroughputTimerConfig, self).__init__()

        self.enabled = None
        self.synchronized = None

        timers_dict = {}
        if param_dict and TIMERS in param_dict:
            if TIMERS_THROUGHPUT in param_dict[TIMERS]:
                timers_dict = param_dict[TIMERS][TIMERS_THROUGHPUT]

        self._initialize(timers_dict)

    def _initialize(self, param_dict):
        self.enabled = get_scalar_param(param_dict, TIMERS_THROUGHPUT_ENABLED, TIMERS_THROUGHPUT_ENABLED_DEFAULT)
        self.synchronized = get_scalar_param(param_dict, TIMERS_THROUGHPUT_SYNCHRONIZED,
                                             TIMERS_THROUGHPUT_SYNCHRONIZED_DEFAULT)
