# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import CPUOpBuilder

import sys


class FusedAdamBuilder(CPUOpBuilder):
    BUILD_VAR = "DS_BUILD_FUSED_ADAM"
    NAME = "fused_adam"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.adam.{self.NAME}_op'

    def sources(self):
        return ['csrc/cpu/adam/fused_adam.cpp']

    def include_paths(self):
        return ['csrc/cpu/includes', 'csrc/cpu/adam']
