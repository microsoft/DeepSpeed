# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
from .builder import SYCLAutoOpBuilder


class CPUAdamBuilder(SYCLAutoOpBuilder):
    BUILD_VAR = "DS_BUILD_CPU_ADAM"
    NAME = "cpu_adam"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.adam.{self.NAME}_op'

    def sources(self):
        if self.build_for_cpu:
            return ['csrc/adam/cpu_adam.cpp', 'csrc/adam/cpu_adam_impl.cpp']

        return ['csrc/adam/cpu_adam.cpp', 'csrc/adam/cpu_adam_impl.cpp', 'csrc/common/custom_cuda_kernel.cu']

    def include_paths(self):
        return ['csrc/includes']
