# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
from .builder import TorchCPUOpBuilder


class CPUAdamBuilder(TorchCPUOpBuilder):
    BUILD_VAR = "DS_BUILD_CPU_ADAM"
    NAME = "cpu_adam"

    def __init__(self, dtype=None):
        super().__init__(name=self.NAME)
        self.dtype = dtype

    def absolute_name(self):
        return f'deepspeed.ops.adam.{self.NAME}_op'

    def sources(self):
        if self.build_for_cpu:
            return ['csrc/adam/cpu_adam.cpp', 'csrc/adam/cpu_adam_impl.cpp']

        return ['csrc/adam/cpu_adam.cpp', 'csrc/adam/cpu_adam_impl.cpp', 'csrc/common/custom_cuda_kernel.cu']

    def libraries_args(self):
        args = super().libraries_args()
        if self.build_for_cpu:
            return args

        if not self.is_rocm_pytorch():
            args += ['curand']

        return args

    def cxx_args(self):
        import torch
        assert self.dtype is not None, "dype not set"
        args = super().cxx_args()
        if self.dtype == torch.bfloat16:
            args += ['-DHALF_DTYPE=__nv_bfloat16']
        elif self.dtype == torch.half:
            args += ['-DHALF_DTYPE=__half']
        else:
            args += ['-DHALF_DTYPE=float']
        return args

    def include_paths(self):
        import torch
        if self.build_for_cpu:
            CUDA_INCLUDE = []
        elif not self.is_rocm_pytorch():
            CUDA_INCLUDE = [os.path.join(torch.utils.cpp_extension.CUDA_HOME, "include")]
        else:
            CUDA_INCLUDE = []
        return ['csrc/includes'] + CUDA_INCLUDE
