# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import CUDAOpBuilder

import sys


class FusedAdamBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_FUSED_ADAM"
    NAME = "fused_adam"

    def __init__(self):
        super().__init__(name=self.NAME)
        self.dtype = None

    def set_dtype(self, dtype):
        import torch
        assert (dtype in [torch.bfloat16, torch.half, torch.float32])
        self.dtype = dtype
        return self

    def absolute_name(self):
        return f'deepspeed.ops.adam.{self.NAME}_op'

    def sources(self):
        return ['csrc/adam/fused_adam_frontend.cpp', 'csrc/adam/multi_tensor_adam.cu']

    def include_paths(self):
        return ['csrc/includes', 'csrc/adam']

    def cxx_args(self):
        import torch
        args = super().cxx_args()
        assert self.dtype is not None, "dype not set"
        if self.dtype == torch.bfloat16:
            args += ['-DHALF_DTYPE=c10::BFloat16']
        elif self.dtype == torch.half:
            args += ['-DHALF_DTYPE=c10::Half']
        else:
            args += ['-DHALF_DTYPE=float']

        return args + self.version_dependent_macros()

    def nvcc_args(self):
        nvcc_flags = ['-O3'] + self.version_dependent_macros()
        if not self.is_rocm_pytorch():
            nvcc_flags.extend(
                ['-allow-unsupported-compiler' if sys.platform == "win32" else '', '-lineinfo', '--use_fast_math'] +
                self.compute_capability_args())
        return nvcc_flags
