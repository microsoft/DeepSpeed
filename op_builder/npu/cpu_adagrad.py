# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import NPUOpBuilder


class CPUAdagradBuilder(NPUOpBuilder):
    BUILD_VAR = "DS_BUILD_CPU_ADAGRAD"
    NAME = "cpu_adagrad"

    def __init__(self):
        super().__init__(name=self.NAME)
        self.dtype = None

    def set_dtype(self, dtype):
        import torch
        assert (dtype in [torch.bfloat16, torch.half, torch.float32])
        self.dtype = dtype
        return self

    def absolute_name(self):
        return f'deepspeed.ops.adagrad.{self.NAME}_op'

    def sources(self):
        return ['csrc/adagrad/cpu_adagrad.cpp']

    def include_paths(self):
        args = super().include_paths()
        args += ['csrc/includes']
        return args

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
        return args
