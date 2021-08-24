"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
import torch
from .builder import CUDAOpBuilder, is_rocm_pytorch

if is_rocm_pytorch:
    with open('/opt/rocm/.info/version-dev', 'r') as file:
        ROCM_VERSION_DEV_RAW = file.read()
    ROCM_MAJOR = (ROCM_VERSION_DEV_RAW.split('.')[0])
    ROCM_MINOR = (ROCM_VERSION_DEV_RAW.split('.')[1])
else:
    ROCM_MAJOR = '0'
    ROCM_MINOR = '0'


class FusedLambBuilder(CUDAOpBuilder):
    BUILD_VAR = 'DS_BUILD_FUSED_LAMB'
    NAME = "fused_lamb"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.lamb.{self.NAME}_op'

    def sources(self):
        return ['csrc/lamb/fused_lamb_cuda.cpp', 'csrc/lamb/fused_lamb_cuda_kernel.cu']

    def include_paths(self):
        return ['csrc/includes']

    def cxx_args(self):
        args = super().cxx_args()
        return args + self.version_dependent_macros()

    def nvcc_args(self):
        nvcc_flags=['-O3'] + self.version_dependent_macros()
        if is_rocm_pytorch:
             nvcc_flags+= [
                '-DROCM_VERSION_MAJOR=%s' % ROCM_MAJOR,
                '-DROCM_VERSION_MINOR=%s' % ROCM_MINOR
            ]
        else:
            nvcc_flags.extend(['-lineinfo', '--use_fast_math'] + self.compute_capability_args())
        return nvcc_flags

