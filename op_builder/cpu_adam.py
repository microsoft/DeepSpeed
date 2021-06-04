"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
import os
import sys
import torch
import subprocess
from .builder import CUDAOpBuilder, is_rocm_pytorch


class CPUAdamBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_CPU_ADAM"
    NAME = "cpu_adam"

    def __init__(self):
        super().__init__(name=self.NAME)

    def is_compatible(self):
        # Disable on Windows.
        return sys.platform != "win32"

    def absolute_name(self):
        return f'deepspeed.ops.adam.{self.NAME}_op'

    def sources(self):
        return ['csrc/adam/cpu_adam.cpp', 'csrc/adam/custom_cuda_kernel.cu']

    def include_paths(self):
        if not is_rocm_pytorch:
            CUDA_INCLUDE = [os.path.join(torch.utils.cpp_extension.CUDA_HOME, "include")]
        else:
            CUDA_INCLUDE = [
                            os.path.join(torch.utils.cpp_extension.ROCM_HOME, "include"),
                            os.path.join(torch.utils.cpp_extension.ROCM_HOME, "include", "rocrand"), 
                            os.path.join(torch.utils.cpp_extension.ROCM_HOME, "include", "hiprand"),
            ]
        return ['csrc/includes'] + CUDA_INCLUDE

    def simd_width(self):
        if not self.command_exists('lscpu'):
            self.warning(
                "CPUAdam attempted to query 'lscpu' to detect the existence "
                "of AVX instructions. However, 'lscpu' does not appear to exist on "
                "your system, will fall back to non-vectorized execution.")
            return ''

        result = subprocess.check_output('lscpu', shell=True)
        result = result.decode('utf-8').strip().lower()
        if 'genuineintel' in result:
            if not is_rocm_pytorch and 'avx512' in result:
                return '-D__AVX512__'
            elif 'avx2' in result:
                return '-D__AVX256__'
        return '-D__SCALAR__'

    def cxx_args(self):
        if not is_rocm_pytorch:
            CUDA_LIB64 = os.path.join(torch.utils.cpp_extension.CUDA_HOME, "lib64")
        else:
            CUDA_LIB64 = os.path.join(torch.utils.cpp_extension.ROCM_HOME, "lib")
        SIMD_WIDTH = self.simd_width()

        return [
            '-O3',
            '-std=c++14',
            f'-L{CUDA_LIB64}',
            '-lcudart',
            '-lcublas',
            '-g',
            '-Wno-reorder',
            '-march=native',
            '-fopenmp',
            SIMD_WIDTH
        ]

    def nvcc_args(self):
        args = [
            '-O3',
            '-std=c++14'
        ]
        if is_rocm_pytorch:
            args += [
                '-U__HIP_NO_HALF_OPERATORS__',
                '-U__HIP_NO_HALF_CONVERSIONS__',
                '-U__HIP_NO_HALF2_OPERATORS__'
            ]
        else:
            args += [
                '--use_fast_math',
                '-U__CUDA_NO_HALF_OPERATORS__',
                '-U__CUDA_NO_HALF_CONVERSIONS__',
                '-U__CUDA_NO_HALF2_OPERATORS__'
            ]
            args += self.compute_capability_args()
        return args
