import os
import torch
from .builder import OpBuilder


class CPUAdamBuilder(OpBuilder):
    BUILD_VAR = "DS_BUILD_CPU_ADAM"
    OP_NAME = "cpu_adam_op"

    def __init__(self, name_prefix=''):
        super().__init__(name=self.OP_NAME, name_prefix=name_prefix)

    def sources(self):
        return ['csrc/adam/cpu_adam.cpp', 'csrc/adam/custom_cuda_kernel.cu']

    def include_paths(self):
        CUDA_INCLUDE = os.path.join(torch.utils.cpp_extension.CUDA_HOME, "include")
        return ['csrc/includes', CUDA_INCLUDE]

    def cxx_args(self):
        CUDA_LIB64 = os.path.join(torch.utils.cpp_extension.CUDA_HOME, "lib64")
        SIMD_WIDTH = '-D__AVX256__'
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
            '--use_fast_math',
            '-std=c++14',
            '-U__CUDA_NO_HALF_OPERATORS__',
            '-U__CUDA_NO_HALF_CONVERSIONS__',
            '-U__CUDA_NO_HALF2_OPERATORS__'
        ]
        args += self.compute_capability_args()
        return args
