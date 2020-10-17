import torch
from .builder import OpBuilder


class FusedLambBuilder(OpBuilder):
    BUILD_VAR = 'DS_BUILD_FUSED_LAMB'
    OP_NAME = "fused_lamb_op"

    def __init__(self, name_prefix=''):
        super().__init__(name=self.OP_NAME, name_prefix=name_prefix)

        # Fix from apex that might be relevant for us as well, related to https://github.com/NVIDIA/apex/issues/456
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        version_ge_1_1 = []
        if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 0):
            version_ge_1_1 = ['-DVERSION_GE_1_1']
        version_ge_1_3 = []
        if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
            version_ge_1_3 = ['-DVERSION_GE_1_3']
        version_ge_1_5 = []
        if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 4):
            version_ge_1_5 = ['-DVERSION_GE_1_5']
        self.version_dependent_macros = version_ge_1_1 + version_ge_1_3 + version_ge_1_5

    def sources(self):
        return ['csrc/lamb/fused_lamb_cuda.cpp', 'csrc/lamb/fused_lamb_cuda_kernel.cu']

    def include_paths(self):
        return ['csrc/includes']

    def cxx_args(self):
        return ['-O3'] + self.version_dependent_macros

    def nvcc_args(self):
        return ['-lineinfo', '-O3', '--use_fast_math'] + self.version_dependent_macros
