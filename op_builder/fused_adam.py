import torch
from .builder import OpBuilder


class FusedAdamBuilder(OpBuilder):
    BUILD_VAR = "DS_BUILD_FUSED_ADAM"
    OP_NAME = "fused_adam_op"

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
        return ['csrc/adam/fused_adam_frontend.cpp', 'csrc/adam/multi_tensor_adam.cu']

    def include_paths(self):
        return ['csrc/includes']

    def cxx_args(self):
        return ['-O3'] + self.version_dependent_macros

    def nvcc_args(self):
        return ['-lineinfo', '-O3', '--use_fast_math'] + self.version_dependent_macros
