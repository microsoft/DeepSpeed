import torch
from .builder import OpBuilder


class TransformerBuilder(OpBuilder):
    BUILD_VAR = "DS_BUILD_TRANSFORMER"
    OP_NAME = "transformer_op"

    def __init__(self, name=None, name_prefix=''):
        name = self.OP_NAME if name is None else name
        super().__init__(name=name, name_prefix=name_prefix)

    def sources(self):
        return [
            'csrc/transformer/ds_transformer_cuda.cpp',
            'csrc/transformer/cublas_wrappers.cu',
            'csrc/transformer/transform_kernels.cu',
            'csrc/transformer/gelu_kernels.cu',
            'csrc/transformer/dropout_kernels.cu',
            'csrc/transformer/normalize_kernels.cu',
            'csrc/transformer/softmax_kernels.cu',
            'csrc/transformer/general_kernels.cu'
        ]

    def include_paths(self):
        return ['csrc/includes']

    def nvcc_args(self):
        args = [
            '-O3',
            '--use_fast_math',
            '-std=c++14',
            '-U__CUDA_NO_HALF_OPERATORS__',
            '-U__CUDA_NO_HALF_CONVERSIONS__',
            '-U__CUDA_NO_HALF2_OPERATORS__'
        ]

        return args + self.compute_capability_args()

    def cxx_args(self):
        return ['-O3', '-std=c++14', '-g', '-Wno-reorder']
