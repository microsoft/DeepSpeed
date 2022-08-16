"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
from .builder import CUDAOpBuilder, SYCLOpBuilder


class TransformerBuilder(
        SYCLOpBuilder if SYCLOpBuilder.is_xpu_pytorch() else CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_TRANSFORMER"
    NAME = "transformer"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.transformer.{self.NAME}_op'

    def extra_ldflags(self):
        if not self.is_rocm_pytorch():
            return ['-lcurand']
        else:
            return []

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
        includes = ['csrc/includes']
        if self.is_rocm_pytorch():
            from torch.utils.cpp_extension import ROCM_HOME
            includes += [
                '{}/hiprand/include'.format(ROCM_HOME),
                '{}/rocrand/include'.format(ROCM_HOME)
            ]
        return includes

    def sycl_sources(self):
        return [
            'third-party/sycl_kernels/csrc/transformer/sycl/onednn_wrappers.dp.cpp',
            'third-party/sycl_kernels/csrc/transformer/sycl/ds_transformer_sycl.dp.cpp',
            'third-party/sycl_kernels/csrc/transformer/sycl/onemkl_wrappers.dp.cpp',
            'third-party/sycl_kernels/csrc/transformer/sycl/transform_kernels.dp.cpp',
            'third-party/sycl_kernels/csrc/transformer/sycl/gelu_kernels.dp.cpp',
            'third-party/sycl_kernels/csrc/transformer/sycl/dropout_kernels.dp.cpp',
            'third-party/sycl_kernels/csrc/transformer/sycl/normalize_kernels.dp.cpp',
            'third-party/sycl_kernels/csrc/transformer/sycl/softmax_kernels.dp.cpp',
            'third-party/sycl_kernels/csrc/transformer/sycl/general_kernels.dp.cpp'
        ]

    def sycl_include_paths(self):
        includes = ['third-party/sycl_kernels/csrc/includes']
        return includes
