from .builder import CUDAOpBuilder, SYCLOpBuilder


class FeedForwardBuilder(
        SYCLOpBuilder if SYCLOpBuilder.is_xpu_pytorch() else CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_FEEDFORWARD"
    NAME = "feedforward"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.transformer.{self.NAME}_op'

    def sources(self):
        return []

    def include_paths(self):
        return []

    def sycl_sources(self):
        return [
            'third-party/sycl_kernels/csrc/transformer/sycl/ds_feedforward_sycl.dp.cpp',
            'third-party/sycl_kernels/csrc/transformer/sycl/onemkl_wrappers.dp.cpp',
            'third-party/sycl_kernels/csrc/transformer/sycl/onednn_wrappers.dp.cpp',
            'third-party/sycl_kernels/csrc/transformer/sycl/general_kernels.dp.cpp',
        ]

    def sycl_include_paths(self):
        return ['third-party/sycl_kernels/csrc/includes', 'csrc/includes']
