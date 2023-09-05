"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
from .builder import SYCLOpBuilder, sycl_kernel_path, sycl_kernel_include


class FusedAdamBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_FUSED_ADAM"
    NAME = "fused_adam"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.adam.{self.NAME}_op'

    def sources(self):
        return [
            sycl_kernel_path('csrc/adam/fused_adam_frontend.cpp'),
            sycl_kernel_path('csrc/adam/multi_tensor_adam.dp.cpp'),
        ]

    def include_paths(self):
        return [
            sycl_kernel_include('csrc/includes'),
            sycl_kernel_include('csrc/adam'), 'csrc/includes'
        ]

    def cxx_args(self):
        args = super().cxx_args()
        return args + self.version_dependent_macros()
