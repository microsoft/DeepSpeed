"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
from .builder import SYCLOpBuilder, sycl_kernel_path, sycl_kernel_include


class CPUAdamBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_CPU_ADAM"
    NAME = "cpu_adam"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.adam.{self.NAME}_op'

    def sources(self):
        return [
            sycl_kernel_path('csrc/adam/cpu_adam.dp.cpp'),
            sycl_kernel_path('csrc/adam/custom_sycl_kernel.dp.cpp'),
        ]

    def libraries_args(self):
        args = super().libraries_args()
        return args

    def include_paths(self):
        return [
            sycl_kernel_include('csrc/includes'),
            sycl_kernel_include('csrc/adam'), 'csrc/includes'
        ]
