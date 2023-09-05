"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
from .builder import SYCLOpBuilder, sycl_kernel_path, sycl_kernel_include


class CPUAdagradBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_CPU_ADAGRAD"
    NAME = "cpu_adagrad"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.adagrad.{self.NAME}_op'

    def sources(self):
        return [
            sycl_kernel_path('csrc/adagrad/cpu_adagrad.dp.cpp'),
            sycl_kernel_path('csrc/adam/custom_sycl_kernel.dp.cpp'),
        ]

    def include_paths(self):
        return [
            sycl_kernel_include('csrc/includes'),
            sycl_kernel_include('csrc/adagrad'),
            'csrc/includes'
        ]
