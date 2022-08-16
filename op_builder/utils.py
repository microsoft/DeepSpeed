"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
from .builder import OpBuilder, SYCLOpBuilder


class UtilsBuilder(SYCLOpBuilder if SYCLOpBuilder.is_xpu_pytorch() else OpBuilder):
    BUILD_VAR = "DS_BUILD_UTILS"
    NAME = "utils"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.{self.NAME}_op'

    def sources(self):
        return ['csrc/utils/flatten_unflatten.cpp']

    def sycl_sources(self):
        return ['csrc/utils/flatten_unflatten.cpp']
