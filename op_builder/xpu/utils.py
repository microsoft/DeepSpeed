"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
from .builder import SYCLOpBuilder


class UtilsBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_UTILS"
    NAME = "utils"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.{self.NAME}_op'

    def sources(self):
        return ['csrc/utils/flatten_unflatten.cpp']

    def include_paths(self):
        return []
