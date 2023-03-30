'''Copyright The Microsoft DeepSpeed Team'''

from .builder import CPUOpBuilder


class NotImplementedBuilder(CPUOpBuilder):
    BUILD_VAR = "DS_BUILD_NOT_IMPLEMENTED"
    NAME = "deepspeed_not_implemented"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.comm.{self.NAME}_op'

    def sources(self):
        return []
