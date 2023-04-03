from .builder import OpBuilder


class TMapBuilder(OpBuilder):
    BUILD_VAR = "DS_BUILD_TENSOR_MAP"
    NAME = "tensor_map"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.{self.NAME}_op'

    def sources(self):
        return ['csrc/map/tensor_map.cpp']
