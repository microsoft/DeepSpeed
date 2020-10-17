from .builder import OpBuilder


class UtilsBuilder(OpBuilder):
    BUILD_VAR = "DS_BUILD_UTILS"
    OP_NAME = "utils_op"

    def __init__(self, name_prefix=''):
        super().__init__(name=self.OP_NAME, name_prefix=name_prefix, cuda=False)

    def sources(self):
        return ['csrc/utils/flatten_unflatten.cpp']
