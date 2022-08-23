"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
from .builder import CUDAOpBuilder


class TransformerKernelsBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_TRANSFORMER_KERNELS"
    NAME = "transformer_kernels"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.transformer_kernels.{self.NAME}_op'

    def sources(self):
        return [
            'csrc/transformer/pt_binding.cpp',
            'csrc/transformer/softmax_kernels.cu',
        ]

    def extra_ldflags(self):
        return ['-lcurand']

    def include_paths(self):
        return ['csrc/includes']
