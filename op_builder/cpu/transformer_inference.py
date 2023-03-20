'''Copyright The Microsoft DeepSpeed Team'''

from .builder import CPUOpBuilder, cpu_kernel_path


class InferenceBuilder(CPUOpBuilder):
    BUILD_VAR = "DS_BUILD_TRANSFORMER_INFERENCE"
    NAME = "transformer_inference"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.transformer.inference.{self.NAME}_op'

    def sources(self):
        return [
            cpu_kernel_path('csrc/foo.c'),
        ]

    def extra_ldflags(self):
        return []

    def include_paths(self):
        return []
