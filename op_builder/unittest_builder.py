from .transformer_inference import InferenceBuilder


class UnitTestBuilder(InferenceBuilder):
    NAME = "deepspeed_unittest"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def sources(self):
        return [
            'csrc/transformer/inference/csrc/pt_binding_unittest.cpp',
            'csrc/transformer/inference/csrc/gelu.cu',
        ]
