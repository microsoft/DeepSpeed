from .transformer_inference import InferenceBuilder


class InferenceUnitTestBuilder(InferenceBuilder):
    NAME = "deepspeed_unittest"

    def sources(self):
        return [
            'csrc/transformer/inference/csrc/pt_binding_unittest.cpp',
            'csrc/transformer/inference/csrc/gelu.cu',
        ]
