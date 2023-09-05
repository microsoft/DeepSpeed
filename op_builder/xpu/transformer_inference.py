from .builder import SYCLOpBuilder


class InferenceBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_TRANSFORMER_INFERENCE"
    NAME = "transformer_inference"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.transformer.inference.{self.NAME}_op'

    def is_compatible(self, verbose=True):
        return super().is_compatible(verbose)

    def sources(self):
        return [
            'csrc/transformer/inference/csrc/softmax.cpp',
            'csrc/transformer/inference/csrc/pt_binding.cpp',
            'csrc/transformer/inference/csrc/gelu.cpp',
            'csrc/transformer/inference/csrc/inference_onednn_wrappers.cpp',
            'csrc/transformer/inference/csrc/inference_onemkl_wrappers.cpp',
            'csrc/transformer/inference/csrc/layer_norm.cpp',
            'csrc/transformer/inference/csrc/pointwise_ops.cpp'
        ]

    def include_paths(self):
        includes = ['csrc/xpu/transformer/inference/includes', 'csrc/transformer/inference/includes']
        return includes

