from .builder import SYCLOpBuilder, sycl_kernel_path, sycl_kernel_include


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
            sycl_kernel_path('csrc/transformer/inference/csrc/softmax.cpp'),
            sycl_kernel_path('csrc/transformer/inference/csrc/pt_binding.cpp'),
            sycl_kernel_path('csrc/transformer/inference/csrc/gelu.cpp'),
            sycl_kernel_path('csrc/transformer/inference/csrc/inference_onednn_wrappers.cpp'),
            sycl_kernel_path('csrc/transformer/inference/csrc/inference_onemkl_wrappers.cpp'),
            sycl_kernel_path('csrc/transformer/inference/csrc/layer_norm.cpp'),
            sycl_kernel_path('csrc/transformer/inference/csrc/pointwise_ops.cpp'),
        ]

    def include_paths(self):
        includes = [sycl_kernel_include('csrc/transformer/inference/includes'), 'csrc/transformer/inference/includes']
        return includes

