import torch
from packaging import version
from .builder import CUDAOpBuilder, installed_cuda_version


class InferenceBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_TRANSFORMER_INFERENCE"
    NAME = "transformer_inference"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.transformer.inference.{self.NAME}_op'

    def is_compatible(self, verbose=True):
        sys_cuda_major, _ = installed_cuda_version()
        torch_cuda_major = version.parse(torch.version.cuda).major
        cuda_capability = torch.cuda.get_device_properties(0).major
        cuda_okay = True
        if cuda_capability >= 8:
            if torch_cuda_major < 11 or sys_cuda_major < 11:
                self.warning("On Ampere and higher architectures please use CUDA 11+")
                cuda_okay = False
        return super().is_compatible(verbose) and cuda_okay

    def sources(self):
        return [
            'csrc/transformer/inference/csrc/pt_binding.cpp',
            'csrc/transformer/inference/csrc/gelu.cu',
            'csrc/transformer/inference/csrc/normalize.cu',
            'csrc/transformer/inference/csrc/softmax.cu',
            'csrc/transformer/inference/csrc/dequantize.cu',
            'csrc/transformer/inference/csrc/apply_rotary_pos_emb.cu',
            'csrc/transformer/inference/csrc/transform.cu',
        ]

    def extra_ldflags(self):
        if not self.is_rocm_pytorch():
            return ['-lcurand']
        else:
            return []

    def include_paths(self):
        return ['csrc/transformer/inference/includes']
