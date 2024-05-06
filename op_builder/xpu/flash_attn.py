# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from .builder import SYCLOpBuilder


class FlashAttentionBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_FlashAttention"
    NAME = "flash_attn"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.{self.NAME}_op'

    def sources(self):
        return

    def include_paths(self):
        return []

    def extra_ldflags(self):
        return []

    def cxx_args(self):
        return []

    def load(self):
        try:
            import torch.nn.functional.scaled_dot_product_attention
            import intel_extension_for_pytorch
            return torch.nn.functional.scaled_dot_product_attention
        except ImportError:
            raise ImportError("Please install pytorch and intel_extension_for_pytorch to include scaled dot product attention.")
