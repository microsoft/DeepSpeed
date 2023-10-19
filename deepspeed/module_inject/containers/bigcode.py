# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .base import *
from ..policy import TransformerPolicy
from deepspeed.model_implementations.transformers.ds_bigcode import DeepSpeedBigCodeInference


class DS_BigCodeContainer(BaseTransformerContainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # All model specific things should be defined here instead of the base class.

    def create_module(self, config=None):
        _config = config if config is not None else self.ds_model_config

        _config.num_kv = self.policy.client_module.attn.kv_heads

        self.module = DeepSpeedBigCodeInference(_config)
        return self.module


class HFBigCodeLayerPolicy(TransformerPolicy):
    _orig_layer_class = None

    def __init__(self, client_module, inference=True):
        super().__init__(inference)

        self.client_module = client_module
        try:
            import transformers
            HFBigCodeLayerPolicy._orig_layer_class = transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeBlock
        except:
            HFBigCodeLayerPolicy._orig_layer_class = None

    def attention(self, enable_training=False):
        return self.client_module.attn.c_attn.weight, \
            self.client_module.attn.c_attn.bias, \
            self.client_module.attn.c_proj.weight, \
            self.client_module.attn.c_proj.bias

    def get_hidden_heads(self):
        return self.client_module.attn.embed_dim, \
            self.client_module.attn.num_heads, \
            self.client_module.ln_1.eps, \
            DEFAULT_INTERMEDIATE_SIZE

    def mlp(self, enable_training=False):
        return self.client_module.mlp.c_fc.weight, \
            self.client_module.mlp.c_fc.bias, \
            self.client_module.mlp.c_proj.weight, \
            self.client_module.mlp.c_proj.bias

    def layernorm(self):
        return self.client_module.ln_2.weight, \
            self.client_module.ln_2.bias, \
            self.client_module.ln_1.weight, \
            self.client_module.ln_1.bias
