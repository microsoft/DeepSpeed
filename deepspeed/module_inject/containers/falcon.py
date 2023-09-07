# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .base import *
from .features.meta_tensor import MetaTensorContainer
from .features.hybrid_engine import HybridEngineContainer
from deepspeed.model_implementations.transformers.ds_gpt import DeepSpeedGPTInference

from ..policy import transformer_param_names
from ..policy import (
    TransformerPolicy,
    maybe_copy,
    maybe_get_lora,
)


class DS_FALCONContainer(MetaTensorContainer, BaseTransformerContainer, HybridEngineContainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # All model specific things should be defined here instead of the base class.

    def create_module(self, config=None):
        _config = config if config is not None else self.ds_model_config

        _config.rotate_half = True
        _config.rotate_every_two = False
        _config.rotary_dim = self.hidden_size // self.num_attention_heads
        _config.multi_query = True
        _config.global_kv_sharing = (self.policy.num_kv == 1) and self.policy.config.multi_query 
        _config.num_kv = self.policy.num_kv
        _config.mlp_after_attn = not self.policy.config.parallel_attn
        self.module = DeepSpeedGPTInference(_config, mp_group=self.mp_group)

        self.module.config.rotate_half = True
        self.module.config.rotate_every_two = False

        return self.module

    def set_lora_params(self):
        """
        Necessary to implement for `HybridEngineContainer`
        """
        attention = self.policy.client_module.self_attention

        self.lora_params = [
            maybe_get_lora(p) for p in [
                self.policy.client_module.mlp.dense_h_to_4h, self.policy.client_module.mlp.dense_4h_to_h,
                attention.query_key_value, attention.dense
            ]
        ]

    def load_params(self, module, sd, weight_quantizer, mp_replace, prefix):
        ln_2 = ('ln_mlp' if hasattr(self.policy.client_module, 'ln_mlp') else 'post_attention_layernorm') if not self.policy.config.parallel_attn else None
        ln_1 = 'ln_attn' if hasattr(self.policy.client_module, 'ln_attn') else 'input_layernorm'
        param_names = (
            'self_attention.query_key_value.weight', \
            'self_attention.dense.weight', \
            'mlp.dense_h_to_4h.weight', \
            'mlp.dense_4h_to_h.weight', \
            None if ln_2 is None else ln_2 + '.weight', \
            None if ln_2 is None else ln_2 + '.bias', \
            ln_1 + '.weight', \
            ln_1 + '.bias'
        )
        for i in range(0, 2):
            maybe_copy(module.attention, 
                       sd, 
                       weight_quantizer, 
                       mp_replace, 
                       transformer_param_names[i * 2],
                       prefix + param_names[i],
                       qkv=True,
                       megatron_v2=self.policy.is_megatron_v2,
                       split_qkv=self.policy.split_qkv)
        maybe_copy(module.mlp, sd, weight_quantizer, mp_replace, transformer_param_names[4],
                       prefix + param_names[2])
        maybe_copy(module.mlp, sd, weight_quantizer, mp_replace, transformer_param_names[6],
                       prefix + param_names[3])
        if ln_2 is not None:
            maybe_copy(module.mlp, sd, weight_quantizer, mp_replace, transformer_param_names[8],
                       prefix + param_names[4])
        if ln_2 is not None:
            maybe_copy(module.mlp, sd, weight_quantizer, mp_replace, transformer_param_names[9],
                       prefix + param_names[5])
        for i in range(6, 8):
            maybe_copy(module, sd, weight_quantizer, mp_replace, transformer_param_names[i + 4], prefix + param_names[i])


    def attention_qkv_mp(self, mp_replace, reversed_dim=False):
        self.module.attention.attn_qkvw = mp_replace.copy(self.module.attention.attn_qkvw,
                                                          self.qkvw,
                                                          int8=reversed_dim)
        self.module.attention.attn_qkvb = mp_replace.copy(self.module.attention.attn_qkvb,
                                                          self.qkvb,
                                                          int8=reversed_dim)


class FALCONLayerPolicy(TransformerPolicy):

    def __init__(self, client_module, inference=True):
        super().__init__(inference, split_qkv=False)
        self.client_module = client_module
        FALCONLayerPolicy.name = 'falcon'
        FALCONLayerPolicy._orig_layer_class = None
        if client_module is not None:
            self.num_kv = self.client_module.self_attention.num_kv
            self.config = self.client_module.config

    def get_hidden_heads(self):
        heads = self.client_module.self_attention.num_heads
        #if heads % 2 == 1:
        #    heads = heads + 1
        #    attention.query_key_value.weight.data = torch.cat(attention.query_key_value.weight)
        return self.client_module.self_attention.query_key_value.weight.shape[1], \
                heads, \
                self.client_module.ln_mlp.eps if hasattr(self.client_module, 'ln_mlp') else self.client_module.input_layernorm.eps, \
                DEFAULT_INTERMEDIATE_SIZE

    def attention(self, enable_training=False):
        attention = self.client_module.self_attention

        return attention.query_key_value.weight, \
               None, \
               attention.dense.weight, \
               None

    def mlp(self, enable_training=False):
        return self.client_module.mlp.dense_h_to_4h.weight, \
                   None, \
                   self.client_module.mlp.dense_4h_to_h.weight, \
                   None

    def layernorm(self):
        ln_2 = (self.client_module.ln_mlp if hasattr(self.client_module, 'ln_mlp') else self.client_module.post_attention_layernorm) if not self.config.parallel_attn else None
        ln_1 = self.client_module.ln_attn if hasattr(self.client_module, 'ln_attn') else self.client_module.input_layernorm
        #import pdb;pdb.set_trace()
        return ln_2.weight if ln_2 is not None else None, \
               ln_2.bias if ln_2 is not None else None, \
               ln_1.weight, \
               ln_1.bias
