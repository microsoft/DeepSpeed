# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Copyright (c) wangruohui

import sys

from .base import *
from .features import HybridSplitQKVContainer, HybridGatedMLPContainer
from deepspeed.utils.types import ActivationFuncType, NormType
from deepspeed.model_implementations.transformers.ds_gpt import DeepSpeedGPTInference
import torch
from torch.nn.parameter import Parameter

from ..policy import (
    TransformerPolicy,
    transformer_param_names,
    maybe_copy,
    maybe_copy_qkv,
    maybe_copy_geglu,
    maybe_get_lora,
)


class DS_InternLMContainer(HybridGatedMLPContainer, HybridSplitQKVContainer, BaseTransformerContainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # All model specific things should be defined here instead of the base class.

    def create_module(self, config=None):
        _config = config if config is not None else self.ds_model_config

        _config.rotate_half = True
        _config.rotate_every_two = False
        _config.rotary_dim = self.hidden_size // self.num_attention_heads
        self.module = DeepSpeedGPTInference(_config, mp_group=self.mp_group)

        return self.module

    def set_lora_params(self):
        """
        Necessary to implement for `HybridEngineContainer`
        """
        self.lora_params = [
            maybe_get_lora(p) for p in [
                self.policy.client_module.mlp.up_proj.weight, self.policy.client_module.mlp.gate_proj.weight,
                self.policy.client_module.mlp.down_proj.weight, self.policy.client_module.self_attn.q_proj.weight,
                self.policy.client_module.self_attn.k_proj.weight, self.policy.client_module.self_attn.v_proj.weight,
                self.policy.client_module.self_attn.o_proj.weight
            ]
        ]

    def get_lora_matched_pair(self):
        up_proj_lora, gate_proj_lora, down_proj_lora, q_lora, k_lora, v_lora, out_lora = self.get_lora_params()
        ret = [(up_proj_lora, self.inter_up_w), (gate_proj_lora, self.inter_gate_w), (down_proj_lora, self._4hh_w),
               (out_lora, self.dense_w), (q_lora, self.qw), (k_lora, self.kw), (v_lora, self.vw)]
        return ret

    def set_q_k_v(self):
        """
        Necessary to implement for `HybridSplitQKVContainer`
        """
        self.qw = self.policy.client_module.self_attn.q_proj.weight
        self.qb = self.policy.client_module.self_attn.q_proj.bias
        self.kw = self.policy.client_module.self_attn.k_proj.weight
        self.kb = self.policy.client_module.self_attn.k_proj.bias
        self.vw = self.policy.client_module.self_attn.v_proj.weight
        self.vb = self.policy.client_module.self_attn.v_proj.bias

    def set_mlp_gate(self):
        """
        Necessary to implement for `HybridGatedMLPContainer`
        """
        self.inter_up_w = self.policy.client_module.mlp.up_proj.weight
        self.inter_up_b = None
        self.inter_gate_w = self.policy.client_module.mlp.gate_proj.weight
        self.inter_gate_b = None

    def load_params(self, module, sd, weight_quantizer, mp_replace, prefix):
        param_names = (
            'self_attn.q_proj.weight', \
            'self_attn.k_proj.weight', \
            'self_attn.v_proj.weight', \
            'self_attn.o_proj.weight', \
            'mlp.up_proj.weight', \
            'mlp.gate_proj.weight', \
            'mlp.down_proj.weight', \
            'input_layernorm.weight', \
            'post_attention_layernorm.weight'
            'self_attn.q_proj.bias', \
            'self_attn.k_proj.bias', \
            'self_attn.v_proj.bias', \
            'self_attn.o_proj.bias', \
        )

        maybe_copy_qkv(module.attention,
                       sd,
                       weight_quantizer,
                       mp_replace,
                       'attn_qkvw', [prefix + param_names[0], prefix + param_names[1], prefix + param_names[2]],
                       split_qkv=self.policy.split_qkv)
        maybe_copy_qkv(module.attention,
                       sd,
                       weight_quantizer,
                       mp_replace,
                       'attn_qkvb', [prefix + param_names[9], prefix + param_names[10], prefix + param_names[11]],
                       split_qkv=self.policy.split_qkv)
        maybe_copy(module.attention, sd, weight_quantizer, mp_replace, transformer_param_names[2],
                   prefix + param_names[3])
        maybe_copy(module.attention, sd, weight_quantizer, mp_replace, transformer_param_names[3],
                   prefix + param_names[12])
        maybe_copy_geglu(module.mlp, sd, weight_quantizer, mp_replace, 'inter_w',
                         [prefix + param_names[4], prefix + param_names[5]])
        maybe_copy(module.mlp, sd, weight_quantizer, mp_replace, 'output_w', prefix + param_names[6])

        maybe_copy(module.mlp, sd, weight_quantizer, mp_replace, transformer_param_names[8], prefix + param_names[7])
        maybe_copy(module, sd, weight_quantizer, mp_replace, transformer_param_names[10], prefix + param_names[8])


class InternLMLayerPolicy(TransformerPolicy):

    def __init__(self, client_module, inference=True):
        super().__init__(
            inference,
            mlp_act_func_type=ActivationFuncType.GATED_SILU,
            norm_type=NormType.RMSNorm,
        )
        self.client_module = client_module

        from transformers.utils import HF_MODULES_CACHE

        sys.path.append(HF_MODULES_CACHE)
        from transformers_modules.internlm.modeling_internlm import InternLMDecoderLayer
        InternLMLayerPolicy._orig_layer_class = InternLMDecoderLayer

    def get_hidden_heads(self):
        return self.client_module.self_attn.q_proj.weight.shape[1], \
                self.client_module.self_attn.num_heads, \
                self.client_module.input_layernorm.variance_epsilon, \
                self.client_module.mlp.gate_proj.weight.shape[0]

    def attention(self, enable_training=False):
        qw = self.client_module.self_attn.q_proj.weight
        kw = self.client_module.self_attn.k_proj.weight
        vw = self.client_module.self_attn.v_proj.weight
        qb = self.client_module.self_attn.q_proj.bias
        kb = self.client_module.self_attn.k_proj.bias
        vb = self.client_module.self_attn.v_proj.bias

        qkvw = Parameter(torch.cat((qw, kw, vw), dim=0), requires_grad=enable_training)
        qkvb = Parameter(torch.cat((qb, kb, vb), dim=0), requires_grad=enable_training)

        return qkvw, \
                qkvb, \
                self.client_module.self_attn.o_proj.weight, \
                self.client_module.self_attn.o_proj.bias

    def mlp(self, enable_training=False):
        mlp1_up = self.client_module.mlp.up_proj.weight
        mlp1_gate = self.client_module.mlp.gate_proj.weight
        mlp2 = self.client_module.mlp.down_proj.weight

        mlp1 = Parameter(torch.cat((mlp1_up, mlp1_gate), dim=0), requires_grad=enable_training)

        return mlp1, None, mlp2, None

    def layernorm(self):
        return self.client_module.post_attention_layernorm.weight, \
               None, \
               self.client_module.input_layernorm.weight, \
               None
