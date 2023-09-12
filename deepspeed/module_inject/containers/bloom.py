# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .base import *
from .features.meta_tensor import MetaTensorContainer
from .features.hybrid_engine import HybridEngineContainer
from deepspeed.model_implementations.transformers.ds_bloom import DeepSpeedBloomInference
from ..policy import TransformerPolicy
from ..policy import transformer_param_names
from ..policy import maybe_copy

from ..policy import maybe_get_lora

supported_models = {None}


class DS_BloomContainer(MetaTensorContainer, HybridEngineContainer, BaseTransformerContainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # All model specific things should be defined here instead of the base class.
        self.bigscience_bloom = True

    def create_module(self, config=None):
        _config = config if config is not None else self.ds_model_config

        self.module = DeepSpeedBloomInference(_config, mp_group=self.mp_group)
        self.module.config.scale_attention = self.scale_attention
        return self.module

    def attention_qkv_mp(self, mp_replace, reversed_dim=False):
        # The shape of qkv weight is like [3d, d], and it can be split alone column axis like this:
        # [w_q_head1, w_k_head1, w_v_head1, w_q_head2, w_k_head2, w_v_head2, ...]
        # So here we use mp_replace.copy instead of mp_replace.strided_copy.
        attn_qkvw_dst = torch.empty_like(self.qkvw[:self.qkvw.shape[0]//mp_replace.mp_size], dtype=self.dtype)
        attn_qkvb_dst = torch.empty_like(self.qkvb[:self.qkvw.shape[0]//mp_replace.mp_size], dtype=self.dtype)
        self.module.attention.attn_qkvw = mp_replace.copy(attn_qkvw_dst, self.qkvw, int8=reversed_dim)
        self.module.attention.attn_qkvb = mp_replace.copy(attn_qkvb_dst, self.qkvb, int8=reversed_dim)

    def attention_o_mp(self, mp_replace, reversed_dim=False):
        # row split
        attn_ow_dst = torch.empty_like(self.dense_w[:, :self.dense_w.shape[0]//mp_replace.mp_size], dtype=self.dtype)
        attn_ob_dst = torch.empty_like(self.dense_b, dtype=self.dtype)
        self.module.attention.attn_ow = mp_replace.copy(attn_ow_dst, self.dense_w, int8=reversed_dim)
        self.module.attention.attn_ob = mp_replace.copy(attn_ob_dst, self.dense_b, int8=reversed_dim)

    def mlp_inter_mp(self, mp_replace, reversed_dim=False):
        # column split
        inter_w_dst = torch.empty_like(self._h4h_w[:self._h4h_w.shape[0]//mp_replace.mp_size], dtype=self.dtype)
        inter_b_dst = torch.empty_like(self._h4h_b[:self._h4h_b.shape[0]//mp_replace.mp_size], dtype=self.dtype)
        self.module.mlp.inter_w = mp_replace.copy(inter_w_dst, self._h4h_w, int8=reversed_dim)
        self.module.mlp.inter_b = mp_replace.copy(inter_b_dst, self._h4h_b, int8=reversed_dim)

    def mlp_output_mp(self, mp_replace, reversed_dim=False):
        # row split
        output_w_dst = torch.empty_like(self._4hh_w[:, :self._4hh_w.shape[1]//mp_replace.mp_size], dtype=self.dtype)
        output_b_dst = torch.empty_like(self._4hh_b, dtype=self.dtype)
        self.module.mlp.output_w = mp_replace.copy(output_w_dst, self._4hh_w, int8=reversed_dim)
        self.module.mlp.output_b = mp_replace.copy(output_b_dst, self._4hh_b, int8=reversed_dim)

    def release_memory(self,):
        super().release_memory()
        if self.module.layer_past is not None:
            del self.module.layer_past
            self.module.layer_past = None

    def get_lora_matched_pair(self):
        """
        Necessary to implement for `HybridEngineContainer`
        """
        fc1_lora, fc2_lora, qkv_lora, out_lora = self.get_lora_params()
        ret = [(fc1_lora, self._h4h_w), (fc2_lora, self._4hh_w), (qkv_lora, self.qkvw), (out_lora, self.dense_w)]
        return ret

    def set_lora_params(self):
        """
        Necessary to implement for `HybridEngineContainer`
        """
        self.lora_params = [
            maybe_get_lora(p) for p in [
                self.policy.client_module.mlp.dense_h_to_4h, self.policy.client_module.mlp.dense_4h_to_h, self.policy.
                client_module.self_attention.query_key_value, self.policy.client_module.self_attention.dense
            ]
        ]

    def load_params(self, module, sd, weight_quantizer, mp_replace, prefix):
        param_names = (
            'self_attention.query_key_value.weight', \
            'self_attention.query_key_value.bias', \
            'self_attention.dense.weight', \
            'self_attention.dense.bias', \
            'mlp.dense_h_to_4h.weight', \
            'mlp.dense_h_to_4h.bias', \
            'mlp.dense_4h_to_h.weight', \
            'mlp.dense_4h_to_h.bias', \
            'post_attention_layernorm.weight', \
            'post_attention_layernorm.bias', \
            'input_layernorm.weight', \
            'input_layernorm.bias'
        )
        for i in range(0, 2):
            maybe_copy(module.attention,
                       sd,
                       weight_quantizer,
                       mp_replace,
                       transformer_param_names[i],
                       prefix + param_names[i],
                       qkv=True,
                       megatron_v2=self.policy.is_megatron_v2,
                       split_qkv=self.policy.split_qkv)
        for i in range(2, 4):
            maybe_copy(module.attention, sd, weight_quantizer, mp_replace, transformer_param_names[i],
                       prefix + param_names[i])
        for i in range(4, 10):
            maybe_copy(module.mlp, sd, weight_quantizer, mp_replace, transformer_param_names[i],
                       prefix + param_names[i])
        for i in range(10, 12):
            maybe_copy(module, sd, weight_quantizer, mp_replace, transformer_param_names[i], prefix + param_names[i])


class BLOOMLayerPolicy(TransformerPolicy):
    _orig_layer_class = None

    def __init__(self, client_module, inference=True, use_load_prefix=True, split_qkv=False):
        super().__init__(inference, linear_layer=True, use_load_prefix=use_load_prefix, split_qkv=split_qkv)
        self.client_module = client_module
        try:
            import transformers
            BLOOMLayerPolicy._orig_layer_class = transformers.models.bloom.modeling_bloom.BloomBlock
            global supported_models
            supported_models.update({transformers.models.bloom.modeling_bloom.BloomModel})
        except Exception as e:
            print(f"WARNING! Setting BLOOMLayerPolicy._orig_layer_class to None due to Exception: {e}")
            BLOOMLayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        return self.client_module.self_attention.hidden_size, \
                self.client_module.self_attention.num_heads, \
                self.client_module.input_layernorm.eps, \
                DEFAULT_INTERMEDIATE_SIZE

    def attention(self, enable_training=False):
        return self.client_module.self_attention.query_key_value.weight, \
                self.client_module.self_attention.query_key_value.bias, \
                self.client_module.self_attention.dense.weight, \
                self.client_module.self_attention.dense.bias,

    def mlp(self, enable_training=False):
        return self.client_module.mlp.dense_h_to_4h.weight, \
               self.client_module.mlp.dense_h_to_4h.bias, \
               self.client_module.mlp.dense_4h_to_h.weight, \
               self.client_module.mlp.dense_4h_to_h.bias

    def layernorm(self):
        return self.client_module.post_attention_layernorm.weight, \
               self.client_module.post_attention_layernorm.bias, \
               self.client_module.input_layernorm.weight, \
               self.client_module.input_layernorm.bias
