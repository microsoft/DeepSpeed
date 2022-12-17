from .base import *
from .features.meta_tensor import MetaTensorContainer

from deepspeed.model_implementations.transformers.ds_bloom import DeepSpeedBloomInference

from deepspeed.ops.transformer.inference.config import DeepSpeedInferenceConfig


class DS_BloomContainer(MetaTensorContainer, BaseTransformerContainer):
    def __init__(self, policy, config, model_config):
        super().__init__(policy, config, model_config)

        # All model specific things should be defined here instead of the base class.
        self.scale_attention = self.policy.scale_attention
        self.pre_attn_norm = False
        self.attn_linear_layer = True
        self.mlp_linear_layer = True
        self.layer_norm_eps = 1e-05  # hardcode for now, todo: take it from the top config or user args
        self.pre_layer_norm = True
        self.window_size = 1  # hardcode for 3b, todo: take it from the config or user args
        self.bigscience_bloom = True

    def create_config(self):
        self.config = DeepSpeedInferenceConfig(hidden_size=self.hidden_size,
                                               heads=self.num_attention_heads,
                                               layer_norm_eps=self.layer_norm_eps,
                                               fp16=self.fp16,
                                               pre_layer_norm=self.pre_layer_norm,
                                               mp_size=self.mp_size,
                                               window_size=self.window_size,
                                               bigscience_bloom=self.bigscience_bloom,
                                               q_int8=self.quantize)
        return self.config

    def create_module(self, config=None):
        print(f"BLOOM create_module")
        _config = config if config is not None else self.config

        self.module = DeepSpeedBloomInference(_config, mp_group=self.mp_group)
        self.module.config.scale_attention = self.scale_attention
        return self.module

    def attention_qkv_mp(self, mp_replace):
        print("Entered BLOOM defined attention_qkv_mp!!")
        self.module.attention.attn_qkvw = mp_replace.copy(
            self.module.attention.attn_qkvw,
            self.qkvw)
        self.module.attention.attn_qkvb = mp_replace.copy(
            self.module.attention.attn_qkvb,
            self.qkvb)
