from .base import *
from deepspeed.model_implementations.transformers.ds_bert import DeepSpeedBERTInference
from deepspeed.ops.transformer.inference.config import DeepSpeedInferenceConfig


class DS_BERTContainer(BaseTransformerContainer):
    def __init__(self, policy, config, model_config):
        super().__init__(policy, config, model_config)

        # All model specific things should be defined here instead of the base class.
        self.scale_attention = self.policy.scale_attention
        self.layer_norm_eps = 1e-12
        self.attn_linear_layer = True
        self.mlp_linear_layer = True
        self.pre_layer_norm = False
        self.pre_attn_norm = False
        self.triangular_masking = False

    def create_config(self):
        self.config = DeepSpeedInferenceConfig(
            hidden_size=self.hidden_size,
            heads=self.num_attention_heads,
            layer_norm_eps=self.layer_norm_eps,
            fp16=self.fp16,
            pre_layer_norm=self.pre_layer_norm,
            mp_size=self.mp_size,
            triangular_masking=self.triangular_masking,
            q_int8=self.quantize)
        return self.config

    def create_module(self, config=None):
        _config = config if config is not None else self.config
        self.module = DeepSpeedBERTInference(_config,
                                             mp_group=self.mp_group,
                                             qkv_merging=True)
        self.module.config.scale_attention = self.scale_attention
        return self.module
