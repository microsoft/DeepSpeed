from .base import *
from .features.meta_tensor import MetaTensorContainer
from deepspeed.model_implementations.transformers.ds_opt import DeepSpeedOPTInference
from deepspeed.ops.transformer.inference.config import DeepSpeedInferenceConfig


class DS_OPTContainer(MetaTensorContainer, BaseTransformerContainer):
    def __init__(self, policy):
        super().__init__(policy)

        self.attn_linear_layer = True
        self.mlp_linear_layer = True
        self.scale_attention = self.policy.scale_attention
        self.mlp_act_func_type = self.policy.mlp_act_func_type
        self.window_size = 1

    def create_config(self):
        self.config = DeepSpeedInferenceConfig(hidden_size=self.hidden_size,
                                               heads=self.num_attention_heads,
                                               fp16=self.fp16,
                                               mp_size=self.mp_size,
                                               mlp_act_func_type=self.mlp_act_func_type,
                                               window_size=self.window_size,
                                               q_int8=self.quantize)
        return self.config

    def create_module(self, config=None):
        _config = config if config is not None else self.config
        self.module = DeepSpeedOPTInference(_config, mp_group=self.mp_group)
        self.module.config.scale_attention = self.scale_attention
        return self.module
