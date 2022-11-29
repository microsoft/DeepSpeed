from .base import *
from deepspeed.model_implementations.transformers.ds_transformer import DeepSpeedTransformerInference
from deepspeed.ops.transformer.inference.config import DeepSpeedInferenceConfig


class DS_MegatronContainer(BaseTransformerContainer):
    def __init__(self, policy):
        super().__init__(policy)

        self.attn_linear_layer = True
        self.mlp_linear_layer = True
        self.scale_attention = True
        self.window_size = 1

    def create_config(self):
        self.config = DeepSpeedInferenceConfig(hidden_size=self.hidden_size,
                                               heads=self.num_attention_heads,
                                               fp16=self.fp16,
                                               mp_size=self.mp_size,
                                               window_size=self.window_size)
        return self.config

    def create_module(self, config=None):
        _config = config if config is not None else self.config
        self.module = DeepSpeedTransformerInference(_config, mp_group=self.mp_group)
        self.module.config.scale_attention = self.scale_attention
        return self.module
