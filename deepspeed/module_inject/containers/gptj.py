from .base import *
from .features.meta_tensor import MetaTensorContainer
from deepspeed.model_implementations.transformers.ds_gpt import DeepSpeedGPTInference


class DS_GPTJContainer(MetaTensorContainer, BaseTransformerContainer):
    def __init__(self, policy, config, model_config, layer_id):
        super().__init__(policy, config, model_config, layer_id)

        #self.attn_linear_layer = True
        #self.mlp_linear_layer = True
        #self.scale_attention = self.policy.scale_attention
        #self.layer_norm_eps = 1e-05
        #self.pre_layer_norm = True
        #self.rotary_dim = 64
        #self.window_size = 1
        #self.mlp_after_attn = False

    def create_module(self, config=None):
        _config = config if config is not None else self.config
        self.module = DeepSpeedGPTInference(_config, mp_group=self.mp_group)
        self.module.config.scale_attention = self.scale_attention
        return self.module
