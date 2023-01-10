from .base import *
from .features.meta_tensor import MetaTensorContainer
from .features.megatron import MegatronContainer
from deepspeed.model_implementations.transformers.ds_gpt import DeepSpeedGPTInference


class DS_GPTNEOXContainer(MetaTensorContainer,
                          MegatronContainer,
                          BaseTransformerContainer):
    def __init__(self, policy, config, model_config, layer_id):
        super().__init__(policy, config, model_config, layer_id)

        #self.attn_linear_layer = True
        #self.mlp_linear_layer = True
        #self.layer_norm_eps = 1e-05
        #self.scale_attention = self.policy.scale_attention
        #self.window_size = 1
        #self.rotary_dim = 24
        #self.mlp_after_attn = False

    def create_module(self, config=None):
        _config = config if config is not None else self.config
        self.module = DeepSpeedGPTInference(_config, mp_group=self.mp_group)
        self.module.config.scale_attention = self.scale_attention

        if self.megatron_v2:
            self.module.config.rotate_half = True
            self.module.config.rotate_every_two = False

        return self.module
