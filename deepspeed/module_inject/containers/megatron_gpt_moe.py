from .base import *
from .base_moe import *
from .features.megatron import MegatronContainer
from deepspeed.model_implementations.transformers.ds_megatron_gpt import DeepSpeedMegatronGPTInference


class DS_MegatronGPTMoEContainer(MegatronContainer, BaseTransformerMoEContainer):
    def __init__(self, policy, config, model_config, layer_id):
        super().__init__(policy, config, model_config, layer_id)

        #self.attn_linear_layer = True
        #self.mlp_linear_layer = True
        #self.scale_attention = self.policy.scale_attention
        #self.window_size = 1

    def create_module(self, config=None):
        _config = config if config is not None else self.config
        self.module = DeepSpeedMegatronGPTInference(_config, mp_group=self.mp_group)
        self.module.config.scale_attention = self.scale_attention

        if self.megatron_v2:
            self.module.config.rotate_half = True
            self.module.config.rotate_every_two = False

        return self.module
