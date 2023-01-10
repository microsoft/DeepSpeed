from .base import *
from .features.meta_tensor import MetaTensorContainer
from deepspeed.model_implementations.transformers.ds_opt import DeepSpeedOPTInference


class DS_OPTContainer(MetaTensorContainer, BaseTransformerContainer):
    def __init__(self, policy, config, model_config, layer_id):
        super().__init__(policy, config, model_config, layer_id)

        #self.attn_linear_layer = True
        #self.mlp_linear_layer = True
        #self.scale_attention = self.policy.scale_attention
        #self.mlp_act_func_type = self.policy.mlp_act_func_type
        #self.window_size = 1

    def create_module(self, config=None):
        _config = config if config is not None else self.config
        self.module = DeepSpeedOPTInference(_config, mp_group=self.mp_group)
        self.module.config.scale_attention = self.scale_attention
        return self.module
