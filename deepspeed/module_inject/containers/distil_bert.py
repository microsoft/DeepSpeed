from .base import *
from deepspeed.model_implementations.transformers.ds_bert import DeepSpeedBERTInference


class DS_DistilBERTContainer(BaseTransformerContainer):
    def __init__(self, policy, config, model_config, layer_id):
        super().__init__(policy, config, model_config, layer_id)

        # All model specific things should be defined here instead of the base class.
        #self.scale_attention = self.policy.scale_attention
        #self.layer_norm_eps = 1e-12
        #self.attn_linear_layer = True
        #self.mlp_linear_layer = True
        #self.pre_layer_norm = False
        #self.pre_attn_norm = False
        #self.triangular_masking = False

        self.triangular_masking = False
        self.return_single_tuple = True

    def create_module(self, config=None):
        _config = config if config is not None else self.config
        self.module = DeepSpeedBERTInference(_config,
                                             mp_group=self.mp_group,
                                             qkv_merging=True)
        self.module.config.scale_attention = self.scale_attention
        return self.module
