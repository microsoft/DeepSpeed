from .base import *
from .features.meta_tensor import MetaTensorContainer
from deepspeed.model_implementations.transformers.ds_bloom import DeepSpeedBloomInference


class DS_BloomContainer(MetaTensorContainer, BaseTransformerContainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # All model specific things should be defined here instead of the base class.
        self.bigscience_bloom = True

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
