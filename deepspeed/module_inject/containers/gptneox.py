from .base import *
from .features.meta_tensor import MetaTensorContainer
from .features.megatron import MegatronContainer
from deepspeed.model_implementations.transformers.ds_gpt import DeepSpeedGPTInference


class DS_GPTNEOXContainer(MetaTensorContainer,
                          MegatronContainer,
                          BaseTransformerContainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # All model specific things should be defined here instead of the base class.

    def create_module(self, config=None):
        _config = config if config is not None else self.config
        self.module = DeepSpeedGPTInference(_config, mp_group=self.mp_group)
        self.module.config.scale_attention = self.scale_attention

        if self.megatron_v2:
            self.module.config.rotate_half = True
            self.module.config.rotate_every_two = False

        return self.module
