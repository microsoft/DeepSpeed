from .base import *
from deepspeed.model_implementations.transformers.ds_bert import DeepSpeedBERTInference


class DS_BERTContainer(BaseTransformerContainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # All model specific things should be defined here instead of the base class.
        self.return_tuple = True
        self.triangular_masking = False

    def create_module(self, config=None):
        _config = config if config is not None else self.config
        self.module = DeepSpeedBERTInference(_config,
                                             mp_group=self.mp_group,
                                             qkv_merging=True)
        self.module.config.scale_attention = self.scale_attention
        return self.module
