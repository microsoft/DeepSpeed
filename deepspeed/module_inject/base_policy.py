'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
from abc import ABC


class InjectBasePolicy(ABC):
    def __init__(self,
                 inference=True,
                 linear_layer=True,
                 scale_attention=True,
                 megatron_v2=False):
        self.inference = inference
        self.linear_layer = linear_layer
        self.scale_attention = scale_attention
        self.is_megatron_v2 = megatron_v2

    def attention(self):
        """
        Returns attention qkv and dense parameters
        weight: (3*hidden, hidden) and (hidden, hidden)
        bias: (3*hidden) and (hidden)
        """
        raise NotImplementedError

    def get_hidden_heads(self):
        """
        return hidden_size and number of heads
        """
        raise NotImplementedError

    def mlp(self):
        """
        Returns mlp intermediate and output
        weight: (intermediate, hidden) and (hidden, intermediate)
        bias: (intermediate) and (hidden)
        """
        raise NotImplementedError

    def layerNorm(self):
        """
        Returns LayerNorms used in transformer layer
        Post-Attention and pre/post layer norm
        gamma and beta with shape: (hidden)
        """
        raise NotImplementedError
