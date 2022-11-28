import torch
from ..policy import TransformerPolicy

from packaging import version as pkg_version


class GPTNEOXLayerPolicy(TransformerPolicy):
    _orig_layer_class = None
    version = 0

    def __init__(self, client_module, inference=True, megatron_v2=True):
        super().__init__(inference, megatron_v2=megatron_v2)
        self.client_module = client_module
        if GPTNEOXLayerPolicy._orig_layer_class is None:
            if pkg_version.parse(torch.__version__) <= pkg_version.parse("1.2"):
                GPTNEOXLayerPolicy._orig_layer_class = None
            else:
                try:
                    from transformers import GPTNeoXLayer
                    GPTNEOXLayerPolicy._orig_layer_class = GPTNeoXLayer
                except ImportError:
                    GPTNEOXLayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        if GPTNEOXLayerPolicy.version == 0:
            attention = self.client_module.attention
        else:
            attention = self.client_module.self_attention

        return self.client_module.attention.query_key_value.weight.shape[1], \
                self.client_module.attention.num_attention_heads

    def attention(self):
        if GPTNEOXLayerPolicy.version == 0:
            attention = self.client_module.attention
        else:
            attention = self.client_module.self_attention

        return attention.query_key_value.weight, \
               attention.query_key_value.bias, \
               attention.dense.weight, \
               attention.dense.bias

    def mlp(self):
        return self.client_module.mlp.dense_h_to_4h.weight, \
               self.client_module.mlp.dense_h_to_4h.bias, \
               self.client_module.mlp.dense_4h_to_h.weight, \
               self.client_module.mlp.dense_4h_to_h.bias

    def layernorm(self):
        return self.client_module.post_attention_layernorm.weight, \
               self.client_module.post_attention_layernorm.bias, \
               self.client_module.input_layernorm.weight, \
               self.client_module.input_layernorm.bias
