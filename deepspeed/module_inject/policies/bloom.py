'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
from ..policy import TransformerPolicy

supported_models = {None}


class BLOOMLayerPolicy(TransformerPolicy):
    _orig_layer_class = None

    def __init__(self,
                 client_module,
                 inference=True,
                 use_load_prefix=True,
                 split_qkv=False):
        super().__init__(inference, linear_layer=True)
        self.client_module = client_module
        try:
            import transformers
            BLOOMLayerPolicy._orig_layer_class = transformers.models.bloom.modeling_bloom.BloomBlock
            global supported_models
            supported_models.update(
                {transformers.models.bloom.modeling_bloom.BloomModel})
        except Exception as e:
            print(
                f"WARNING! Setting BLOOMLayerPolicy._orig_layer_class to None due to Exception: {e}"
            )
            BLOOMLayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        return self.client_module.self_attention.hidden_size, \
                self.client_module.self_attention.num_heads

    def attention(self):
        return self.client_module.self_attention.query_key_value.weight, \
                self.client_module.self_attention.query_key_value.bias, \
                self.client_module.self_attention.dense.weight, \
                self.client_module.self_attention.dense.bias,

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

    def get_param_names(self):
        return 'self_attention.query_key_value.weight', \
               'self_attention.query_key_value.bias', \
               'self_attention.dense.weight', \
               'self_attention.dense.bias', \
               'mlp.dense_h_to_4h.weight', \
               'mlp.dense_h_to_4h.bias', \
               'mlp.dense_4h_to_h.weight', \
               'mlp.dense_4h_to_h.bias', \
               'input_layernorm.weight', \
               'input_layernorm.bias', \
               'post_attention_layernorm.weight', \
               'post_attention_layernorm.bias', \
               self.use_load_prefix, \
               self.split_qkv
