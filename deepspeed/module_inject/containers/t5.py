import torch
from .base import BaseTransformerContainer
from ...model_implementations.encoder_decoder.ds_t5 import DeepSpeedT5Inference
from ..policy import TransformerPolicy


class DS_T5Container(BaseTransformerContainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # All model specific things should be defined here instead of the base class.
        print("Using DS_T5Container")

    def create_module(self, config=None):
        _config = config if config is not None else self.ds_model_config
        self.module = DeepSpeedT5Inference(_config,self.child, mp_group=self.mp_group)
        return self.module

    def set_attention(self, attn_linear_layer, qkvw, qkvb, dense_w, dense_b, scale_attention, relative_bias):
        self.attn_linear_layer = attn_linear_layer
        self.qkvw = qkvw
        self.qkvb = qkvb
        self.dense_w = dense_w
        self.dense_b = dense_b
        self.scale_attention = scale_attention
        self.relative_bias = relative_bias

    def set_mlp(self, mlp_linear_layer, _h4h_w, _h4h_b, _h4h_w2, _h4h_b2, _4hh_w, _4hh_b):
        self.mlp_linear_layer = mlp_linear_layer
        self._h4h_w = _h4h_w
        self._h4h_b = _h4h_b
        self._h4h_w2 = _h4h_w2
        self._h4h_b2 = _h4h_b2
        self._4hh_w = _4hh_w
        self._4hh_b = _4hh_b

    def set_hidden_heads(self, hidden_size, num_attention_heads,intermediate_size):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

    def apply_tensor_parallelism(self, module):
        pass

    def copy_data_to_new_module(self,):
        pass


class HFT5LayerPolicy(TransformerPolicy):
    _orig_layer_class = None

    def __init__(self, client_module, inference=True):
        # HuggingFace GPT2 uses convolutional layer instead of linear layer
        super().__init__(inference, scale_attention=False)
        self.client_module = client_module
        try:
            import transformers
            HFT5LayerPolicy._orig_layer_class = transformers.models.t5.modeling_t5.T5Block
        except ImportError:
            HFT5LayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        return self.client_module.layer[0].SelfAttention.d_model, \
                self.client_module.layer[0].SelfAttention.n_heads, \
                self.client_module.layer[-1].DenseReluDense.wi_0.out_features

    def attention(self):
        qw = self.client_module.layer[0].SelfAttention.q.weight.data
        kw = self.client_module.layer[0].SelfAttention.k.weight.data
        vw = self.client_module.layer[0].SelfAttention.v.weight.data

        qkvw = torch.cat((qw, kw, vw), dim=0)
        return self.linear_layer, \
                qkvw, \
                None, \
                self.client_module.layer[0].SelfAttention.o.weight.data.clone(), \
                None, \
                self.scale_attention, \
                self.client_module.layer[0].SelfAttention.relative_attention_bias if \
                    self.client_module.layer[0].SelfAttention.has_relative_attention_bias else None

    def mlp(self):
        return self.linear_layer, \
            self.client_module.layer[-1].DenseReluDense.wi_0.weight.data.clone(), \
            None, \
            self.client_module.layer[-1].DenseReluDense.wi_1.weight.data.clone(), \
            None, \
            self.client_module.layer[-1].DenseReluDense.wo.weight.data.clone(), \
            None

    def layernorm(self):
        return self.client_module.layer[-1].layer_norm.weight.data, \
               None, \
               self.client_module.layer[0].layer_norm.weight.data, \
               None

    def decoder(self):
        qw = self.client_module.layer[1].EncDecAttention.q.weight.data
        kw = self.client_module.layer[1].EncDecAttention.k.weight.data
        vw = self.client_module.layer[1].EncDecAttention.v.weight.data

        qkvw = torch.cat((qw, kw, vw), dim=0)
        return self.linear_layer, \
                qkvw, \
                None, \
                self.client_module.layer[1].EncDecAttention.o.weight.data, \
                None, \
                self.scale_attention, \
                self.client_module.layer[1].layer_norm.weight.data, \
                self.client_module.layer[1].EncDecAttention.relative_attention_bias if \
                    self.client_module.layer[1].EncDecAttention.has_relative_attention_bias else None