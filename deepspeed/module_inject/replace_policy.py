from abc import ABC

import torch


class DSPolicy(ABC):
    def __init__(self, inference=True, linear_layer=True, scale_attention=True):
        self.inference = inference
        self.linear_layer = linear_layer
        self.scale_attention = scale_attention

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

    def decoder(self):
        """
        Returns LayerNorms used in transformer layer
        Post-Attention and pre/post layer norm
        gamma and beta with shape: (hidden)
        """
        raise NotImplementedError


class HFBertLayerPolicy(DSPolicy):
    _orig_layer_class = None

    def __init__(self, client_module, inference=False, preln=False):
        super().__init__(inference)
        self.client_module = client_module
        self.preln = preln
        if HFBertLayerPolicy._orig_layer_class is None:
            try:
                import transformers
                HFBertLayerPolicy._orig_layer_class = transformers.models.bert.modeling_bert.BertLayer
            except:
                HFBertLayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        return self.client_module.attention.self.query.weight.data.shape[1], \
                self.client_module.attention.self.num_attention_heads

    def attention(self):
        qw = self.client_module.attention.self.query.weight.data
        qb = self.client_module.attention.self.query.bias.data
        kw = self.client_module.attention.self.key.weight.data
        kb = self.client_module.attention.self.key.bias.data
        vw = self.client_module.attention.self.value.weight.data
        vb = self.client_module.attention.self.value.bias.data

        qkvw = torch.cat((qw, kw, vw), dim=0)
        qkvb = torch.cat((qb, kb, vb), dim=0)

        return self.linear_layer, \
               qkvw, \
               qkvb, \
               self.client_module.attention.output.dense.weight.data, \
               self.client_module.attention.output.dense.bias.data, \
               self.scale_attention

    def mlp(self):
        if self.preln:
            intermediate_ff = self.client_module.intermediate.dense_act
        else:
            intermediate_ff = self.client_module.intermediate.dense

        return self.linear_layer, intermediate_ff.weight.data, intermediate_ff.bias.data, \
            self.client_module.output.dense.weight.data, \
            self.client_module.output.dense.bias.data

    def layerNorm(self):
        if self.preln:
            attention_layernorm = self.client_module.PostAttentionLayerNorm
            transformer_layernorm = self.client_module.PreAttentionLayerNorm
        else:
            attention_layernorm = self.client_module.attention.output.LayerNorm
            transformer_layernorm = self.client_module.output.LayerNorm
        return attention_layernorm.weight.data, \
               attention_layernorm.bias.data, \
               transformer_layernorm.weight.data, \
               transformer_layernorm.bias.data


class HFGPTNEOLayerPolicy(DSPolicy):
    _orig_layer_class = None

    def __init__(self, client_module, inference=True):
        super().__init__(inference, scale_attention=False)
        self.client_module = client_module
        try:
            import transformers
            HFGPTNEOLayerPolicy._orig_layer_class = transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoBlock
        except:
            HFGPTNEOLayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        return self.client_module.attn.attention.q_proj.weight.data.shape[1], \
                self.client_module.attn.attention.num_heads

    def attention(self):
        qw = self.client_module.attn.attention.q_proj.weight.data
        kw = self.client_module.attn.attention.k_proj.weight.data
        vw = self.client_module.attn.attention.v_proj.weight.data

        qkvw = torch.cat((qw, kw, vw), dim=0)

        return self.linear_layer, \
                qkvw, \
                None, \
                self.client_module.attn.attention.out_proj.weight.data, \
                self.client_module.attn.attention.out_proj.bias.data, \
                self.scale_attention

    def mlp(self):
        return self.linear_layer, \
                self.client_module.mlp.c_fc.weight.data, \
                self.client_module.mlp.c_fc.bias.data, \
                self.client_module.mlp.c_proj.weight.data, \
                self.client_module.mlp.c_proj.bias.data

    def layerNorm(self):
        return self.client_module.ln_2.weight.data, \
               self.client_module.ln_2.bias.data, \
               self.client_module.ln_1.weight.data, \
               self.client_module.ln_1.bias.data


class MegatronLayerPolicy(DSPolicy):
    _orig_layer_class = None

    def __init__(self, client_module, version=0, inference=True):
        super().__init__(inference)
        self.client_module = client_module
        # we use megatron version to differentiate between the old and new
        # megatron-lm source code
        self.version = version
        if MegatronLayerPolicy._orig_layer_class is None:
            try:
                import megatron
                from megatron.model.transformer import ParallelTransformerLayer
                MegatronLayerPolicy._orig_layer_class = ParallelTransformerLayer
            except ImportError:
                MegatronLayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        return self.client_module.attention.query_key_value.weight.data.shape[1], \
                self.client_module.attention.num_attention_heads

    def attention(self):
        if self.inference:
            if self.version == 0:
                attention = self.client_module.attention
            else:
                attention = self.client_module.self_attention

        return self.linear_layer, \
                attention.query_key_value.weight.data, \
                attention.query_key_value.bias.data, \
                attention.dense.weight.data, \
                attention.dense.bias.data, \
                self.scale_attention

    def mlp(self):
        return self.linear_layer, \
            self.client_module.mlp.dense_h_to_4h.weight.data, \
            self.client_module.mlp.dense_h_to_4h.bias.data, \
            self.client_module.mlp.dense_4h_to_h.weight.data, \
            self.client_module.mlp.dense_4h_to_h.bias.data

    def layerNorm(self):
        return self.client_module.post_attention_layernorm.weight.data, \
               self.client_module.post_attention_layernorm.bias.data, \
               self.client_module.input_layernorm.weight.data, \
               self.client_module.input_layernorm.bias.data


class HFGPT2LayerPolicy(DSPolicy):
    _orig_layer_class = None

    def __init__(self, client_module, inference=True):
        # HuggingFace GPT2 uses convolutional layer instead of linear layer
        super().__init__(inference, linear_layer=False)
        self.client_module = client_module
        try:
            import transformers
            HFGPT2LayerPolicy._orig_layer_class = transformers.models.gpt2.modeling_gpt2.GPT2Block
        except ImportError:
            HFGPT2LayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        return self.client_module.attn.embed_dim, \
                self.client_module.attn.num_heads

    def attention(self):
        return self.linear_layer, \
                self.client_module.attn.c_attn.weight.data, \
                self.client_module.attn.c_attn.bias.data, \
                self.client_module.attn.c_proj.weight.data, \
                self.client_module.attn.c_proj.bias.data, \
                self.scale_attention

    def mlp(self):
        return self.linear_layer, \
            self.client_module.mlp.c_fc.weight.data, \
            self.client_module.mlp.c_fc.bias.data, \
            self.client_module.mlp.c_proj.weight.data, \
            self.client_module.mlp.c_proj.bias.data

    def layerNorm(self):
        return self.client_module.ln_2.weight.data, \
               self.client_module.ln_2.bias.data, \
               self.client_module.ln_1.weight.data, \
               self.client_module.ln_1.bias.data


class HFT5LayerPolicy(DSPolicy):
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
                self.client_module.layer[0].SelfAttention.o.weight.data, \
                None, \
                self.scale_attention, \
                self.client_module.layer[0].SelfAttention.relative_attention_bias if \
                    self.client_module.layer[0].SelfAttention.has_relative_attention_bias else None

    def mlp(self):
        return self.linear_layer, \
            self.client_module.layer[-1].DenseReluDense.wi_0.weight.data, \
            None, \
            self.client_module.layer[-1].DenseReluDense.wi_1.weight.data, \
            None, \
            self.client_module.layer[-1].DenseReluDense.wo.weight.data, \
            None

    def layerNorm(self):
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


replace_policies = [
    HFBertLayerPolicy,
    HFGPTNEOLayerPolicy,
    MegatronLayerPolicy,
    HFGPT2LayerPolicy,
    HFT5LayerPolicy,
]
