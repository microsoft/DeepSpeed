from .base import *
from deepspeed.model_implementations.transformers.ds_gpt import DeepSpeedGPTInference
import torch
from torch.nn.parameter import Parameter
from ..policy import TransformerPolicy


class DS_CLIPContainer(BaseTransformerContainer):
    def __init__(self, policy, config, model_config, layer_id):
        super().__init__(policy, config, model_config, layer_id)

        #self.attn_linear_layer = False
        #self.mlp_linear_layer = False
        #self.scale_attention = self.policy.scale_attention
        #self.layer_norm_eps = 1e-5
        #self.pre_layer_norm = True

    def create_module(self, config=None):
        _config = config if config is not None else self.config
        self.module = DeepSpeedGPTInference(_config, mp_group=self.mp_group)
        self.module.config.scale_attention = self.scale_attention
        return self.module


class HFCLIPLayerPolicy(TransformerPolicy):
    def __init__(self, client_module, inference=False):
        super().__init__(inference, pre_attn_norm=True, scale_attention=True)
        self.client_module = client_module
        self.cuda_graph_supported = True

        if HFCLIPLayerPolicy._orig_layer_class is None:
            try:
                import transformers
                HFCLIPLayerPolicy._orig_layer_class = transformers.models.clip.modeling_clip.CLIPEncoderLayer
            except:
                HFCLIPLayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        return self.client_module.self_attn.q_proj.weight.shape[1], \
                self.client_module.self_attn.num_heads

    def attention(self):
        qw = self.client_module.self_attn.q_proj.weight
        qb = self.client_module.self_attn.q_proj.bias
        kw = self.client_module.self_attn.k_proj.weight
        kb = self.client_module.self_attn.k_proj.bias
        vw = self.client_module.self_attn.v_proj.weight
        vb = self.client_module.self_attn.v_proj.bias

        qkvw = Parameter(torch.cat((qw, kw, vw), dim=0), requires_grad=False)
        qkvb = Parameter(torch.cat((qb, kb, vb), dim=0), requires_grad=False)

        return self.linear_layer, \
               qkvw, \
               qkvb, \
               self.client_module.self_attn.out_proj.weight, \
               self.client_module.self_attn.out_proj.bias, \
               self.scale_attention, \
               self.is_megatron_v2

    def mlp(self):
        return self.linear_layer, \
            self.client_module.mlp.fc1.weight, \
            self.client_module.mlp.fc1.bias, \
            self.client_module.mlp.fc2.weight, \
            self.client_module.mlp.fc2.bias

    def layerNorm(self):
        return self.client_module.layer_norm2.weight, \
               self.client_module.layer_norm2.bias, \
               self.client_module.layer_norm1.weight, \
               self.client_module.layer_norm1.bias
