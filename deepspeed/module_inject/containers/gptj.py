from .base import *
from .features.meta_tensor import MetaTensorContainer
from deepspeed.model_implementations.transformers.ds_gpt import DeepSpeedGPTInference
import torch
from torch.nn.parameter import Parameter
from ..policy import TransformerPolicy
from ..policy import transformer_param_names
from ..policy import maybe_copy
from ..policy import maybe_copy_qkv


class DS_GPTJContainer(MetaTensorContainer, BaseTransformerContainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # All model specific things should be defined here instead of the base class.

    def create_module(self, config=None):
        _config = config if config is not None else self.ds_model_config
        self.module = DeepSpeedGPTInference(_config, mp_group=self.mp_group)
        self.module.config.scale_attention = self.scale_attention
        return self.module

    def load_params(self, module, sd, weight_quantizer, mp_replace, prefix):
        param_names = (
            'attn.q_proj.weight', \
            'attn.k_proj.weight', \
            'attn.v_proj.weight', \
            'attn.out_proj.weight', \
            'mlp.fc_in.weight', \
            'mlp.fc_in.bias', \
            'mlp.fc_out.weight', \
            'mlp.fc_out.bias', \
            'ln_1.weight', \
            'ln_1.bias'
        )
        maybe_copy_qkv(
            module.attention,
            sd,
            weight_quantizer,
            mp_replace,
            'attn_qkvw',
            [prefix + param_names[0],
             prefix + param_names[1],
             prefix + param_names[2]],
            split_qkv=self.split_qkv)
        for i in range(3, 4):
            maybe_copy(module.attention,
                       sd,
                       weight_quantizer,
                       mp_replace,
                       transformer_param_names[i - 1],
                       prefix + param_names[i])
        for i in range(4, 8):
            maybe_copy(module.mlp,
                       sd,
                       weight_quantizer,
                       mp_replace,
                       transformer_param_names[i],
                       prefix + param_names[i])
        for i in range(8, 10):
            maybe_copy(module,
                       sd,
                       weight_quantizer,
                       mp_replace,
                       transformer_param_names[i + 2],
                       prefix + param_names[i])


class HFGPTJLayerPolicy(TransformerPolicy):
    _orig_layer_class = None

    def __init__(self, client_module, inference=True):
        super().__init__(inference, scale_attention=True)
        self.client_module = client_module
        try:
            import transformers
            HFGPTJLayerPolicy._orig_layer_class = [
                transformers.models.gptj.modeling_gptj.GPTJBlock,
                transformers.models.codegen.modeling_codegen.CodeGenBlock
            ]
        except:
            HFGPTJLayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        return self.client_module.attn.out_proj.weight.shape[1], \
                self.client_module.attn.num_attention_heads

    def attention(self):
        if self.client_module.__class__ == HFGPTJLayerPolicy._orig_layer_class[0]:
            qw = self.client_module.attn.q_proj.weight
            kw = self.client_module.attn.k_proj.weight
            vw = self.client_module.attn.v_proj.weight
            qkvw = Parameter(torch.cat((qw, kw, vw), dim=0), requires_grad=False)
        else:  # this branch is added to support codegen which is based on gpt-j
            mp_num = 4
            qkvw = self.client_module.attn.qkv_proj.weight
            hidden_size = qkvw.shape[1]
            local_dim = hidden_size // mp_num
            qkvw = qkvw.reshape(mp_num, -1, hidden_size)
            qkvw_split = torch.split(qkvw, local_dim, dim=-2)
            qkvw = torch.cat([qkvw_split[0].reshape(-1, hidden_size), \
                             qkvw_split[2].reshape(-1, hidden_size), \
                             qkvw_split[1].reshape(-1, hidden_size)], dim=0).contiguous()
        return qkvw, \
               None, \
               self.client_module.attn.out_proj.weight, \
               None,

    def mlp(self):
        return self.client_module.mlp.fc_in.weight, \
               self.client_module.mlp.fc_in.bias, \
               self.client_module.mlp.fc_out.weight, \
               self.client_module.mlp.fc_out.bias

    def layernorm(self):
        return None, \
               None, \
               self.client_module.ln_1.weight, \
               self.client_module.ln_1.bias
