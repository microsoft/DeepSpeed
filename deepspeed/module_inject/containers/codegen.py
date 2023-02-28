from .base import *
from .features.meta_tensor import MetaTensorContainer
from deepspeed.model_implementations.transformers.ds_codegen import DeepSpeedCodeGenInference
import torch
from torch.nn.parameter import Parameter
from ..policy import TransformerPolicy
from ..policy import transformer_param_names
from ..policy import maybe_copy
from ..policy import maybe_copy_qkv


class DS_CodeGenContainer(MetaTensorContainer, BaseTransformerContainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # All model specific things should be defined here instead of the base class.

    def create_module(self, config=None):
        _config = config if config is not None else self.ds_model_config
        self.module = DeepSpeedCodeGenInference(_config, mp_group=self.mp_group)
        self.module.config.scale_attention = self.scale_attention
        return self.module

    def load_params(self, module, sd, weight_quantizer, mp_replace, prefix):
        param_names = (
            'attn.qkv_proj.weight', \
            'attn.out_proj.weight', \
            'mlp.fc_in.weight', \
            'mlp.fc_in.bias', \
            'mlp.fc_out.weight', \
            'mlp.fc_out.bias', \
            'ln_1.weight', \
            'ln_1.bias'
        )
        maybe_copy(module.attention,
                    sd,
                    weight_quantizer,
                    mp_replace,
                    transformer_param_names[0],
                    prefix + param_names[0],
                    qkv=True,
                    megatron_v2=self.is_megatron_v2,
                    split_qkv=self.split_qkv)
        for i in range(1, 2):
            maybe_copy(module.attention,
                       sd,
                       weight_quantizer,
                       mp_replace,
                       transformer_param_names[i + 1],
                       prefix + param_names[i])
        for i in range(2, 6):
            maybe_copy(module.mlp,
                       sd,
                       weight_quantizer,
                       mp_replace,
                       transformer_param_names[i + 2],
                       prefix + param_names[i])
        for i in range(6, 8):
            maybe_copy(module,
                       sd,
                       weight_quantizer,
                       mp_replace,
                       transformer_param_names[i + 4],
                       prefix + param_names[i])


class HFCodeGenLayerPolicy(TransformerPolicy):
    _orig_layer_class = None

    def __init__(self, client_module, inference=True):
        super().__init__(inference, scale_attention=True)
        self.client_module = client_module
        try:
            import transformers
            HFCodeGenLayerPolicy._orig_layer_class = transformers.models.codegen.modeling_codegen.CodeGenBlock
        except:
            HFCodeGenLayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        return self.client_module.attn.embed_dim, \
                self.client_module.attn.num_attention_heads

    def attention(self):
        return self.client_module.attn.qkv_proj.weight,, \
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