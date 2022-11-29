from .base import *
from deepspeed.model_implementations.transformers.ds_gpt import DeepSpeedGPTInference
from deepspeed.ops.transformer.inference.config import DeepSpeedInferenceConfig


class DS_GPTNEOXContainer(BaseTransformerContainer):
    def __init__(self, policy):
        super().__init__(policy)

        self.attn_linear_layer = True
        self.mlp_linear_layer = True
        self.layer_norm_eps = 1e-05
        self.scale_attention = True
        self.window_size = 1
        self.rotary_dim = 24
        self.mlp_after_attn = False
        self.rotate_half = True
        self.rotate_every_two = False

    def create_config(self):
        self.config = DeepSpeedInferenceConfig(
            hidden_size=self.hidden_size,
            heads=self.num_attention_heads,
            fp16=self.fp16,
            mp_size=self.mp_size,
            layer_norm_eps=self.layer_norm_eps,
            scale_attention=self.scale_attention,
            window_size=self.window_size,
            rotary_dim=self.rotary_dim,
            mlp_after_attn=self.mlp_after_attn,
            rotate_half=self.rotate_half,
            rotate_every_two=self.rotate_every_two,
        )
        return self.config

    def create_module(self, config=None):
        _config = config if config is not None else self.config
        self.module = DeepSpeedGPTInference(_config, mp_group=self.mp_group)
        self.module.config.scale_attention = self.scale_attention
        return self.module

    def transpose(self):
        #if self.attn_linear_layer:
        self.qkvw = self.transpose_impl(self.qkvw.data)
        self.dense_w = self.transpose_impl(self.dense_w.data)

        self.qkvw = torch.nn.parameter.Parameter(self._transpose(self.qkvw).contiguous())
        self.qkvb = torch.nn.parameter.Parameter(self._transpose(self.qkvb).contiguous())

        #if self.mlp_linear_layer:
        self._h4h_w = self.transpose_impl(self._h4h_w.data)
        self._4hh_w = self.transpose_impl(self._4hh_w.data)

    def _transpose(self, x):
        #attention_head_size = x.shape[-1] // transformer_config.heads
        #new_x_shape = x.size()[:-1] + (transformer_config.heads,
        #                                attention_head_size)
        attention_head_size = x.shape[-1] // self.num_attention_heads
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, attention_head_size)
        x_1 = x.view(*new_x_shape)
        (q, k, v) = torch.split(x_1, (x_1.shape[-1] // 3), dim=(x_1.dim() - 1))
        if len(q.shape) > 2:
            return torch.cat((q.reshape(q.shape[0],
                                        -1),
                              k.reshape(q.shape[0],
                                        -1),
                              v.reshape(q.shape[0],
                                        -1)),
                             dim=-1).reshape(x.shape)
        else:
            return torch.cat((q.reshape(-1),
                              k.reshape(-1),
                              v.reshape(-1)),
                             dim=-1).reshape(x.shape)
