from .base import *
from deepspeed.model_implementations.transformers.ds_megatron_gpt import DeepSpeedMegatronGPTInference
from deepspeed.ops.transformer.inference.config import DeepSpeedInferenceConfig


class DS_MegatronGPTContainer(BaseTransformerContainer):
    def __init__(self, policy):
        super().__init__(policy)

        self.megatron_v2 = self.policy.is_megatron_v2

        self.attn_linear_layer = True
        self.mlp_linear_layer = True
        self.scale_attention = True
        self.window_size = 1

        # Create DS_MegatronGPTMoE container
        # Override MLP part
        # new variables for tensors (list)

    def create_config(self):
        self.config = DeepSpeedInferenceConfig(hidden_size=self.hidden_size,
                                               heads=self.num_attention_heads,
                                               fp16=self.fp16,
                                               mp_size=self.mp_size,
                                               window_size=self.window_size)
        return self.config

    def create_module(self, config=None):
        _config = config if config is not None else self.config
        self.module = DeepSpeedMegatronGPTInference(_config, mp_group=self.mp_group)
        self.module.config.scale_attention = self.scale_attention

        if self.megatron_v2:
            self.module.config.rotate_half = True
            self.module.config.rotate_every_two = False

        return self.module

    def transpose(self):
        if self.attn_linear_layer:
            self.qkvw = self.transpose_impl(self.qkvw.data)
            self.dense_w = self.transpose_impl(self.dense_w.data)

        if self.megatron_v2:
            self.qkvw = torch.nn.parameter.Parameter(
                self.transpose_qkv_alignment(self.qkvw).contiguous())
            self.qkvb = torch.nn.parameter.Parameter(
                self.transpose_qkv_alignment(self.qkvb).contiguous())

        if self.mlp_linear_layer:
            self._h4h_w = self.transpose_impl(self._h4h_w.data)
            self._4hh_w = self.transpose_impl(self._4hh_w.data)
