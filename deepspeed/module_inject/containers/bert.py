from .base import *
from deepspeed.model_implementations.transformers.ds_bert import DeepSpeedBERTInference
from deepspeed.ops.transformer.inference.config import DeepSpeedInferenceConfig


class DS_BERTContainer(BaseTransformerContainer):
    def __init__(self, policy):
        super().__init__(policy)

        # All model specific things should be defined here instead of the base class.
        self.scale_attention = self.policy.scale_attention
        self.layer_norm_eps = 1e-12
        self.attn_linear_layer = True
        self.mlp_linear_layer = True
        self.pre_layer_norm = False
        self.pre_attn_norm = False
        self.triangular_masking = False

    def create_config(self):
        self.config = DeepSpeedInferenceConfig(
            hidden_size=self.hidden_size,
            heads=self.num_attention_heads,
            layer_norm_eps=self.layer_norm_eps,
            fp16=self.fp16,
            pre_layer_norm=self.pre_layer_norm,
            mp_size=self.mp_size,
            triangular_masking=self.triangular_masking)
        return self.config

    def create_module(self, config=None):
        _config = config if config is not None else self.config
        self.module = DeepSpeedBERTInference(_config,
                                             mp_group=self.mp_group,
                                             qkv_merging=True)
        self.module.config.scale_attention = self.scale_attention
        return self.module

    def apply_tensor_parallelism(self, mp_replace):
        # todo: Ask Reza if there is a fixed strategy for this copying and if possible without mp_replace when mp_size=1

        # setup the new Attention module
        #print("bert model comes here ------------------")
        #print(f"attn_block.attn_qkvw: {self.module.attention.attn_qkvw.shape}, {self.qkvw.shape}")
        #attn_block.attn_qkvw = quantizer.quantize(
        #            mp_replace.copy(attn_block.attn_qkvw, qkvw) if bigscience_bloom else \
        #            mp_replace.qkv_copy(attn_block.attn_qkvw, qkvw))

        # Quantizer quantize() is a no-op when q_int8 is False. Tested with bert and it gives correct outputs. See the apply_quant. todo. Test with q_int8=True
        # self.module.attention.attn_qkvw = self.quantizer.quantize(mp_replace.qkv_copy(self.module.attention.attn_qkvw, self.qkvw))
        # self.module.attention.attn_ow = self.quantizer.quantize(mp_replace.copy(self.module.attention.attn_ow, self.dense_w))
        # self.module.mlp.inter_w = self.quantizer.quantize(mp_replace.copy(self.module.mlp.inter_w, self._h4h_w))
        # self.module.mlp.output_w = self.quantizer.quantize(mp_replace.copy(self.module.mlp.output_w, self._4hh_w))

        # setup the new Attention module
        self.module.attention.attn_qkvw = mp_replace.qkv_copy(
            self.module.attention.attn_qkvw,
            self.qkvw)
        self.module.attention.attn_qkvb = mp_replace.qkv_copy(
            self.module.attention.attn_qkvb,
            self.qkvb)
        self.module.attention.attn_ow = mp_replace.copy(self.module.attention.attn_ow,
                                                        self.dense_w)
        self.module.attention.attn_ob = mp_replace.copy(self.module.attention.attn_ob,
                                                        self.dense_b)

        # setup the new MLP module
        self.module.mlp.inter_w = mp_replace.copy(self.module.mlp.inter_w, self._h4h_w)
        self.module.mlp.inter_b = mp_replace.copy(self.module.mlp.inter_b, self._h4h_b)
        self.module.mlp.output_w = mp_replace.copy(self.module.mlp.output_w, self._4hh_w)
        self.module.mlp.output_b = mp_replace.copy(self.module.mlp.output_b, self._4hh_b)

    def apply_weight_quantization(self):
        # quantize attention weights
        self.module.attention.attn_qkvw = self.quantizer.quantize(
            self.module.attention.attn_qkvw)
        self.module.attention.attn_ow = self.quantizer.quantize(
            self.module.attention.attn_ow)
        # quantize mlp weights
        self.module.mlp.inter_w = self.quantizer.quantize(self.module.mlp.inter_w)
        self.module.mlp.output_w = self.quantizer.quantize(self.module.mlp.output_w)
