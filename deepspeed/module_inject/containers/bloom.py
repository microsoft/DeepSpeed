from .base import *

use_containers = True
if use_containers:
    from deepspeed.model_implementations.transformers.ds_bloom import DeepSpeedBloomInference
else:
    import deepspeed.ops.transformer as transformer_inference

from deepspeed.ops.transformer.inference.config import DeepSpeedInferenceConfig
from ...runtime.zero import GatheredParameters


class DS_BloomContainer(BaseTransformerContainer):
    def __init__(self, policy):
        super().__init__(policy)

        # All model specific things should be defined here instead of the base class.
        self.scale_attention = self.policy.scale_attention
        self.pre_attn_norm = False
        self.attn_linear_layer = True
        self.mlp_linear_layer = True
        self.layer_norm_eps = 1e-05  # hardcode for now, todo: take it from the top config or user args
        self.pre_layer_norm = True
        self.window_size = 1  # hardcode for 3b, todo: take it from the config or user args
        self.bigscience_bloom = True

    def create_config(self):
        self.config = DeepSpeedInferenceConfig(hidden_size=self.hidden_size,
                                               heads=self.num_attention_heads,
                                               layer_norm_eps=self.layer_norm_eps,
                                               fp16=self.fp16,
                                               pre_layer_norm=self.pre_layer_norm,
                                               mp_size=self.mp_size,
                                               window_size=self.window_size,
                                               bigscience_bloom=self.bigscience_bloom)
        return self.config

    def apply_tensor_parallelism(self, mp_replace):
        # Quantizer quantize() is a no-op when q_int8 is False. Tested with bert and it gives correct outputs. See the apply_quant. todo. Test with q_int8=True
        # self.module.attention.attn_qkvw = self.quantizer.quantize(mp_replace.qkv_copy(self.module.attention.attn_qkvw, self.qkvw))
        # self.module.attention.attn_ow = self.quantizer.quantize(mp_replace.copy(self.module.attention.attn_ow, self.dense_w))
        # self.module.mlp.inter_w = self.quantizer.quantize(mp_replace.copy(self.module.mlp.inter_w, self._h4h_w))
        # self.module.mlp.output_w = self.quantizer.quantize(mp_replace.copy(self.module.mlp.output_w, self._4hh_w))
        self.apply_attn_tp(mp_replace)
        self.apply_mlp_tp(mp_replace)

    def apply_attn_tp(self, mp_replace):
        print(f"BLOOM apply_attn_tp")
        # setup the new Attention module
        if self.qkvw.is_meta or self.qkvw.numel() == 0 or self.qkvw.is_meta:
            #print("bert model comes qkvw.is_meta ------------------")
            #exit(0)
            if self.qkvw.is_meta or self.qkvw.ds_tensor.numel(
            ) < self.module.attn_block.attn_qkvw.numel():
                pass
            else:
                print(
                    "bloom model comes into attn. GatheredParameters ------------------")
                #exit(0)
                with GatheredParameters([
                        self.module.attn_block.attn_qkvw,
                        self.module.attn_block.attn_qkvb,
                        self.module.attn_block.attn_ow,
                        self.module.attn_block.attn_ob
                ],
                                        modifier_rank=0):
                    self.module.attn_block.attn_qkvw = mp_replace.copy(
                        self.module.attn_block.attn_qkvw,
                        self.qkvw)
                    self.module.attn_block.attn_qkvb = mp_replace.copy(
                        self.module.attn_block.attn_qkvb,
                        self.qkvb)
                    self.module.attn_block.attn_ow = mp_replace.copy(
                        self.module.attn_block.attn_ow,
                        self.dense_w)
                    self.module.attn_block.attn_ob = mp_replace.copy(
                        self.module.attn_block.attn_ob,
                        self.dense_b)
        else:
            # note that we don't use qkv_copy here and this is bloom specific
            self.module.attention.attn_qkvw = mp_replace.copy(
                self.module.attention.attn_qkvw,
                self.qkvw)
            self.module.attention.attn_qkvb = mp_replace.copy(
                self.module.attention.attn_qkvb,
                self.qkvb)
            self.module.attention.attn_ow = mp_replace.copy(
                self.module.attention.attn_ow,
                self.dense_w)
            self.module.attention.attn_ob = mp_replace.copy(
                self.module.attention.attn_ob,
                self.dense_b)

    def apply_mlp_tp(self, mp_replace):
        print(f"BLOOM apply_mlp_tp")
        # setup the new MLP module
        if self._4hh_w.numel() == 0 or self._4hh_w.is_meta:
            if self._4hh_w.is_meta or self._4hh_w.ds_tensor.numel(
            ) < self.module.mlp.inter_w.numel():
                pass
            else:
                with GatheredParameters(
                    [self._h4h_w,
                     self._4hh_w,
                     self._4hh_w,
                     self._4hh_b],
                        modifier_rank=0):
                    self.module.mlp.inter_w = mp_replace.copy(self.module.mlp.inter_w,
                                                              self._h4h_w)
                    self.module.mlp.inter_b = mp_replace.copy(self.module.mlp.inter_b,
                                                              self._h4h_b)
                    self.module.mlp.output_w = mp_replace.copy(
                        self.module.mlp.output_w,
                        self._4hh_w)
                    self.module.mlp.output_b = mp_replace.copy(
                        self.module.mlp.output_b,
                        self._4hh_b)
        else:
            self.module.mlp.inter_w = mp_replace.copy(self.module.mlp.inter_w,
                                                      self._h4h_w)
            self.module.mlp.inter_b = mp_replace.copy(self.module.mlp.inter_b,
                                                      self._h4h_b)
            self.module.mlp.output_w = mp_replace.copy(self.module.mlp.output_w,
                                                       self._4hh_w)
            self.module.mlp.output_b = mp_replace.copy(self.module.mlp.output_b,
                                                       self._4hh_b)

    def copy_data_to_new_module(self):
        print(f"BLOOM copy_data_to_new_module")
        if self.attn_nw is None:
            self.module.mlp.attn_nw = self.attn_nw
            self.module.mlp.attn_nb = self.attn_nb
        else:
            if self.attn_nw.is_meta or self.attn_nw.numel() == 0:
                if self.attn_nw.is_meta or self.attn_nw.ds_tensor.numel(
                ) < self.module.mlp.attn_nw.numel():
                    pass
                else:
                    with GatheredParameters([self.attn_nw,
                                             self.attn_nb],
                                            modifier_rank=0):
                        self.module.mlp.attn_nw.data.copy_(
                            self.attn_nw.to(torch.cuda.current_device()))
                        self.module.mlp.attn_nb.data.copy_(
                            self.attn_nb.to(torch.cuda.current_device()))
            else:
                self.module.mlp.attn_nw.data.copy_(
                    self.attn_nw.to(torch.cuda.current_device()))
                self.module.mlp.attn_nb.data.copy_(
                    self.attn_nb.to(torch.cuda.current_device()))

            if self.input_nw.is_meta or self.input_nw.numel() == 0:
                if self.input_nw.is_meta or self.input_nw.ds_tensor.numel(
                ) < self.module.norm_w.numel():
                    pass
                else:
                    with GatheredParameters([self.input_nw,
                                             self.input_nb],
                                            modifier_rank=0):
                        self.module.norm_w.data.copy_(
                            self.input_nw.to(torch.cuda.current_device()))
                        self.module.norm_b.data.copy_(
                            self.input_nb.to(torch.cuda.current_device()))
            else:
                self.module.norm_w.data.copy_(
                    self.input_nw.to(torch.cuda.current_device()))
                self.module.norm_b.data.copy_(
                    self.input_nb.to(torch.cuda.current_device()))

    def transpose(self):
        print(f"BLOOM transpose")
        if self.attn_linear_layer:
            if self.qkvw.numel() == 0 or self.qkvw.is_meta:
                if self.qkvw.is_meta or self.qkvw.ds_tensor.numel(
                ) < self.module.attn_block.attn_qkvw.numel():
                    pass
                else:
                    with GatheredParameters(
                        [self.qkvw,
                         self.dense_w,
                         self.qkvb,
                         self.dense_b],
                            modifier_rank=0):
                        self.qkvw = self.transpose_impl(self.qkvw.data)
                        self.dense_w = self.transpose_impl(self.dense_w.data)
                        self.qkvb = self.qkvb.data
                        self.dense_b = self.dense_b.data
            else:
                self.qkvw = self.transpose_impl(self.qkvw.data)
                self.dense_w = self.transpose_impl(self.dense_w.data)

        if self.mlp_linear_layer:
            if self._4hh_w.numel() == 0 or self._4hh_w.is_meta:
                if self._4hh_w.is_meta or self._4hh_w.ds_tensor.numel(
                ) < self.module.mlp.inter_w.numel():
                    pass
                else:
                    with GatheredParameters(
                        [self._h4h_w,
                         self._4hh_w,
                         self._4hh_b,
                         self._h4h_b],
                            modifier_rank=0):
                        self._h4h_w = self.transpose_impl(self._h4h_w.data)
                        self._4hh_w = self.transpose_impl(self._4hh_w.data)
                        self._h4h_b = self._h4h_b.data
                        self._4hh_b = self._4hh_b.data
            else:
                self._h4h_w = self.transpose_impl(self._h4h_w.data)
                self._4hh_w = self.transpose_impl(self._4hh_w.data)

    def create_module(self, config=None):
        print(f"BLOOM create_module")
        _config = config if config is not None else self.config

        if use_containers:
            self.module = DeepSpeedBloomInference(
                _config,
                mp_group=self.mp_group)  # w/ containers
        else:
            self.module = transformer_inference.DeepSpeedTransformerInference(
                _config,
                mp_group=self.mp_group)  # w/o containers
        self.module.config.scale_attention = self.scale_attention
        return self.module
