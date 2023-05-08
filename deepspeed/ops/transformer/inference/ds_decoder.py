'''
Copyright 2020 The Microsoft DeepSpeed Team
'''
import json
import math
import importlib
import torch
from torch import nn
from torch.autograd import Function
import time
from ... import op_builder
#from ...inference.engine import inference_cuda_module, specialized_mode
# Cuda modules will be imported if needed
inference_cuda_module = None
specialized_mode = None
import torch.nn as nn

F = nn.functional


class DeepSpeedSelfCrossAttentionFunction(Function):
    @staticmethod
    def forward(ctx,
                input,
                input_mask,
                head_mask,
                layer_past,
                get_present,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
                norm_w,
                norm_b,
                config,
                attn_qkvw,
                attn_qkvb,
                num_attention_heads_per_partition,
                norm_factor,
                hidden_size_per_partition,
                attn_ow,
                attn_ob,
                mp_group,
                q_scales,
                q_groups,
                merge_count,
                qkv_merging):
        def _transpose_for_scores(x, key=False, reshape=False):
            attention_head_size = x.shape[-1] // num_attention_heads_per_partition
            new_x_shape = x.size()[:-1] + (num_attention_heads_per_partition,
                                           attention_head_size)
            x_1 = x.view(*new_x_shape)
            if key:
                x_1 = x_1.permute(0, 2, 3, 1)
            else:
                x_1 = x_1.permute(0, 2, 1, 3)
            if reshape:
                return x_1.reshape(x.shape)
            return x_1

        def _transpose_for_context(x):
            x = x.permute(0, 2, 1, 3).contiguous()
            new_x_layer_shape = x.size()[:-2] + \
                                      (-1,)
            return x.view(*new_x_layer_shape)

        def compute_attention(qkv_out, input_mask):
            score_context_func = inference_cuda_module.softmax_context_fp32 if (not config.fp16) else \
                                    inference_cuda_module.softmax_context_fp16
            #if not config.triangular_masking:
            #    qkv_out = qkv_out.float()

            if merge_count > 0 and config.q_int8:
                split_dim = (qkv_out.dim() - 1)
                qkv_split = torch.split(qkv_out,
                                        (qkv_out.shape[-1] // (2**merge_count)),
                                        dim=split_dim)
                qkv_split = [
                    torch.split(s,
                                (s.shape[-1] // 3),
                                dim=split_dim) for s in qkv_split
                ]
                (mixed_query,
                 key_layer,
                 value_layer) = [
                     torch.cat([s[i] for s in qkv_split],
                               axis=-1) for i in range(len(qkv_split[0]))
                 ]
            else:
                (mixed_query,
                 key_layer,
                 value_layer) = torch.split(qkv_out,
                                            (qkv_out.shape[-1] // 3),
                                            dim=(qkv_out.dim() - 1))
            no_masking = input_mask is None
            if no_masking:
                input_mask = torch.empty(1)
            head_size = (mixed_query.shape[-1] // num_attention_heads_per_partition)
            unfused_mode = not config.specialized_mode or \
                                mixed_query.shape[1] >= 32 or head_size > 128

            if layer_past is not None:
                past_key, past_value = layer_past
                if unfused_mode:
                    key_layer = torch.cat((past_key.type_as(key_layer),
                                           key_layer),
                                          dim=-2)
                    value_layer = torch.cat((past_value.type_as(value_layer),
                                             value_layer),
                                            dim=-2)
            if unfused_mode:
                mixed_query = _transpose_for_scores(mixed_query, False, True)
                key_layer1 = _transpose_for_scores(
                    key_layer,
                    True,
                    True) / (norm_factor if config.scale_attention else 1.0)
                value_layer1 = _transpose_for_scores(value_layer, False, True)

            if layer_past is None:
                attn_key_value = score_context_func(
                    mixed_query,
                    (key_layer1 if unfused_mode else key_layer),
                    torch.empty(1),
                    (input_mask),
                    (value_layer1 if unfused_mode else value_layer),
                    torch.empty(1),
                    num_attention_heads_per_partition,
                    (1 / norm_factor if config.scale_attention else 1.0),
                    (not unfused_mode),
                    config.triangular_masking,
                    config.local_attention,
                    config.window_size,
                    no_masking)
            else:
                attn_key_value = score_context_func(
                    mixed_query,
                    (key_layer1 if unfused_mode else past_key.type_as(key_layer)),
                    (key_layer1 if unfused_mode else key_layer),
                    (input_mask),
                    (value_layer1 if unfused_mode else past_value.type_as(value_layer)),
                    (value_layer1 if unfused_mode else value_layer),
                    num_attention_heads_per_partition,
                    (1 / norm_factor if config.scale_attention else 1.0),
                    (not unfused_mode),
                    config.triangular_masking,
                    config.local_attention,
                    config.window_size,
                    no_masking)
            #import pdb;pdb.set_trace()
            if unfused_mode:
                context_layer, _, _ = attn_key_value
            else:
                context_layer, key_layer, value_layer = attn_key_value

            # Transpose Context
            context_layer = _transpose_for_context(context_layer)
            #if (config.fp16 or config.q_int8) and not config.triangular_masking:
            #    context_layer = context_layer.half()

            return context_layer, key_layer, value_layer

        def selfAttention_fp():
            vector_matmul_func = inference_cuda_module.vector_matmul_fp16 if config.fp16 else \
                                    inference_cuda_module.vector_matmul_fp32
            if not config.pre_layer_norm:
                linear_func = inference_cuda_module.linear_layer_fp16 if config.fp16 else \
                                    inference_cuda_module.linear_layer_fp32

                qkv_out = linear_func(input, attn_qkvw, attn_qkvb)
            else:
                qkv_func = inference_cuda_module.qkv_gemm_fp16 if config.fp16 else \
                                    inference_cuda_module.qkv_gemm_fp32
                qkv_out = qkv_func(input,
                                   attn_qkvw,
                                   (attn_qkvb if attn_qkvb is not None else attn_qkvw),
                                   norm_w,
                                   (norm_b if norm_b is not None else norm_w),
                                   config.epsilon,
                                   (attn_qkvb is not None),
                                   (norm_b is not None))
            context_layer, key_layer, value_layer = compute_attention(qkv_out, input_mask)
            output = vector_matmul_func(context_layer, attn_ow)

            return output, key_layer, value_layer, context_layer

        def selfAttention_int8():
            if not config.pre_layer_norm:
                qkv_out = inference_cuda_module.linear_layer_int8(
                    input,
                    attn_qkvw,
                    attn_qkvb,
                    q_scales[0],
                    (q_groups * (3 if qkv_merging else 1) * (2**merge_count)))
            else:
                qkv_out = inference_cuda_module.qkv_gemm_int8(
                    input,
                    attn_qkvw,
                    attn_qkvb,
                    norm_w,
                    norm_b,
                    config.epsilon,
                    q_scales[0],
                    (q_groups * (3 if qkv_merging else 1) * (2**merge_count)),
                    (attn_qkvb is not None))
            context_layer, key_layer, value_layer = compute_attention(qkv_out)
            output = inference_cuda_module.vector_matmul_int8(context_layer,
                                                              attn_ow,
                                                              q_scales[1],
                                                              q_groups,
                                                              (merge_count))
            return output, key_layer, value_layer, context_layer

        if config.q_int8:
            output, key_layer, value_layer, context_layer = selfAttention_int8()
        else:
            output, key_layer, value_layer, context_layer = selfAttention_fp()

        if mp_group is not None and torch.distributed.get_world_size(group=mp_group) > 1:
            torch.distributed.all_reduce(output, group=mp_group)

        return (output, key_layer, value_layer, context_layer)

    @staticmethod
    def backward(ctx, grad_output, grad_output1, grad_output2, grad_output3):
        raise RuntimeError('You are running with DeepSpeed Inference mode. \
                            Please switch to Training mode for running backward!')


class DeepSpeedSelfCrossAttention(nn.Module):
    def __init__(self,
                 config,
                 mp_group=None,
                 q_scales=None,
                 q_groups=1,
                 merge_count=1,
                 qkv_merging=False):
        super(DeepSpeedSelfCrossAttention, self).__init__()
        self.config = config

        self.attn_qkvw = nn.Parameter(
            torch.Tensor(self.config.hidden_size,
                         (self.config.hidden_size // self.config.mp_size) * 3))
        self.attn_qkvb = nn.Parameter(
            torch.Tensor((self.config.hidden_size // self.config.mp_size) * 3))

        self.attn_ow = nn.Parameter(
            torch.Tensor(self.config.hidden_size // self.config.mp_size,
                         self.config.hidden_size))

        self.attn_ob = nn.Parameter(torch.Tensor(self.config.hidden_size))

        self.num_attention_heads_per_partition = self.config.heads // self.config.mp_size
        self.hidden_size_per_partition = self.config.hidden_size // self.config.mp_size
        self.hidden_size_per_attention_head = self.config.hidden_size // self.config.heads

        self.mp_group = mp_group

        # used for quantization
        self.q_scales = q_scales
        self.q_groups = q_groups
        self.merge_count = int(math.log2(merge_count))

        self.norm_factor = math.sqrt(
            math.sqrt(self.config.hidden_size // self.config.heads))
        self.qkv_merging = qkv_merging

        self.relative_attention_num_buckets = 32
        self.relative_attention_bias_weight = None

    def compute_position_baias(self, query_length, key_length):
        # Adapted from https://github.com/huggingface/transformers/blob/513fa30a636642ccc1d93f3e6a48d612d08dbce8/src/transformers/models/t5/modeling_t5.py#L394
        import transformers
        from transformers.models.t5.modeling_t5 import T5Attention
        """Compute binned relative position bias"""
        context_position = torch.arange(
            query_length,
            dtype=torch.long,
            device=self.relative_attention_bias_weight.device)[:,
                                                               None]
        memory_position = torch.arange(
            key_length, dtype=torch.long, device=self.relative_attention_bias_weight.device
        )[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = T5Attention._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=False,
            num_buckets=self.relative_attention_num_buckets,
        )
        values = F.embedding(relative_position_bucket,
                             self.relative_attention_bias_weight)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def forward(self,
                input,
                input_mask,
                head_mask=None,
                layer_past=None,
                get_present=False,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=False,
                norm_w=None,
                norm_b=None,
                position_bias=None):

        if position_bias is None:
            if self.relative_attention_bias_weight is not None:
                position_bias = self.compute_position_baias(
                    input.shape[1] if layer_past is None else
                    (input.shape[1] + layer_past[0].shape[1]),
                    input.shape[1] if layer_past is None else layer_past[0].shape[1])
            if layer_past is not None and position_bias is not None:
                position_bias = position_bias[:, :, -input.size(1):, :]

        output = DeepSpeedSelfCrossAttentionFunction.apply(
            input,
            input_mask,
            head_mask,
            layer_past,
            get_present,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
            norm_w,
            norm_b,
            self.config,
            self.attn_qkvw,
            self.attn_qkvb,
            self.num_attention_heads_per_partition,
            self.norm_factor,
            self.hidden_size_per_partition,
            self.attn_ow,
            self.attn_ob,
            self.mp_group,
            self.q_scales,
            self.q_groups,
            self.merge_count,
            self.qkv_merging,
        )

        return output + (position_bias, )


class DeepSpeedCrossMLPFunction(Function):
    @staticmethod
    def forward(ctx,
                input,
                residual,
                bias,
                inter_w,
                inter_b,
                attn_nw,
                attn_nb,
                config,
                mp_group,
                output_b,
                output_w,
                q_scales,
                q_groups,
                merge_count,
                inter_w1):
        if config.q_int8:

            (intermediate,
             residual_add) = inference_cuda_module.mlp_gemm_int8(
                 input,
                 residual,
                 (bias if bias is not None else attn_nw),
                 inter_w,
                 (inter_b if inter_b is not None else attn_nw),
                 attn_nw,
                 (attn_nb if attn_nb is not None else attn_nw),
                 config.epsilon,
                 q_scales[2],
                 (q_groups * (2**merge_count)),
                 config.pre_layer_norm)
            output = inference_cuda_module.vector_matmul_int8(intermediate,
                                                              output_w,
                                                              q_scales[3],
                                                              q_groups,
                                                              (merge_count))
        else:
            mlp_gemm_func = inference_cuda_module.mlp_gemm_fp16 if config.fp16 else \
                                    inference_cuda_module.mlp_gemm_fp32
            vector_matmul_func = inference_cuda_module.vector_matmul_fp16 if config.fp16 else \
                                    inference_cuda_module.vector_matmul_fp32
            (intermediate,
             residual_add) = mlp_gemm_func(input,
                                           residual,
                                           (bias if bias is not None else attn_nw),
                                           inter_w,
                                           (inter_b if inter_b is not None else attn_nw),
                                           attn_nw,
                                           (attn_nb if attn_nb is not None else attn_nw),
                                           config.epsilon,
                                           config.pre_layer_norm,
                                           (bias is not None),
                                           (inter_b is not None),
                                           (attn_nb is not None))
            if inter_w1 is not None:
                intermediate = vector_matmul_func(input, inter_w1) * intermediate
            output = vector_matmul_func(intermediate, output_w)

        if mp_group is not None and torch.distributed.get_world_size(group=mp_group) > 1:
            torch.distributed.all_reduce(output, group=mp_group)

        bias_residual_func = inference_cuda_module.bias_residual_fp16 if config.fp16 or config.q_int8 else \
                                    inference_cuda_module.bias_residual_fp32

        output = bias_residual_func(output,
                                    residual_add,
                                    output_b if output_b is not None else residual_add,
                                    output_b is not None)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('You are running with DeepSpeed Inference mode. \
                            Please switch to Training mode for running backward!')


class DeepSpeedCrossMLP(nn.Module):
    def __init__(self,
                 config,
                 mp_group=None,
                 q_scales=None,
                 q_groups=1,
                 merge_count=1,
                 mlp_extra_grouping=False):
        super(DeepSpeedCrossMLP, self).__init__()

        self.config = config
        self.attn_nw = nn.Parameter(torch.Tensor(self.config.hidden_size))
        self.attn_nb = nn.Parameter(torch.Tensor(self.config.hidden_size))
        self.inter_w = nn.Parameter(
            torch.Tensor(self.config.hidden_size,
                         self.config.intermediate_size // self.config.mp_size))
        self.inter_w1 = nn.Parameter(
            torch.Tensor(self.config.hidden_size,
                         self.config.intermediate_size // self.config.mp_size))

        self.inter_b = nn.Parameter(
            torch.Tensor(self.config.intermediate_size // self.config.mp_size))
        self.output_w = nn.Parameter(
            torch.Tensor((self.config.intermediate_size // self.config.mp_size),
                         self.config.hidden_size))
        self.output_b = nn.Parameter(torch.Tensor(self.config.hidden_size))

        # used for quantization
        self.q_scales = q_scales
        self.q_groups = q_groups * 2 if mlp_extra_grouping else q_groups
        self.merge_count = int(math.log2(merge_count))

        self.mp_group = mp_group

    def forward(self, input, residual, bias):
        return DeepSpeedCrossMLPFunction.apply(input,
                                               residual,
                                               bias,
                                               self.inter_w,
                                               self.inter_b,
                                               self.attn_nw,
                                               self.attn_nb,
                                               self.config,
                                               self.mp_group,
                                               self.output_b,
                                               self.output_w,
                                               self.q_scales,
                                               self.q_groups,
                                               self.merge_count,
                                               self.inter_w1)


class DeepSpeedEncoderDecoder(nn.Module):
    """Initialize the DeepSpeed Transformer Layer.

        Arguments:
            layer_id: The layer index starting from 0, e.g. if model has 24 transformer layers,
                layer_id will be 0,1,2...23 when each layer object is instantiated

            config: An object of DeepSpeedInferenceConfig
            mp_group: Model parallelism group initialized on the modeling side.
            quantize_scales: This argument groups all the layers' scales used for quantization
            quantize_groups: Number of groups used for quantizing the model
            merge_count: Shows the number of model-parallel checkpoints merged before running inference.
                We use this argument to control the quantization scale for the model parameters if a bigger
                quantize-grouping than 1 is used.
            mlp_extra_grouping: This flag is used to show a 2x higher number of groups used for the MLP part
                of a Transformer layer. We use this feature for quantization to reduce the convergence impact
                for specific downstream tasks.
    """
    layer_id = 0

    def __init__(self,
                 config,
                 mp_group=None,
                 quantize_scales=None,
                 quantize_groups=1,
                 merge_count=1,
                 mlp_extra_grouping=False,
                 qkv_merging=False):
        super(DeepSpeedEncoderDecoder, self).__init__()

        self.config = config
        self.is_decoder = True
        self.config.layer_id = DeepSpeedEncoderDecoder.layer_id
        DeepSpeedEncoderDecoder.layer_id += 1
        self.self_attention = DeepSpeedSelfCrossAttention(self.config,
                                                          mp_group,
                                                          quantize_scales,
                                                          quantize_groups,
                                                          merge_count,
                                                          qkv_merging)
        self.cross_attention = DeepSpeedSelfCrossAttention(self.config,
                                                           mp_group,
                                                           quantize_scales,
                                                           quantize_groups,
                                                           merge_count,
                                                           qkv_merging)
        self.mlp = DeepSpeedCrossMLP(self.config,
                                     mp_group,
                                     quantize_scales,
                                     quantize_groups,
                                     merge_count,
                                     mlp_extra_grouping)

        self.norm_w = nn.Parameter(torch.Tensor(self.config.hidden_size))
        self.norm_b = nn.Parameter(torch.Tensor(self.config.hidden_size))

        self.cross_norm_w = nn.Parameter(torch.Tensor(self.config.hidden_size))

        global inference_cuda_module
        global specialized_mode
        if inference_cuda_module is None:
            specialized_mode = False
            if hasattr(op_builder, 'InferenceSpecializedBuilder'):
                builder = op_builder.InferenceSpecializedBuilder()
                if builder.is_compatible():
                    inference_cuda_module = builder.load()
                    specialized_mode = True
                else:
                    inference_cuda_module = op_builder.InferenceBuilder().load()
            else:
                inference_cuda_module = op_builder.InferenceBuilder().load()
        self.config.specialized_mode = specialized_mode
        print("DeepSpeed Transformer Inference config is ", self.config.__dict__)

    def forward(self,
                input,
                input_mask=None,
                attention_mask=None,
                head_mask=None,
                position_bias=None,
                layer_past=None,
                get_key_value=False,
                get_present=False,
                encoder_output=None,
                enc_dec_attn_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                encoder_decoder_position_bias=None,
                layer_head_mask=None,
                cross_attn_layer_head_mask=None,
                past_key_value=None,
                use_cache=False,
                output_attentions=False,
                encoder_layer_head_mask=None):
        #self.config.triangular_masking = False
        #import pdb;pdb.set_trace()
        get_present = (get_present or get_key_value or use_cache)
        input_mask = input_mask if attention_mask is None else attention_mask

        input_type = input.dtype

        if (self.config.fp16 or self.config.q_int8) \
            and input.dtype == torch.float:
            input = input.half()

        with torch.no_grad():
            past_key_value = (layer_past if past_key_value is None else past_key_value)
            if past_key_value is not None:
                self_attn_past_key_value = past_key_value[:2]
                cross_attn_past_key_value = past_key_value[2:]
            else:
                self_attn_past_key_value, cross_attn_past_key_value = None, None
            attention_outputs = self.self_attention(input,
                                                    input_mask,
                                                    head_mask,
                                                    self_attn_past_key_value,
                                                    get_present,
                                                    encoder_hidden_states,
                                                    encoder_attention_mask,
                                                    output_attentions,
                                                    self.norm_w,
                                                    self.norm_b,
                                                    position_bias)

            attention_output, p_key, p_value = attention_outputs[0:3]
            self_position_attention = attention_outputs[3:]
            presents = (p_key, p_value)

            attention_output = attention_output + input
            if self.self_attention.attn_ob is not None:
                attention_output = attention_output + self.self_attention.attn_ob

            crose_attention_outputs = self.cross_attention(
                attention_output,
                encoder_attention_mask,
                head_mask,
                cross_attn_past_key_value,
                get_present,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
                self.norm_w,
                self.norm_b,
                encoder_decoder_position_bias)

            crose_attention_output, p_key, p_value = crose_attention_outputs[0:3]
            cross_position_attention = crose_attention_outputs[3:]
            presents += (p_key, p_value)

            output = self.mlp(crose_attention_output,
                              attention_output,
                              self.cross_attention.attn_ob)

            if not self.config.pre_layer_norm:
                ds_layernorm = inference_cuda_module.layer_norm_fp16 if self.config.fp16 or self.config.q_int8 else \
                                        inference_cuda_module.layer_norm_fp32
                output = ds_layernorm(output,
                                      self.norm_w,
                                      self.norm_b,
                                      self.config.epsilon)

            if input_type != output.dtype:
                output = output.to(input_type)

        output = (output, presents)
        output = output + self_position_attention + cross_position_attention
        if self.config.return_tuple:
            return output
        else:
            return output[0]