# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math
import torch
import torch.nn as nn
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from .op_binding import LinearOp, VectorMatMulOp, SoftmaxContextOp, QKVGemmOp, SoftmaxOp

minus_inf = -10000.0


class DeepSpeedSelfAttention(nn.Module):
    num_layers = 0
    _qkv_buffers = []

    def __init__(self, config, mp_group=None, q_scales=None, q_groups=1, merge_count=1):
        super(DeepSpeedSelfAttention, self).__init__()
        self.config = config
        #data_type = self.config.dtype
        data_type_fp = torch.half if self.config.fp16 else torch.float #== torch.int8 else self.config.dtype
        data_type = data_type_fp
        self.config.layer_id = DeepSpeedSelfAttention.num_layers
        DeepSpeedSelfAttention.num_layers = DeepSpeedSelfAttention.num_layers + 1
        device = get_accelerator().current_device_name()  #if config.bigscience_bloom else 'cpu'

        self.self_attn_layer_norm = nn.LayerNorm(self.config.hidden_size,
                                                 elementwise_affine=True,
                                                 dtype=data_type,
                                                 device=device)
        self.k_proj = nn.Linear(self.config.hidden_size,
                                self.config.hidden_size,
                                bias=True,
                                device=device,
                                dtype=data_type)
        self.v_proj = nn.Linear(self.config.hidden_size,
                                self.config.hidden_size,
                                bias=True,
                                device=device,
                                dtype=data_type)
        self.q_proj = nn.Linear(self.config.hidden_size,
                                self.config.hidden_size,
                                bias=True,
                                device=device,
                                dtype=data_type)
        self.out_proj = nn.Linear(self.config.hidden_size,
                                  self.config.hidden_size,
                                  bias=True,
                                  device=device,
                                  dtype=data_type)

        if self.config.set_empty_params:
            self.attn_qw = None
            self.attn_qb = None
            self.attn_kw = None
            self.attn_kb = None
            self.attn_vw = None
            self.attn_vb = None
            self.attn_qkvw = None
            self.attn_qkvb = None
            self.attn_ow = None
            self.attn_ob = None
        else:
            qkv_size_per_partition = (self.config.hidden_size // self.config.mp_size) * 3
            self.attn_qkvw = nn.Parameter(torch.empty(self.config.hidden_size,
                                                      qkv_size_per_partition,
                                                      dtype=data_type,
                                                      device=device),
                                          requires_grad=False)
            self.attn_qkvb = nn.Parameter(torch.empty(qkv_size_per_partition, dtype=data_type_fp, device=device),
                                          requires_grad=False)
            out_size_per_partition = self.config.hidden_size // self.config.mp_size
            self.attn_ow = nn.Parameter(torch.empty(out_size_per_partition,
                                                    self.config.hidden_size,
                                                    dtype=data_type,
                                                    device=device),
                                        requires_grad=False)

            self.attn_ob = nn.Parameter(torch.empty(self.config.hidden_size, dtype=data_type_fp, device=device),
                                        requires_grad=False)

        self.num_attention_heads_per_partition = self.config.heads // self.config.mp_size
        self.hidden_size_per_partition = self.config.hidden_size // self.config.mp_size
        self.hidden_size_per_attention_head = self.config.hidden_size // self.config.heads

        self.mp_group = mp_group

        # used for quantization
        self.q_scales = q_scales
        self.q_groups = q_groups
        self.merge_count = int(math.log2(merge_count))

        self.norm_factor = math.sqrt(self.config.hidden_size // self.config.heads)
        if not config.use_mup:
            self.norm_factor = math.sqrt(self.norm_factor)

        if self.config.scale_attn_by_inverse_layer_idx is True:
            self.norm_factor *= math.sqrt(self.config.layer_id + 1)
            # https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/models/gpt2/modeling_gpt2.py#L191

        self.qkv_func = QKVGemmOp(config)
        self.score_context_func = SoftmaxContextOp(config)
        self.linear_func = LinearOp(config)
        self.vector_matmul_func = VectorMatMulOp(config)
        if len(DeepSpeedSelfAttention._qkv_buffers) == 0:
            DeepSpeedSelfAttention._qkv_buffers = [
                torch.empty(self.hidden_size_per_partition * 3,
                            self.config.hidden_size,
                            dtype=data_type_fp,
                            device=device),
                torch.empty(self.hidden_size_per_partition * 3, dtype=data_type_fp, device=device)
            ]

    def compute_attention(self, qkv_out, input_mask, layer_past, alibi):
        if isinstance(qkv_out, list) or isinstance(qkv_out, tuple):
            qkv_out = qkv_out[0]

        no_masking = input_mask is None

        if no_masking:
            input_mask = torch.empty(1)

        attn_key_value = self.score_context_func(
            query_key_value=qkv_out,
            attn_mask=((1 - input_mask).to(qkv_out.dtype) *
                       minus_inf) if input_mask.dtype == torch.int64 else input_mask,
            heads=self.num_attention_heads_per_partition,
            norm_factor=(1 / self.norm_factor if self.config.scale_attention else 1.0),
            no_masking=no_masking,
            layer_id=self.config.layer_id,
            num_layers=DeepSpeedSelfAttention.num_layers,
            alibi=alibi)

        context_layer, key_layer, value_layer = attn_key_value
        return context_layer, key_layer, value_layer

    def _merge_qkv(self):
        qvkw = DeepSpeedSelfAttention._qkv_buffers[0]
        qvkw[:self.hidden_size_per_partition, :] = self.attn_qw  # type: ignore
        qvkw[self.hidden_size_per_partition:2 * self.hidden_size_per_partition, :] = self.attn_kw  # type: ignore
        qvkw[2 * self.hidden_size_per_partition:, :] = self.attn_vw  # type: ignore
        if self.attn_qb is not None:
            qvkb = DeepSpeedSelfAttention._qkv_buffers[1]
            qvkb[:self.hidden_size_per_partition] = self.attn_qb
            qvkb[self.hidden_size_per_partition:2 * self.hidden_size_per_partition] = self.attn_kb  # type: ignore
            qvkb[2 * self.hidden_size_per_partition:] = self.attn_vb  # type: ignore
        return DeepSpeedSelfAttention._qkv_buffers

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        head_dim = self.hidden_size_per_partition // self.num_attention_heads_per_partition
        return tensor.view(bsz, seq_len, self.num_attention_heads_per_partition, head_dim).transpose(1, 2).contiguous()


    def attn_baseline(
        self,
        input,
        input_mask,
        head_mask=None,
        layer_past=None,
        output_attentions=False,
        norm_w=None,
        norm_b=None
    ):
        if self.config.transposed_mode:
            qkvw = self._attn_qkvw.split(self.config.hidden_size, 0)
            self.k_proj.weight.data.copy_(qkvw[1])
            self.v_proj.weight.data.copy_(qkvw[2])
            self.out_proj.weight.data.copy_(self.attn_ow)
        else:
            qkvw = self._attn_qkvw.split(self.config.hidden_size, 1)
            self.k_proj.weight.data.copy_(qkvw[1].transpose(0, 1))
            self.v_proj.weight.data.copy_(qkvw[2].transpose(0, 1))
            self.out_proj.weight.data.copy_(self.attn_ow.transpose(0, 1))
        qkvb = self._attn_qkvb.split(self.config.hidden_size, 0)
        self.k_proj.bias.data.copy_(qkvb[1])
        self.v_proj.bias.data.copy_(qkvb[2])
        self.out_proj.bias.data.copy_(self.attn_ob)

        self.self_attn_layer_norm.weight.data.copy_(norm_w)
        self.self_attn_layer_norm.bias.data.copy_(norm_b)

        hidden_states = self.self_attn_layer_norm(input)
        bsz, tgt_len, _ = hidden_states.size()

        if layer_past is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([layer_past[0], key_states], dim=2)
            value_states = torch.cat([layer_past[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        past_key_value = (key_states, value_states)
        key_layer = key_states
        value_layer = value_states

        head_dim = self.hidden_size_per_partition // self.num_attention_heads_per_partition
        scaling = head_dim**-0.5
        if self.config.transposed_mode:
            self.q_proj.weight.data.copy_(qkvw[0])
        else:
            self.q_proj.weight.data.copy_(qkvw[0].transpose(0, 1))

        self.q_proj.bias.data.copy_(qkvb[0])
        query_states = self.q_proj(hidden_states) * scaling

        proj_shape = (bsz * self.num_attention_heads_per_partition, -1, head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        src_len = key_states.size(1)

        if attn_weights.size() != (bsz * self.num_attention_heads_per_partition, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_attention_heads_per_partition, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}")

        if input_mask is not None:
            if input_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {input_mask.size()}")
            attn_weights = attn_weights.view(bsz, self.num_attention_heads_per_partition, tgt_len,
                                             src_len) + input_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = attn_weights.view(bsz * self.num_attention_heads_per_partition, tgt_len, src_len)

        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if head_mask is not None:
            if head_mask.size() != (self.num_attention_heads_per_partition, ):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_attention_heads_per_partition,)}, but is"
                    f" {head_mask.size()}")
            attn_weights = head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_attention_heads_per_partition,
                                                                           tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_attention_heads_per_partition, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_attention_heads_per_partition, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_attention_heads_per_partition, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        value_states = value_states.view(*proj_shape)
        attn_output = torch.bmm(attn_weights, value_states)

        context_layer = attn_output

        attn_output = attn_output.view(bsz, self.num_attention_heads_per_partition, tgt_len, head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.config.hidden_size)
        attn_output = self.out_proj(attn_output)

        output = attn_output
        inp_norm = hidden_states.norm()

        return (output, key_layer, value_layer, context_layer, inp_norm)

    # Copy of attn_baseline for debugging
    def debug_attn_baseline(
        self,
        input,
        input_mask,
        head_mask=None,
        layer_past=None,
        # get_present=False,
        # encoder_hidden_states=None,
        # encoder_attention_mask=None,
        output_attentions=False,
        norm_w=None,
        norm_b=None,
        # alibi=None
    ):
        debug = True
        print_tensors = True
        #print(len(self._attn_qkvw), len(self._attn_qkvb))

        if self.config.transposed_mode:
            qkvw = self._attn_qkvw.split(self.config.hidden_size, 0)
            self.k_proj.weight.data.copy_(qkvw[1])
            self.v_proj.weight.data.copy_(qkvw[2])
            self.out_proj.weight.data.copy_(self.attn_ow)
        else:
            qkvw = self._attn_qkvw.split(self.config.hidden_size, 1)
            self.k_proj.weight.data.copy_(qkvw[1].transpose(0, 1))
            self.v_proj.weight.data.copy_(qkvw[2].transpose(0, 1))
            self.out_proj.weight.data.copy_(self.attn_ow.transpose(0, 1))
        qkvb = self._attn_qkvb.split(self.config.hidden_size, 0)
        self.k_proj.bias.data.copy_(qkvb[1])
        self.v_proj.bias.data.copy_(qkvb[2])
        self.out_proj.bias.data.copy_(self.attn_ob)

        self.self_attn_layer_norm.weight.data.copy_(norm_w)
        self.self_attn_layer_norm.bias.data.copy_(norm_b)

        if debug: print(f"ds attn: b4 ln hidden_states norm = {input.norm()}")
        if print_tensors: print(f"ds attn: b4 ln hidden_states = {input}")
        hidden_states = self.self_attn_layer_norm(input)
        bsz, tgt_len, _ = hidden_states.size()

        if layer_past is not None:
            if debug:
                print(f"ds attn: layer_past key (norm, size)  = {layer_past[0].norm()}, {layer_past[0].size()}")
            if print_tensors:
                print(f"ds attn: layer_past key   = {layer_past[0]}")
            if debug: print(f"ds attn: layer_past value norm = {layer_past[1].norm()}")
            if print_tensors: print(f"ds attn: layer_past value  = {layer_past[1]}")
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([layer_past[0], key_states], dim=2)
            value_states = torch.cat([layer_past[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        past_key_value = (key_states, value_states)
        key_layer = key_states
        value_layer = value_states
        if debug: print(f"ds attn: key_states (norm, size) = {key_states.norm()}, {key_states.size()}")
        if print_tensors: print(f"ds attn: key_states   = {key_states}")
        if debug: print(f"ds attn: value_states norm = {value_states.norm()}")
        if print_tensors: print(f"ds attn: value_states  = {value_states}")
        if debug: print(f"ds attn: hidden_states norm = {hidden_states.norm()}")
        if print_tensors: print(f"ds attn: hidden_states = {hidden_states}")

        #same as qkv_func return
        #return (hidden_states, hidden_states.norm())

        head_dim = self.hidden_size_per_partition // self.num_attention_heads_per_partition
        scaling = head_dim**-0.5
        if self.config.transposed_mode:
            self.q_proj.weight.data.copy_(qkvw[0])
        else:
            self.q_proj.weight.data.copy_(qkvw[0].transpose(0, 1))

        self.q_proj.bias.data.copy_(qkvb[0])
        query_states = self.q_proj(hidden_states) * scaling

        proj_shape = (bsz * self.num_attention_heads_per_partition, -1, head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        if debug: print(f"ds attn: a4 view key_states (norm, size) = {key_states.norm()}, {key_states.size()}")
        if print_tensors: print(f"ds attn: a4 view key_states = {key_states}")
        if debug: print(f"ds attn: query_states norm = {query_states.norm()}")
        if print_tensors: print(f"ds attn: query_states = {query_states}")
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if debug: print(f"ds attn: a4 bmm attn_weights norm = {attn_weights.norm()}")
        if print_tensors: print(f"ds attn: a4 bmm attn_weights = {attn_weights}")

        src_len = key_states.size(1)
        if debug: print(f"ds attn: src_len = {src_len}")

        if attn_weights.size() != (bsz * self.num_attention_heads_per_partition, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_attention_heads_per_partition, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}")

        if input_mask is not None:
            if print_tensors: print(f"ds attn: input_mask = {input_mask}")
            if input_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {input_mask.size()}")
            attn_weights = attn_weights.view(bsz, self.num_attention_heads_per_partition, tgt_len,
                                             src_len) + input_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = attn_weights.view(bsz * self.num_attention_heads_per_partition, tgt_len, src_len)
        if debug: print(f"ds attn: a4 attn_weights size = {attn_weights.size()}")
        if print_tensors: print(f"ds attn: a4 attn_weights = {attn_weights}")

        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if debug: print(f"ds attn: a4 softmax attn_weights = {attn_weights.norm()}")

        if head_mask is not None:
            if head_mask.size() != (self.num_attention_heads_per_partition, ):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_attention_heads_per_partition,)}, but is"
                    f" {head_mask.size()}")
            attn_weights = head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_attention_heads_per_partition,
                                                                           tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_attention_heads_per_partition, tgt_len, src_len)
            if debug: print(f"ds attn: a4 head_mask attn_weights norm = {attn_weights.norm()}")

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_attention_heads_per_partition, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_attention_heads_per_partition, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        if debug: print(f"ds attn: a4 output_attentions attn_weights norm = {attn_weights.norm()}")

        value_states = value_states.view(*proj_shape)
        attn_output = torch.bmm(attn_weights, value_states)
        if debug: print(f"ds attn: a4 bmm attn_output norm = {attn_output.norm()}")
        context_layer = attn_output

        # same as compute_attention return
        #return context_layer, key_layer, value_layer

        attn_output = attn_output.view(bsz, self.num_attention_heads_per_partition, tgt_len, head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.config.hidden_size)
        attn_output = self.out_proj(attn_output)
        if debug: print(f"inside ds attn: key_states (norm, size)  = {key_layer.norm()}, {key_states.size()}")
        if print_tensors: print(f"inside ds attn: key_states   = {key_layer}")
        if debug: print(f"inside ds attn: value_states norm  = {value_layer.norm()}")
        if print_tensors: print(f"inside ds attn: value_states  = {value_layer}")
        if debug: print(f"ds attn: return attn_output norm = {attn_output.norm()}")
        if print_tensors: print(f"ds attn: return attn_output = {attn_output}")

        output = attn_output
        inp_norm = hidden_states.norm()

        return (output, key_layer, value_layer, context_layer, inp_norm)

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
                alibi=None,
                attn_base=False,
                attn_debug=False):
        if self.attn_qkvw is None:
            self._attn_qkvw, self._attn_qkvb = self._merge_qkv()
        else:
            self._attn_qkvw = self.attn_qkvw
            self._attn_qkvb = self.attn_qkvb

        # only set attn_base=True from the caller. This ensures that only the model that is supported (like opt) calls this
        if attn_base and not attn_debug:
            output, key_layer, value_layer, context_layer, inp_norm = self.attn_baseline(
                input, input_mask, head_mask, layer_past, output_attentions, norm_w, norm_b)
        elif attn_base and attn_debug:
            output, key_layer, value_layer, context_layer, inp_norm = self.debug_attn_baseline(
                input, input_mask, head_mask, layer_past, output_attentions, norm_w, norm_b)
        else:
            debug = attn_debug

            if debug: print(f"inside ds attn: b4 ln weight = {self._attn_qkvw.norm()}")
            if debug: print(f"inside ds attn: b4 ln bias   = {self._attn_qkvb.norm()}")
            if debug: print(f"inside ds attn: b4 ln input  = {input.norm()}")
            #if debug: print(f"inside ds attn: b4 ln input tensor = {input}")
            if debug: print(f"inside ds attn: b4 qkv_func gamma = {norm_w.norm()}")
            if debug: print(f"inside ds attn: b4 qkv_func beta   = {norm_b.norm()}")

            if not self.config.pre_layer_norm:
                qkv_out = self.linear_func(input=input,
                                           weight=self._attn_qkvw,
                                           bias=self._attn_qkvb,
                                           add_bias=self.attn_qkvb is not None,
                                           do_flash_attn=False,
                                           num_heads=self.num_attention_heads_per_partition,
                                           num_layers=DeepSpeedSelfAttention.num_layers)
            else:
                qkv_out = self.qkv_func(input=input,
                                        weight=self._attn_qkvw,
                                        bias=self._attn_qkvb,
                                        gamma=norm_w,
                                        beta=norm_b)

            if debug: print(f"inside ds attn: qkv_out[0] = {qkv_out[0].norm()}")
            if debug: print(f"inside ds attn: qkv_out[1] = {qkv_out[1].norm()}")
            if debug: print(f"inside ds attn: input_mask   = {input_mask.norm()}")
            if debug and layer_past:
                print(f"inside ds attn: layer_past[0]  = {layer_past[0].norm()}")
                print(f"inside ds attn: layer_past[1]  = {layer_past[1].norm()}")
            if debug and alibi: print(f"inside ds attn: alibi  = {alibi.norm()}")

            context_layer, key_layer, value_layer = self.compute_attention(qkv_out=qkv_out,
                                                                           input_mask=input_mask,
                                                                           layer_past=layer_past,
                                                                           alibi=alibi)

            if debug: print(f"inside ds attn: a4 compute attn context_layer   = {context_layer.norm()}")
            if debug: print(f"inside ds attn: a4 compute attn key_layer  = {key_layer.norm()}")
            if debug: print(f"inside ds attn: a4 compute attn value_layer  = {value_layer.norm()}")

            output = self.vector_matmul_func(input=context_layer, weight=self.attn_ow)
            inp_norm = qkv_out[-1]

            if debug: print(f"inside ds attn: a4 matmul output  = {output.norm()}")
            if debug: print(f"inside ds attn: a4 matmul inp_norm  = {inp_norm.norm()}")

            if self.config.mlp_after_attn and self.mp_group is not None and dist.get_world_size(
                    group=self.mp_group) > 1:
                dist.all_reduce(output, group=self.mp_group)

            if debug: print(f"inside ds attn: return output = {output.norm()}")
            if debug: print(f"inside ds attn: return key_layer   = {key_layer.norm()}")
            if debug: print(f"inside ds attn: return value_layer  = {value_layer.norm()}")
            if debug: print(f"inside ds attn: return context_layer  = {context_layer.norm()}")
            if debug: print(f"inside ds attn: return inp_norm  = {inp_norm.norm()}")

        return (output, key_layer, value_layer, context_layer, inp_norm)


class BloomSelfAttention(DeepSpeedSelfAttention):

    def __init__(self, *args, **kwargs):
        super(BloomSelfAttention, self).__init__(*args, **kwargs)
        self.softmax_func = SoftmaxOp(self.config)

    ########### This part is taken/modified form the HF modeling_bloom.py ################
    # Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bloom/modeling_bloom.py

    def _transpose_for_context(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_layer_shape = x.size()[:-2] + \
                                    (self.hidden_size_per_partition,)
        return x.view(*new_x_layer_shape).contiguous()

    def _split_tensor_along_last_dim(self, tensor, num_partitions, contiguous_split_chunks=True):
        """Split a tensor along its last dimension.

        Args:
            tensor: ([`torch.tensor`], *required*):
                input tensor to split
            num_partitions ([`int`], *required*):
                number of partitions to split the tensor
            contiguous_split_chunks ([`bool`], *optional*, default=`False`)::
                If True, make each chunk contiguous in memory.
        """
        # Get the size and dimension.
        last_dim = tensor.dim() - 1
        numerator, denominator = tensor.size()[last_dim], num_partitions
        if not (numerator % denominator == 0):
            raise ValueError(f"{numerator} is not divisible by {denominator}")
        last_dim_size = numerator // denominator
        # Split.
        tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
        # Note: torch.split does not create contiguous tensors by default.
        if contiguous_split_chunks:
            return tuple(chunk.contiguous() for chunk in tensor_list)
        return tensor_list

    def compute_attention(self, qkv_out, input_mask, layer_past, alibi):
        if isinstance(qkv_out, list) or isinstance(qkv_out, tuple):
            qkv_out = qkv_out[0]

        no_masking = input_mask is None

        if no_masking:
            input_mask = torch.empty(1)

        mixed_x_layer = qkv_out
        alibi = alibi.to(get_accelerator().current_device_name())
        head_dim = self.hidden_size_per_partition // self.num_attention_heads_per_partition
        new_tensor_shape = mixed_x_layer.size()[:-1] + (self.num_attention_heads_per_partition, 3 * head_dim)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        query_layer, key_layer, value_layer = self._split_tensor_along_last_dim(mixed_x_layer, 3)

        # [batch_size, head_dim, q_length, k_length]
        output_size = (query_layer.size(0), query_layer.size(2), query_layer.size(1), key_layer.size(1))
        # [batch_size, q_length, num_heads, head_dim] -> [q_length, batch_size * num_heads, head_dim]
        query_layer = query_layer.transpose(1, 2).reshape(output_size[0] * output_size[1], output_size[2], -1)
        # [batch_size, k_length, num_heads, head_dim] -> [k_length, batch_size * num_heads, head_dim]
        key_layer = key_layer.transpose(1, 2).reshape(output_size[0] * output_size[1], output_size[3],
                                                      -1).transpose(-1, -2)
        value_layer = value_layer.transpose(1, 2).reshape(output_size[0] * output_size[1], output_size[3], -1)
        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension -> [batch_size, qk_length, num_heads, head_dim]
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=-1)
            value_layer = torch.cat((past_value.type_as(value_layer), value_layer), dim=-2)

        presents = (key_layer, value_layer)
        # Raw attention scores. [batch_size * num_heads, q_length, k_length]
        matmul_result = torch.matmul(query_layer, key_layer)
        # change view to [batch_size, num_heads, q_length, k_length]
        attention_scores = matmul_result.view(output_size[0], output_size[1], output_size[2], -1)

        offset = dist.get_rank() * self.num_attention_heads_per_partition if dist.is_initialized() else 0
        target_dtype = torch.float16 if self.config.dtype == torch.int8 else self.config.dtype
        attention_probs = self.softmax_func(attn_scores=attention_scores,
                                            attn_mask=((1 - input_mask).to(target_dtype) * minus_inf),
                                            alibi=alibi,
                                            triangular=(self.config.triangular_masking
                                                        and (attention_scores.shape[-2] > 1)),
                                            recompute=False,
                                            local_attention=False,
                                            window_size=1,
                                            async_op=False,
                                            layer_scale=1 / (self.norm_factor * self.norm_factor),
                                            head_offset=offset)

        # change view [batch_size x num_heads, q_length, k_length]
        attention_probs_reshaped = attention_probs.view(*matmul_result.shape)

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = torch.bmm(attention_probs_reshaped, value_layer)

        # change view [batch_size, num_heads, q_length, head_dim]
        context_layer = context_layer.view(
            context_layer.size(0) // self.num_attention_heads_per_partition, self.num_attention_heads_per_partition,
            context_layer.size(1), context_layer.shape[-1])

        context_layer = self._transpose_for_context(context_layer)
        key_layer = presents[0]
        value_layer = presents[1]

        return context_layer, key_layer, value_layer

    ###################### End of HF modeling_bloom addition ########################