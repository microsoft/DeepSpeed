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

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, dtype, device, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states
    
class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )
class DeepSpeedSelfAttention(nn.Module):
    num_layers = 0
    _qkv_buffers = []

    def __init__(self, config, mp_group=None, q_scales=None, q_groups=1, merge_count=1):
        super(DeepSpeedSelfAttention, self).__init__()
        self.config = config
        #data_type = self.config.dtype
        #data_type_fp = torch.half if self.config.dtype == torch.int8 else self.config.dtype
        data_type = torch.int8 if config.q_int8 else torch.half if config.fp16 else torch.float
        data_type_fp = torch.half if config.fp16 else torch.float
        self.config.layer_id = DeepSpeedSelfAttention.num_layers
        DeepSpeedSelfAttention.num_layers = DeepSpeedSelfAttention.num_layers + 1
        device = get_accelerator().current_device_name()  #if config.bigscience_bloom else 'cpu'

        self.head_dim = self.config.hidden_size // self.config.heads
        self.input_layernorm = LlamaRMSNorm(self.config.hidden_size, dtype=data_type, device=device)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim)

        self.self_attn_layer_norm = nn.LayerNorm(self.config.hidden_size,
                                                 elementwise_affine=True,
                                                 dtype=data_type,
                                                 device=device)
        self.k_proj = nn.Linear(self.config.hidden_size,
                                self.config.hidden_size,
                                bias=False,
                                device=device,
                                dtype=data_type)
        self.v_proj = nn.Linear(self.config.hidden_size,
                                self.config.hidden_size,
                                bias=False,
                                device=device,
                                dtype=data_type)
        self.q_proj = nn.Linear(self.config.hidden_size,
                                self.config.hidden_size,
                                bias=False,
                                device=device,
                                dtype=data_type)
        self.o_proj = nn.Linear(self.config.hidden_size,
                                  self.config.hidden_size,
                                  bias=False,
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

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids):
        # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def attn_baseline(
        self,
        input,
        input_mask,
        head_mask=None,
        layer_past=None,
        get_present=False,
        output_attentions=False,
        norm_w=None,
        norm_b=None,
        position_ids=None
    ):
        if self.config.transposed_mode:
            qkvw = self._attn_qkvw.split(self.config.hidden_size, 0)
            self.q_proj.weight.data.copy_(qkvw[0])
            self.k_proj.weight.data.copy_(qkvw[1])
            self.v_proj.weight.data.copy_(qkvw[2])
            self.o_proj.weight.data.copy_(self.attn_ow)
        else:
            qkvw = self._attn_qkvw.split(self.config.hidden_size, 1)
            self.q_proj.weight.data.copy_(qkvw[0].transpose(0, 1))
            self.k_proj.weight.data.copy_(qkvw[1].transpose(0, 1))
            self.v_proj.weight.data.copy_(qkvw[2].transpose(0, 1))
            self.o_proj.weight.data.copy_(self.attn_ow.transpose(0, 1))

        self.input_layernorm.weight.data.copy_(norm_w)

        hidden_states = self.input_layernorm(input)
        bsz, q_len, _ = hidden_states.size()

        head_dim = self.hidden_size_per_partition // self.num_attention_heads_per_partition
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_attention_heads_per_partition, head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_attention_heads_per_partition, head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_attention_heads_per_partition, head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if layer_past is not None:
            kv_seq_len += layer_past[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        seq_length = q_len # todo: double check if actually equivalent

        past_key_values_length = 0
        seq_length_with_past = seq_length
        if layer_past is not None:
            past_key_values_length = layer_past[0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=input.device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if layer_past is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([layer_past[0], key_states], dim=2)
            value_states = torch.cat([layer_past[1], value_states], dim=2)

        layer_past = (key_states, value_states) if get_present else None
        key_layer = key_states
        value_layer = value_states

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_attention_heads_per_partition, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_attention_heads_per_partition, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if input_mask is not None:
            if input_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {input_mask.size()}"
                )
            attn_weights = attn_weights + input_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        context_layer = attn_output

        if attn_output.size() != (bsz, self.num_attention_heads_per_partition, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_attention_heads_per_partition, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size_per_partition)
        attn_output = self.o_proj(attn_output)
        inp_norm = hidden_states.norm()

        return (attn_output, key_layer, value_layer, context_layer, inp_norm)

    # Copy of attn_baseline for debugging
    def debug_attn_baseline(
        self,
        input,
        input_mask,
        head_mask=None,
        layer_past=None,
        get_present=False,
        output_attentions=False,
        norm_w=None,
        norm_b=None,
        position_ids=None
    ):
        if self.config.transposed_mode:
            qkvw = self._attn_qkvw.split(self.config.hidden_size, 0)
            self.q_proj.weight.data.copy_(qkvw[0])
            self.k_proj.weight.data.copy_(qkvw[1])
            self.v_proj.weight.data.copy_(qkvw[2])
            self.o_proj.weight.data.copy_(self.attn_ow)
        else:
            qkvw = self._attn_qkvw.split(self.config.hidden_size, 1)
            self.q_proj.weight.data.copy_(qkvw[0].transpose(0, 1))
            self.k_proj.weight.data.copy_(qkvw[1].transpose(0, 1))
            self.v_proj.weight.data.copy_(qkvw[2].transpose(0, 1))
            self.o_proj.weight.data.copy_(self.attn_ow.transpose(0, 1))

        self.input_layernorm.weight.data.copy_(norm_w)

        hidden_states = self.input_layernorm(input)
        bsz, q_len, _ = hidden_states.size()

        debug = True
        if debug: print(f"ds attn: hidden_states = {hidden_states}, {hidden_states.norm()}, {hidden_states.size()}")

        head_dim = self.hidden_size_per_partition // self.num_attention_heads_per_partition
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_attention_heads_per_partition, head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_attention_heads_per_partition, head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_attention_heads_per_partition, head_dim).transpose(1, 2)

        if debug: print(f"ds attn: query_states = {query_states}, {query_states.norm()}")
        if debug: print(f"ds attn: key_states = {key_states}, {key_states.norm()}, {key_states.size()}")
        if debug: print(f"ds attn: value_states = {value_states}, {value_states.norm()}")

        kv_seq_len = key_states.shape[-2]
        if layer_past is not None:
            kv_seq_len += layer_past[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        seq_length = q_len # todo: double check if actually equivalent
        if debug: print(f"q_len = {seq_length}")

        past_key_values_length = 0
        seq_length_with_past = seq_length
        if layer_past is not None:
            past_key_values_length = layer_past[0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if debug: print(f"ds attn: b4 position_ids seq_length_with_past = {seq_length_with_past}")
        if debug: print(f"ds attn: b4 position_ids past_key_values_length = {past_key_values_length}")
        if debug: print(f"ds attn: b4 position_ids position_ids = {position_ids}")
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=input.device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if debug: print(f"ds attn: b4 rotary cos = {cos}")
        if debug: print(f"ds attn: b4 rotary sin = {sin}")
        if debug: print(f"ds attn: b4 rotary position_ids = {position_ids}")

        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if debug: print(f"ds attn: rotary query_states = {query_states}, {query_states.norm()}")
        if debug: print(f"ds attn: rotary key_states = {key_states}, {key_states.norm()}")

        if layer_past is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([layer_past[0], key_states], dim=2)
            value_states = torch.cat([layer_past[1], value_states], dim=2)

        layer_past = (key_states, value_states) if get_present else None
        if debug: print(f"ds attn: past_key_value[0] = {layer_past[0]}, {layer_past[0].norm()}")
        if debug: print(f"ds attn: past_key_value[1] = {layer_past[1]}, {layer_past[1].norm()}")

        key_layer = key_states
        value_layer = value_states

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if debug: print(f"ds attn: attn_weights = {attn_weights}, {attn_weights.norm()}")

        if attn_weights.size() != (bsz, self.num_attention_heads_per_partition, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_attention_heads_per_partition, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if input_mask is not None:
            if input_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {input_mask.size()}"
                )
            attn_weights = attn_weights + input_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            if debug: print(f"ds attn: a4 attn mask attn_weights = {attn_weights}, {attn_weights.norm()}")

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        if debug: print(f"ds attn: a4 softmax attn_weights = {attn_weights}, {attn_weights.norm()}")
        if debug: print(f"ds attn: a4 softmax attn_output = {attn_output}, {attn_output.norm()}")

        context_layer = attn_output

        if attn_output.size() != (bsz, self.num_attention_heads_per_partition, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_attention_heads_per_partition, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size_per_partition)

        attn_output = self.o_proj(attn_output)
        if debug: print(f"ds attn: self.o_proj weight: {self.o_proj.weight.data.norm()}")
        if debug: print(f"ds attn: a4 o_proj attn_output = {attn_output}, {attn_output.norm()}")

        inp_norm = hidden_states.norm()
        if debug: print(f"ds attn: inp_norm = {inp_norm}, {inp_norm.norm()}")

        return (attn_output, key_layer, value_layer, context_layer, inp_norm)

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
                input, input_mask, head_mask, layer_past, get_present, output_attentions, norm_w, norm_b)
        elif attn_base and attn_debug:
            output, key_layer, value_layer, context_layer, inp_norm = self.debug_attn_baseline(
                input, input_mask, head_mask, layer_past, get_present, output_attentions, norm_w, norm_b)
        else:
            debug = attn_debug

            if debug: print(f"inside ds attn: b4 ln weight = {self._attn_qkvw.norm()}")
            #if debug: print(f"inside ds attn: b4 ln bias   = {self._attn_qkvb.norm()}")
            if debug: print(f"inside ds attn: b4 ln input  = {input.norm()}")
            #if debug: print(f"inside ds attn: b4 ln input tensor = {input}")
            if debug: print(f"inside ds attn: b4 qkv_func gamma = {norm_w.norm()}")
            #if debug: print(f"inside ds attn: b4 qkv_func beta   = {norm_b.norm()}")

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
                                        bias=(self._attn_qkvb if self._attn_qkvb is not None else norm_b),
                                        gamma=norm_w,
                                        beta=norm_b,
                                        add_bias=(self.attn_qkvb is not None),
                                        num_layers=DeepSpeedSelfAttention.num_layers,
                                        num_heads=self.num_attention_heads_per_partition)

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
        #target_dtype = torch.float16 if self.config.dtype == torch.int8 else self.config.dtype
        attention_probs = self.softmax_func(attn_scores=attention_scores,
                                            attn_mask=((1 - input_mask).half() * minus_inf),
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
