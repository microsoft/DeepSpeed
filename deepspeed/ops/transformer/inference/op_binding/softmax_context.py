# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed import comm as dist
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp

import math

from torch import nn
# rotary pos emb helpers (torch.jit.script does not seem to support staticmethod...)
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class FalconRotaryEmbedding(nn.Module):
    """Implementation of RotaryEmbedding from GPT-NeoX.
    This implementation is designed to operate on queries and keys that are compatible with `[batch_size,
    n_heads_per_partition, seq_len, head_dim]` (e.g. MinGPTAttention format).
    """

    def __init__(self, head_dim: int, base=10000, max_position_embeddings=2048):
        super().__init__()
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (self.base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim
        self.seq_len_cached = -1
        self.cos_cached: torch.Tensor | None = None
        self.sin_cached: torch.Tensor | None = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(device)

        if dtype in [torch.float16, torch.bfloat16]:
            emb = emb.float()

        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]
        #if torch.distributed.get_rank() == 0:
        #    print(emb.shape, self.cos_cached.shape)
        #exit()
        self.cos_cached = self.cos_cached.type(dtype)
        self.sin_cached = self.sin_cached.type(dtype)

    def cos_sin(self, seq_len: int, past_key_values_length: int, device="cpu", dtype=torch.bfloat16) -> torch.Tensor:
        total_length = seq_len + past_key_values_length
        if total_length > self.seq_len_cached:
            self._set_cos_sin_cache(total_length, device, dtype)
        return (
            self.cos_cached[:, :, past_key_values_length : seq_len + past_key_values_length],
            self.sin_cached[:, :, past_key_values_length : seq_len + past_key_values_length],
        )

    def forward(self, query, key, past_key_values_length=0):
        batch, _, seq_len, head_dim = query.shape
        cos, sin = self.cos_sin(seq_len, past_key_values_length, query.device, query.dtype)
        try:
            (query * cos) + (rotate_half(query) * sin)
        except:
            print(f'rotary: {query.shape}, {cos.shape}, {seq_len}')
            exit()
        return (query * cos) + (rotate_half(query) * sin), (key * cos) + (rotate_half(key) * sin)




class SoftmaxContextOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig):
        super(SoftmaxContextOp, self).__init__(config)
        try:
            if self.config.dtype in [torch.float16, torch.int8]:
                self.softmax_context_func = self.inference_module.softmax_context_fp16
            elif self.config.dtype == torch.bfloat16:
                self.softmax_context_func = self.inference_module.softmax_context_bf16
            else:
                self.softmax_context_func = self.inference_module.softmax_context_fp32
        except AttributeError:
            self.softmax_context_func = self.softmax_context_fallback
        self.num_kv_per_partition = self.config.num_kv // self.config.mp_size

        self.rotary_emb = FalconRotaryEmbedding(
                64,
                max_position_embeddings=2048,
            )
    def softmax_context_fallback(self, query_key_value, attn_mask, rotary_dim, rotate_half, roteate_every_two, heads,
                                 norm_factor, triangular_masking, local_attention, window_size, no_masking, layer_id,
                                 num_layers, alibi):
        raise NotImplementedError
    # Copied from transformers.models.bloom.modeling_bloom.BloomAttention._merge_heads
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge heads together over the last dimension

        Args:
            x (`torch.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]

        Returns:
            torch.tensor: [batch_size, seq_length, num_heads * head_dim]
        """
        # What we want to achieve is:
        # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
        batch_size, num_heads, seq_length, _ = x.shape

        # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
        x = x.permute(0, 2, 1, 3)

        # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
        return x.reshape(batch_size, seq_length, -1)


    def forward(self, query_key_value: torch.Tensor, attn_mask: torch.Tensor, heads: int, norm_factor: float,
                no_masking: bool, layer_id: int, num_layers: int, alibi: torch.Tensor, layer_past=None):

        if alibi is not None:
            batch_heads = query_key_value.shape[0] * heads
            offset = dist.get_rank() * batch_heads if dist.is_initialized() else 0
            alibi = alibi[offset:batch_heads + offset, :, :]
        else:
            alibi = torch.empty(1)
        
        context_layer, key_layer, value_layer = self.softmax_context_func(query_key_value, attn_mask, self.config.rotary_dim, self.config.rotate_half,
                                           self.config.rotate_every_two, heads, norm_factor,
                                           self.config.triangular_masking, self.config.local_attention,
                                           self.config.window_size, no_masking, layer_id, num_layers, alibi,
                                           self.config.multi_query, self.num_kv_per_partition)
        #return output
        #print(f'q:{output[0][0,:,0,:]}, k:{output[1][0,:,0,:]}, v:{output[2][0,:,0,:]}')
        batch, seq_len, _ = query_key_value.shape
        qkv = query_key_value.view(batch, seq_len, -1, 64)
        query_layer  = qkv[:, :, :-2]
        key_layer    = qkv[:, :, [-2]]
        value_layer_ = qkv[:, :, [-1]]
        key_layer = torch.broadcast_to(key_layer, query_layer.shape)
        value_layer_ = torch.broadcast_to(value_layer_, query_layer.shape)
        query_layer  = query_layer.transpose(1,2)
        key_layer    = key_layer.transpose(1,2)
        value_layer_ = value_layer_.transpose(1,2)
        past_kv_length = 0 if layer_past is None else layer_past[0].shape[-2]
        query_layer_, key_layer_ = self.rotary_emb(query_layer, key_layer, past_kv_length)
        #if torch.distributed.get_rank() == 0:
        #
        if layer_past is not None:
            past_key, past_value = layer_past
            
            try:
                key_layer_ = torch.cat((past_key, key_layer_), dim=-2)
                value_layer_ = torch.cat((past_value, value_layer_), dim=-2)
            except:
                print(f'layer_past: {key_layer_.shape}, {past_key.shape}')
                exit()
        if True: #layer_past is not None:
            print(f'''[{torch.distributed.get_rank()}]: 
                    key_error: {(((key_layer_-key_layer1).abs()/(key_layer_.abs() + 1e-5)).sum() / key_layer.numel()).item()}
                    value_error: {(((value_layer_-value_layer1).abs()/(value_layer_.abs() + 1e-5)).sum() / value_layer_.numel()).item()}
                    query_error:{(((query_layer_-query_layer1).abs()/(query_layer_.abs() + 1e-5)).sum() / query_layer.numel()).item()}''')
            exit()
        attention_scores = query_layer_ @ key_layer_.transpose(-1, -2)
        
        input_dtype = attention_scores.dtype
        if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
            attention_scores = attention_scores.to(torch.float32)
        attention_scores *= (1 / math.sqrt(64))
        attention_mask_float = (attn_mask * 1.0).masked_fill(attn_mask, float("-1e9")).to(torch.float32)
        try:
            attention_probs = torch.nn.functional.softmax(attention_scores + attention_mask_float, dim=-1, dtype=query_layer_.dtype)
        except:
            print(attention_scores.shape, attention_mask_float.shape)
            exit()
        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = (attention_probs @ value_layer_)
    
        context_layer = self._merge_heads(context_layer)
        #print(context_layer, context_layer.shape)
        #exit()
        return context_layer, key_layer_, value_layer_
