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
        
        output = self.softmax_context_func(query_key_value, attn_mask, self.config.rotary_dim, self.config.rotate_half,
                                           self.config.rotate_every_two, heads, norm_factor,
                                           self.config.triangular_masking, self.config.local_attention,
                                           self.config.window_size, no_masking, layer_id, num_layers, alibi,
                                           self.config.multi_query, self.num_kv_per_partition)
        
        return output
