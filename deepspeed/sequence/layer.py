# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from typing import Any, Tuple
from torch import Tensor
from torch.nn import Module

import deepspeed.comm as dist


def single_all_to_all(input, scatter_idx, gather_idx, group):
    seq_world_size = dist.get_world_size(group)
    inp_shape = list(input.shape)
    inp_shape[scatter_idx] = inp_shape[scatter_idx] // seq_world_size
    if scatter_idx < 2:
        input_t = input.reshape(
            [seq_world_size, inp_shape[scatter_idx]] + \
            inp_shape[scatter_idx + 1:]
        ).contiguous()
    else:
        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        input_t = input.reshape(
            [-1, seq_world_size, inp_shape[scatter_idx]] + \
            inp_shape[scatter_idx + 1:]
        ).transpose(0, 1).contiguous()

    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=group)

    # if scattering the seq-dim, transpose the heads back to the original dimension
    if scatter_idx < 2:
        output = output.transpose(0, 1).contiguous()

    return output.reshape(
        inp_shape[: gather_idx] + \
        [inp_shape[gather_idx] * seq_world_size,] + \
        inp_shape[gather_idx + 1:]).contiguous()


class _SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor, scatter_idx: int, gather_idx: int) -> Tensor:

        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        return single_all_to_all(input, scatter_idx, gather_idx, group)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        return (None, _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx), None, None)


class DistributedAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(self,
                 local_attention: Module,
                 sequence_process_group: dist.ProcessGroup,
                 scatter_idx: int = 2,
                 gather_idx: int = 0,
                 hidden_size_per_attention_head: int = 128,
                 num_q_per_kv: int = -1) -> None:

        super(DistributedAttention, self).__init__()
        self.local_attn = local_attention
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.num_q_per_kv = num_q_per_kv

    def forward(self, mixed_x_layer: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """ forward

        Arguments:
            mixed_x_layer including:
                1. query (Tensor): query input to the layer
                2. key (Tensor): key input to the layer
                3. value (Tensor): value input to the layer
            args: other args
            kwargs: other kw args
        Returns:
            * output (Tensor): context output
        """
        sq, bs = mixed_x_layer.shape[:2]
        if self.num_q_per_kv > 0 and \
            mixed_x_layer.shape[-1] % ((self.num_q_per_kv + 2) * self.hidden_size_per_attention_head) == 0:
            intermediate_shape = (sq, bs, -1, (self.num_q_per_kv + 2), self.hidden_size_per_attention_head)
        else:
            intermediate_shape = (sq, bs, -1, self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*intermediate_shape)
        mixed_x_layer = _SeqAllToAll.apply(self.spg, mixed_x_layer, self.scatter_idx, self.gather_idx)

        #out shape : e.g., [s:h/p:]
        context_layer = self.local_attn(mixed_x_layer.reshape(sq, bs, -1), *args, **kwargs)

        output = _SeqAllToAll.apply(self.spg, context_layer, self.gather_idx, self.scatter_idx)

        #out e.g., [s/p::h]
        return output
