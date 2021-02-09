# TODO: Update license and credits in this file
# Code is modified and is based on MoE layer from https://github.com/lucidrains/mixture-of-experts
# and alltoall modifications are based on Fairscale's MoE layer from:
# https://github.com/facebookresearch/fairscale/tree/master/fairscale/nn/moe

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn import Module, ModuleList

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module


import torch.distributed as dist

# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.

# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))

from torch import nn
import torch.nn.functional as F

import math


def top1gating(logits: torch.Tensor, capacity_factor: float) -> Tuple[Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    gates = F.softmax(logits, dim=1)

    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # capacity = (tokens / experts) * capacity_factor
    # round-up
    capacity = math.ceil((num_tokens / num_experts) * capacity_factor)
    # assert num_tokens % num_experts == 0

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.mean(me * ce) * num_experts * num_experts

    # Remove locations outside capacity from mask
    mask1 *= torch.lt(locations1, capacity)

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.type_as(logits)
    gates1_s = torch.einsum("se,se->s", gates, mask1_float)

    # Calculate combine_weights and dispatch_mask
    gates1 = torch.einsum("s,se->se", gates1_s, mask1_float)
    max_location1_s = locations1_s.max() + 1
    if max_location1_s > capacity:
        print("max_location1_s: " + str(max_location1_s))
        print("capacity: " + str(capacity))
    locations1_sc = F.one_hot(locations1_s, num_classes=capacity).type_as(logits)
    combine_weights = torch.einsum("se,sc->sec", gates1, locations1_sc)
    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask

class TopKGate(torch.nn.Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(self, model_dim: int, num_experts: int, capacity_factor: float = 1.0) -> None:
        super().__init__()
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        self.capacity_factor = capacity_factor

    def forward(self, input: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore
        logits = self.wg(input)
        return top1gating(logits, self.capacity_factor)

class MOELayer(Base):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """

    def __init__(self, gate: Module, experts: Module, num_local_experts: int, group: Optional[Any] = None) -> None:
        super().__init__()
        self.gate = gate
        self.experts = experts

        self.group = group if group is not None else dist.group.WORLD
        # TODO below code is in the Experts code
        # for expert in self.experts:
        #     for p in experts.parameters():
        #         p.expert = True  # type: ignore
        self.world_size = dist.get_world_size(self.group)
        self.num_local_experts = num_local_experts

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:
        assert len(input) == 1, "only single input Tensor supported"
        assert len(input[0].shape) == 3, "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"
        # removed wrong assert
        # assert input[0].shape[0] % len(self.experts) == 0, "num tokens must be order of number of local experts"

        # Implement Algorithm 2 from GShard paper.
        d_model = input[0].shape[2]
        # Reshape into S tokens by dropping sequence dimension.
        reshaped_input = input[0].reshape(-1, d_model)
        self.l_aux, combine_weights, dispatch_mask = self.gate(reshaped_input)
        dispatched_input = torch.einsum("sec,sm->ecm", dispatch_mask.type_as(input[0]), reshaped_input)
        dispatched_input = _AllToAll.apply(self.group, dispatched_input)
        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(self.world_size, self.num_local_experts, -1, d_model)
        expert_output = self.experts(dispatched_input)
        expert_output = _AllToAll.apply(self.group, expert_output)
        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(self.world_size * self.num_local_experts, -1, d_model)
        combined_output = torch.einsum("sec,ecm->sm", combine_weights, expert_output)
        return combined_output.reshape(input[0].shape)
