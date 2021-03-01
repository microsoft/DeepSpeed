# TODO: Update license and credits in this file
# Code is modified and is based on MoE layer from https://github.com/lucidrains/mixture-of-experts
# and alltoall modifications are based on Fairscale's MoE layer from:
# https://github.com/facebookresearch/fairscale/tree/master/fairscale/nn/moe

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import time
from time import perf_counter 
import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn import Module, ModuleList

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

gumbel_map: Dict[torch.device, Callable] = {}

def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)

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

def top1gating(logits: torch.Tensor, capacity_factor: float, noisy_gate: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
    """Implements Top1Gating on logits."""
    # everything is in fp32 in this function
    if noisy_gate:
        logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # logits_fp32 = logits.to(torch.float32)
    gates = F.softmax(logits, dim=1)

    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # capacity = (tokens / experts) * capacity_factor
    # round-up
    capacity = math.ceil((num_tokens / num_experts) * capacity_factor)

    # Create a mask for 1st's expert per token
    # noisy gating
    indices1_s = torch.argmax(logits_w_noise if noisy_gate else gates, dim=1)
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
    mask1_float = mask1.float()
    gates1_s = torch.einsum("se,se->s", gates, mask1_float)

    # Calculate combine_weights and dispatch_mask
    gates1 = torch.einsum("s,se->se", gates1_s, mask1_float)
    max_location1_s = locations1_s.max() + 1
    if max_location1_s > capacity:
        print("max_location1_s: " + str(max_location1_s))
        print("capacity: " + str(capacity))
    locations1_sc = F.one_hot(locations1_s, num_classes=capacity).float()
    combine_weights = torch.einsum("se,sc->sec", gates1, locations1_sc)
    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask

def top2gating(logits: torch.Tensor, capacity_factor: float) -> Tuple[Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    # everything is in fp32 in this function
    # logits_fp32 = logits.to(torch.float32)
    gates = F.softmax(logits, dim=1)

    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # capacity = (2 * num_tokens // num_experts) * capacity_factor
    # round-up
    capacity = math.ceil((2*num_tokens / num_experts) * capacity_factor)

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # Create a mask for 2nd's expert per token using Gumbel-max trick
    # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # Replace top-expert with min value
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float("-inf"))
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = F.one_hot(indices2_s, num_classes=num_experts)

    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.mean(me * ce) * num_experts * num_experts

    # Remove locations outside capacity from mask
    mask1 *= torch.lt(locations1, capacity)
    mask2 *= torch.lt(locations2, capacity)

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    mask2_float = mask2.float()
    gates1_s = torch.einsum("se,se->s", gates, mask1_float)
    gates2_s = torch.einsum("se,se->s", gates, mask2_float)
    denom_s = gates1_s + gates2_s
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s

    # Calculate combine_weights and dispatch_mask
    gates1 = torch.einsum("s,se->se", gates1_s, mask1_float)
    gates2 = torch.einsum("s,se->se", gates2_s, mask2_float)
    locations1_sc = F.one_hot(locations1_s, num_classes=capacity).float()
    locations2_sc = F.one_hot(locations2_s, num_classes=capacity).float()
    combine1_sec = torch.einsum("se,sc->sec", gates1, locations1_sc)
    combine2_sec = torch.einsum("se,sc->sec", gates2, locations2_sc)
    combine_weights = combine1_sec + combine2_sec
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

    def __init__(self, model_dim: int, num_experts: int, k: int = 1,
                 capacity_factor: float = 1.0, noisy_gate: bool = False) -> None:
        super().__init__()

        # Only top-1 and top-2 are supported at the moment.
        if k != 1 and k != 2:
            raise ValueError('Only top-1 and top-2 gatings are supported.')
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False).float()
        self.k = k
        self.capacity_factor = capacity_factor
        self.noisy_gate = noisy_gate

    def forward(self, input: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore
        if self.wg.weight.dtype != torch.float32:
            self.wg = self.wg.float()
            print("Cast gate weight to float 32")
        logits = self.wg(input.float())
        if self.k == 1:
            return top1gating(logits, self.capacity_factor, self.noisy_gate)
        else:
            return top2gating(logits, self.capacity_factor)

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
        self.group = group
        self.world_size = dist.get_world_size(group)
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

        #print(f"alltoall called at rank:{dist.get_rank()} with dispatched_input shape:{dispatched_input.shape}")
        a = time.perf_counter()
        dispatched_input = _AllToAll.apply(self.group, dispatched_input)
        b = time.perf_counter()
        #print(f"alltoall took {b-a} seconds at rank:{dist.get_rank()}")
        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(self.world_size, self.num_local_experts, -1, d_model)
        #print(f"reshaped input after alltoall called at rank:{dist.get_rank()} is dispatched_input shape:{dispatched_input.shape}")
        expert_output = self.experts(dispatched_input)
        expert_output = _AllToAll.apply(self.group, expert_output)
        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(self.world_size * self.num_local_experts, -1, d_model)
        combined_output = torch.einsum("sec,ecm->sm", combine_weights.type_as(input[0]), expert_output)
        return combined_output.reshape(input[0].shape)
