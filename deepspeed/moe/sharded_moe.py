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


# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.

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

# # plain mixture of experts

# class ShardedMoE(nn.Module):
#     def __init__(self,
#         dim,
#         num_experts = 16,
#         hidden_dim = None,
#         activation = nn.ReLU,
#         # second_policy_train = 'random',
#         # second_policy_eval = 'random',
#         # second_threshold_train = 0.2,
#         # second_threshold_eval = 0.2,
#         capacity_factor_train = 1.25,
#         capacity_factor_eval = 2.,
#         loss_coef = 1e-2,
#         experts = None):
#         super().__init__()

#         self.num_experts = num_experts

#         # change to top-1 gating
#         gating_kwargs = {'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}
#         self.gate = Top1Gating(dim, num_gates = num_experts, **gating_kwargs)
#         # gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train, 'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}
#         # self.gate = Top2Gating(dim, num_gates = num_experts, **gating_kwargs)
#         self.experts = default(experts, lambda: Experts(dim, num_experts = num_experts, hidden_dim = hidden_dim, activation = activation))
#         self.loss_coef = loss_coef

#     def forward(self, inputs, **kwargs):
#         b, n, d, e = *inputs.shape, self.num_experts
#         dispatch_tensor, combine_tensor, loss = self.gate(inputs)
#         expert_inputs = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor.half())
#         ourgroup = dist.group.WORLD 
#         expert_inputs = _AllToAll.apply(ourgroup, expert_inputs)

#         # Now feed the expert inputs through the experts.
#         orig_shape = expert_inputs.shape
#         expert_inputs = expert_inputs.reshape(e, -1, d)
#         expert_outputs = self.experts(expert_inputs)
#         expert_outputs = expert_outputs.reshape(*orig_shape)
#         expert_outputs = _AllToAll.apply(ourgroup, expert_outputs)

#         output = torch.einsum('ebcd,bnec->bnd', expert_outputs, combine_tensor.half())
#         return output, loss * self.loss_coef
