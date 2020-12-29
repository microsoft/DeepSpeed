# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from .initialize import get_model_parallel_group
from .utils import split_tensor_along_last_dim

from deepspeed.utils import event_manager

def _reduce(input_):
    """All-reduce the the input tensor across model parallel group."""
    group = get_model_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # All-reduce.
    with event_manager.timespan("_reduce::all_reduce"):
        torch.distributed.all_reduce(input_, group=group)

    return input_


def _split(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    group = get_model_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Split along last dimension.
    world_size = torch.distributed.get_world_size(group=group)
    with event_manager.timespan("_split::split_tensor_along_last_dim"):
        input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = torch.distributed.get_rank(group=group)
    output = input_list[rank].contiguous()

    return output


def _gather(input_):
    """Gather tensors and concatinate along the last dimension."""
    group = get_model_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    with event_manager.timespan("_gather::all_gather"):
        torch.distributed.all_gather(tensor_list, input_, group=group)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        event_manager.init_current_thread("backward_thread")
        with event_manager.timespan("copy_to_model_parallel_region"):
            return _reduce(grad_output)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-redcue the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_):
        with event_manager.timespan("reduce_from_model_parallel_region"):
            return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def forward(ctx, input_):
        with event_manager.timespan("scatter_to_model_parallel_region_forward"):
            return _split(input_)

    @staticmethod
    def backward(ctx, grad_output):
        event_manager.init_current_thread("backward_thread")
        with event_manager.timespan("scatter_to_model_parallel_region_backward"):
            return _gather(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def forward(ctx, input_):
        with event_manager.timespan("gather_from_model_parallel_region_forward"):
            return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):
        event_manager.init_current_thread("backward_thread")
        with event_manager.timespan("gather_from_model_parallel_region_backward"):
            return _split(grad_output)


# -----------------
# Helper functions.
# -----------------

def copy_to_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)

def reduce_from_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)

def scatter_to_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)

def gather_from_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)
