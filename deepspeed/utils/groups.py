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


"""Model, expert and data parallel groups."""

import torch


# Model parallel group that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Expert parallel group that the current rank belongs to.
_EXPERT_PARALLEL_GROUP = None
# Expert data parallel group that the current rank belongs to.
_EXPERT_DATA_PARALLEL_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None

def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)


def initialize_model_parallel(model_parallel_size_):
    """
    Initialize model data parallel groups.

    Arguments:
        model_parallel_size: number of GPUs used to parallelize model.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model. The present function will
    create 4 model parallel groups and 2 data parallel grous as:
        4 model parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 data parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    if torch.distributed.get_rank() == 0:
        print('> initializing model parallel with size {}'.format(
            model_parallel_size_))
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    model_parallel_size = min(model_parallel_size_, world_size)
    ensure_divisibility(world_size, model_parallel_size)
    rank = torch.distributed.get_rank()

    # Build the data parallel groups.
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, \
        'data parallel group is already initialized'
    for i in range(model_parallel_size):
        ranks = range(i, world_size, model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if i == (rank % model_parallel_size):
            _DATA_PARALLEL_GROUP = group

    # Build the model parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, \
        'model parallel group is already initialized'
    for i in range(world_size // model_parallel_size):
        ranks = range(i * model_parallel_size,
                      (i + 1) * model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if i == (rank // model_parallel_size):
            _MODEL_PARALLEL_GROUP = group


def initialize_expert_parallel(expert_parallel_size_):
    """
        Initialize expert plus data parallel groups.

        Example - E + D parallel
        world_size = 16
        expert_parallel_size = 2 # number of experts in same group
        expert_data_parallel_group = [0,2,4,6,8,10,12,14], [1,3,5,7,9,11,13,15] - all reduce is only on MoE params
        expert_parallel_group = [0, 1], [2,3], [4,5], [6,7], [8,9] - no all reduce, but all to all
        data_parallel_group = [0,1,...,15] - all reduce is only on non-MoE
    """
    if torch.distributed.get_rank() == 0:
        print('> initializing expert parallel with size {}'.format(
            expert_parallel_size_))
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    expert_parallel_size_ = min(expert_parallel_size_, world_size)
    ensure_divisibility(world_size, expert_parallel_size_)
    rank = torch.distributed.get_rank()

    # TODO: Requires assertion here
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = torch.distributed.group.WORLD

    # Build the data parallel groups.
    global _EXPERT_DATA_PARALLEL_GROUP
    assert _EXPERT_DATA_PARALLEL_GROUP is None, \
        'expert data parallel group is already initialized'
    for i in range(expert_parallel_size_):
        ranks = range(i, world_size, expert_parallel_size_)
        group = torch.distributed.new_group(ranks)

        # TODO: remove
        if rank == 0:
            print(f'Creating Expert data parallel process group with ranks: {list(ranks)}')
        if i == (rank % expert_parallel_size_):
            _EXPERT_DATA_PARALLEL_GROUP = group

    # Build the model parallel groups.
    global _EXPERT_PARALLEL_GROUP
    assert _EXPERT_PARALLEL_GROUP is None, \
        'expert parallel group is already initialized'
    for i in range(world_size // expert_parallel_size_):
        ranks = range(i * expert_parallel_size_,
                      (i + 1) * expert_parallel_size_)
        group = torch.distributed.new_group(ranks)

        # TODO: remove
        if rank == 0:
            print(f'Creating Expert parallel process group with ranks: {list(ranks)}')
        if i == (rank // expert_parallel_size_):
            _EXPERT_PARALLEL_GROUP = group

# TODO: Implement
def initialize_model_and_expert_parallel(model_parallel_size_, expert_parallel_size_):
    """
    Example - E + M + D parallel
    world_size = 16
    model_degree = 2
    expert_degree = 4 # number of experts in same group
    mp_group = [0, 1], [2,3], [4,5]...
    expert_parallel_group = [0,1,2,3], [4,5,6,7] ...
    expert_data_parallel_group = [0,4,8,12], [1,5,9,13], ...
    data_parallel_group = [0,2,4,6,8,10,12,14], [1,3,5,7,9,11,13,15]
    """
    raise NotImplementedError()


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if _MODEL_PARALLEL_GROUP is None or _DATA_PARALLEL_GROUP is None:
        return False
    return True


def expert_parallel_is_initialized():
    """Check if expert and expert data parallel groups are initialized."""
    if _EXPERT_PARALLEL_GROUP is None or _EXPERT_DATA_PARALLEL_GROUP is None:
        return False
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, \
        'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP


def get_expert_parallel_group():
    """Get the expert parallel group the caller rank belongs to."""
    assert _EXPERT_PARALLEL_GROUP is not None, \
        'expert parallel group is not initialized'
    return _EXPERT_PARALLEL_GROUP


def get_expert_data_parallel_group():
    """Get the expert data parallel group the caller rank belongs to."""
    assert _EXPERT_DATA_PARALLEL_GROUP is not None, \
        'expert data parallel group is not initialized'
    return _EXPERT_DATA_PARALLEL_GROUP


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP


def get_model_parallel_world_size():
    """Return world size for the model parallel group."""
    return torch.distributed.get_world_size(group=get_model_parallel_group())


def get_expert_parallel_world_size():
    """Return world size for the expert parallel group."""
    return torch.distributed.get_world_size(group=get_expert_parallel_group())


def get_expert_data_parallel_world_size():
    """Return world size for the expert data parallel group."""
    return torch.distributed.get_world_size(group=get_expert_data_parallel_group())


def get_model_parallel_rank():
    """Return my rank for the model parallel group."""
    return torch.distributed.get_rank(group=get_model_parallel_group())


def get_expert_parallel_rank():
    """Return my rank for the expert parallel group."""
    return torch.distributed.get_rank(group=get_expert_parallel_group())


def get_model_parallel_src_rank():
    """Calculate the global rank corresponding to a local rank zero
    in the model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_expert_parallel_src_rank():
    """Calculate the global rank corresponding to a local rank zero
    in the expert parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_expert_parallel_group()
    return (global_rank // local_world_size) * local_world_size


def get_expert_parallel_world_size():
    """Return world size for the expert parallel group."""
    return torch.distributed.get_world_size(group=get_expert_parallel_group())


def get_expert_data_parallel_rank():
    """Return my rank for the expert data parallel group."""
    return torch.distributed.get_rank(group=get_expert_data_parallel_group())


def get_expert_data_parallel_rank():
    """Return my rank for the expert data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())


def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None