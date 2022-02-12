'''
Copyright 2021 The Microsoft DeepSpeed Team
'''

# The file has been adapted from https://github.com/NVIDIA/Megatron-LM and retains the following license from the original file

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
"""
 Support different forms of parallelism in DeepSped using multiple process groups.
 Given that there are multiple scenarios and use-cases, this file is just retaing
 the group creation needed for the training scenario. For inference and other new
 scenarios, the code will be reused but maintained inside the scenario-specific files
"""

import torch
from deepspeed.utils import logger, log_dist

# The following groups are being used by the training engine of DeepSpeed for MoE models
# Expert parallel group that the current rank belongs to.
_EXPERT_PARALLEL_GROUP = {}
# Expert data parallel group that the current rank belongs to.
_EXPERT_DATA_PARALLEL_GROUP = {}
# torch.distributed world group needs to be cloned for some cases
_WORLD_GROUP = None


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)


# Deprecated old groups initialize function.
def initialize(ep_size=1, mpu=None):
    """
        Deprecated function. Retained for backward compatibility with old MoE/groups API usage.

        Arguments:
        ep_size (int, optional): default=1, maximum expert parallel size, which should be divisible/divided by the world size.
        by each element in num_ep_list.
        mpu (module, optional): default=None, model parallel unit (e.g., from Megatron)
            that describes model/data parallel ranks.
    """
    print(
        "Deprecation Warning! Please do not use this API as it will be deprecated in the next release. Instead, pass the desired ep_size and mpu arguments to deepspeed.moe.layer.MoE(..)"
    )
    if mpu is not None:
        log_dist(message="creating deepspeed groups using mpu", ranks=[0])
        create_expert_data_and_model_parallel(ep_size, mpu)
    else:
        log_dist(message="creating deepspeed groups", ranks=[0])
        create_model_parallel(1)
        create_expert_and_data_parallel(ep_size)


def create_model_parallel(model_parallel_size_):
    """
    Initialize model data parallel groups.

    Arguments:
        model_parallel_size: number of GPUs used to parallelize model.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model. The present function will
    create 4 model parallel groups and 2 data parallel groups as:
        4 model parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 data parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    log_dist(f'Creating model parallel group with size {model_parallel_size_}',
             ranks=[0])
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
        ranks = range(i * model_parallel_size, (i + 1) * model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if i == (rank // model_parallel_size):
            _MODEL_PARALLEL_GROUP = group


def create_expert_and_data_parallel(ep_size):
    """
        Create expert and data parallel groups.

        Note: Caller of this function is responsible to check if the groups already exist.

        Example - E + D parallel
        world_size = 16
        expert_parallel_size = 2 # number of experts in same group
        expert_data_parallel_group = [0,2,4,6,8,10,12,14], [1,3,5,7,9,11,13,15] - all reduce is only on MoE params
        expert_parallel_group = [0, 1], [2,3], [4,5], [6,7], [8,9] - no all reduce, but all to all
        data_parallel_group = [0,1,...,15] - all reduce is only on non-MoE
    """
    assert torch.distributed.is_initialized()

    log_dist(f'Creating expert and data parallel groups with size {ep_size}', ranks=[0])
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    expert_parallel_size_ = min(ep_size, world_size)
    ensure_divisibility(world_size, expert_parallel_size_)

    # Build the expert data parallel groups.
    global _EXPERT_DATA_PARALLEL_GROUP

    group_name = f"ep_size_{expert_parallel_size_}"
    for i in range(expert_parallel_size_):
        ranks = range(i, world_size, expert_parallel_size_)
        group = torch.distributed.new_group(ranks)
        log_dist(
            f'Creating expert data parallel process group named {group_name} with ranks: {list(ranks)}',
            [0])
        if i == (rank % expert_parallel_size_):
            _EXPERT_DATA_PARALLEL_GROUP[group_name] = group

    # Build the expert parallel groups.
    global _EXPERT_PARALLEL_GROUP

    for i in range(world_size // expert_parallel_size_):
        ranks = range(i * expert_parallel_size_, (i + 1) * expert_parallel_size_)
        group = torch.distributed.new_group(ranks)
        log_dist(
            f'creating expert parallel process group named {group_name} with ranks: {list(ranks)}',
            [0])
        if i == (rank // expert_parallel_size_):
            _EXPERT_PARALLEL_GROUP[group_name] = group


def create_expert_data_and_model_parallel(expert_parallel_size_, mpu):
    """
        Create expert and data parallel groups based on MPU (model parallel) group.

        Note: Caller of this function is responsible to check if the groups already exist.

        Example - E + M + D parallel
        world_size = 16
        model_degree = 2
        expert_degree = 4 # number of experts in same group
        mp_group = [0, 1], [2,3], [4,5] ...
        data_parallel_group =[0,2,4,6,8,10, 12,14],                 [1,3,5,7,9,11,13,15]
        expert_parallel_group = [0,2,4,6], [8,10,12,14]             [1,3,5,7], [9,11,13,15]
        expert_data_parallel_group = [0,8],[2,10],[4,12],[6,14],    [1,9],[3,11],[5,13],[]
    """
    assert torch.distributed.is_initialized(), "torch distributed is not initialized"
    assert mpu.model_parallel_is_initialized(), "model parallel group is not initialized"
    model_parallel_size_ = mpu.get_model_parallel_world_size()

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    dp_world_size = mpu.get_data_parallel_world_size()
    dp_rank = mpu.get_data_parallel_rank()

    log_dist(
        f"Creating deepspeed groups with model parallel size {model_parallel_size_}, expert parallel size {expert_parallel_size_}, world size {world_size}, dp world size {dp_world_size}",
        [0])

    global _EXPERT_PARALLEL_GROUP, _EXPERT_DATA_PARALLEL_GROUP

    # Get world size and rank. Ensure some consistencies.
    _DATA_PARALLEL_GROUP = mpu.get_data_parallel_group()
    _MODEL_PARALLEL_GROUP = mpu.get_model_parallel_group()

    expert_parallel_size_ = min(expert_parallel_size_, dp_world_size)
    ensure_divisibility(world_size, expert_parallel_size_)

    group_name = f"ep_size_{expert_parallel_size_}"

    for j in range(model_parallel_size_):
        for i in range(expert_parallel_size_):
            ranks = range(i * model_parallel_size_ + j,
                          world_size,
                          expert_parallel_size_ * model_parallel_size_)
            group = torch.distributed.new_group(ranks)
            if rank in list(ranks):
                _EXPERT_DATA_PARALLEL_GROUP[group_name] = group

            for i in range(dp_world_size // expert_parallel_size_):
                ranks = range(i * num_ep * model_parallel_size_ + j,
                              (i + 1) * expert_parallel_size_ * model_parallel_size_,
                              model_parallel_size_)
                group = torch.distributed.new_group(ranks)
                if rank in list(ranks):
                    _EXPERT_PARALLEL_GROUP[group_name] = group


# Deprecated as groups will not longer be a global entity. Instead check for each group independently.
def is_initialized():
    print(
        "Deprecated. Please do not use this API and instead query the individual group objects"
    )
    return False


def get_max_expert_size():
    """Get the maximum ep_size from all the created groups."""
    assert _EXPERT_PARALLEL_GROUP is not None, "Warning! Process group not initialized"
    keylist = []
    for key in _EXPERT_PARALLEL_GROUP.keys():
        # index 2 is num_experts in the key (ep_size_<num_experts>)
        index = 2
        keylist.append(int(key.split('_')[index]))
    return max(keylist) if len(keylist) > 0 else None


def get_max_expert_size_name():
    """Get the name of the group with max. ep_size"""
    return f'ep_size_{get_max_expert_size()}'


def get_max_expert_parallel_group():
    """Get the max expert parallel size."""
    return get_expert_parallel_group(get_max_expert_size_name())


def get_expert_parallel_group(group_name):
    """Get the expert parallel group the caller rank belongs to."""
    assert _EXPERT_PARALLEL_GROUP is not None, \
        'expert parallel group is not initialized'
    return _EXPERT_PARALLEL_GROUP[group_name]


def get_expert_parallel_group_dict():
    """Get the expert parallel group dict."""
    assert _EXPERT_PARALLEL_GROUP is not None, \
        'expert parallel group is not initialized'
    return _EXPERT_PARALLEL_GROUP


def get_expert_data_parallel_group(group_name):
    """Get the expert data parallel group the caller rank belongs to."""
    assert _EXPERT_DATA_PARALLEL_GROUP is not None, \
        'expert data parallel group is not initialized'
    return _EXPERT_DATA_PARALLEL_GROUP[group_name]


def get_expert_data_parallel_group_dict():
    """Get the expert data parallel group dict."""
    assert _EXPERT_DATA_PARALLEL_GROUP is not None, \
        'expert data parallel group is not initialized'
    return _EXPERT_DATA_PARALLEL_GROUP


def clone_world_group():
    """Create a clone of the world group
        Note: We need to clone the torch.distributed world group because we
        use _get_global_rank() utility function in DeepSpeed at many places.
        As that function does not work on torch.distributed.group.WORLD, we
        need to keep a clone of it.
    """
    assert torch.distributed.is_initialized(), "torch.distributed is not initialized"
    global _WORLD_GROUP
    if _WORLD_GROUP is None:
        # If not cloned already, clone the world group
        _WORLD_GROUP = torch.distributed.new_group(
            ranks=range(torch.distributed.get_world_size()))
    return _WORLD_GROUP


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert torch.distributed.is_initialized(), \
        'torch.distributed is not initialized'
    # Return the clone of torch.distributed world group
    return clone_world_group()


def get_model_parallel_world_size():
    """Return world size for the model parallel group."""
    return torch.distributed.get_world_size(group=get_model_parallel_group())


def get_expert_parallel_world_size(group_name):
    """Return world size for the expert parallel group."""
    return torch.distributed.get_world_size(group=get_expert_parallel_group(group_name))


def get_expert_data_parallel_world_size(group_name):
    """Return world size for the expert data parallel group."""
    return torch.distributed.get_world_size(
        group=get_expert_data_parallel_group(group_name))


def get_model_parallel_rank():
    """Return my rank for the model parallel group."""
    return torch.distributed.get_rank(group=get_model_parallel_group())


def get_expert_parallel_rank(group_name):
    """Return my rank for the expert parallel group."""
    return torch.distributed.get_rank(group=get_expert_parallel_group(group_name))


def get_expert_parallel_src_rank(group_name):
    """Calculate the global rank corresponding to a local rank zero
    in the expert parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_expert_parallel_world_size(group_name)
    return (global_rank // local_world_size) * local_world_size


def get_expert_data_parallel_rank(group_name):
    """Return my rank for the expert data parallel group."""
    return torch.distributed.get_rank(group=get_expert_data_parallel_group(group_name))


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())


# Retaining the different scenarios from older API but can be removed/cleaned up later on.
"""
	Various process groups need to be initialized for supporting MoE models. These differ for training and inference.

        1) For Training:  expert-parallelism (EP), tensor-slicing model-parallelism (MP), and data-parallelism (DP)
        2) For Inference: expert-parallelism (EP), expert-slicing (ES), tensor-slicing model-parallelism (MP), and data-parallelism (DP)

        DeepSpeed considers the following scenarios w.r.t. process group creation.

	For examples below, we assume a top-line of import deepspeed.utils.groups as groups

	For Inference
    -------------

	    A. User inputs:
            1) expert-parallel degree (ep_size)
            2) model-parallel degree (mp_size)
            3) expert-slicing degree (es_size)
            4) number of GPUs (world_size)
            5) number of experts (num_experts) -- this should not be needed and the caller of this groups initialize should be responsible for it

	    B. Scenarios:

	    * S1: There is no expert or model parallelism, only data parallelism (DP)::

	        model = my_model(args)
		    engine = deepspeed.init_inference(model)

	    * S2: There is expert parallelism but no model parallelism (EP)::

		# groups will be initialized here; use ep_size = world_size

            groups.initialize(ep_size=ep_size)
        	model = my_model(args)
        	engine = deepspeed.init_inference(model)

	    * S3: There is model parallelism but no expert parallelism (MP)::

		S3-A: if client initializes an mpu, then use it and don't do anything in groups API

            mpu.init()
            model = my_model(args)
            engine = deepspeed.init_inference(model, mpu=mpu) <-- this will get the mp_group from mpu

		S3-B: if client has no mpu, but want model parallelism (MP), use groups API to create group

            model = my_model(args)
            groups.initialize(mp_size=mp_size)
            engine = deepspeed.init_inference(model) <-- remove mp_size as an input arg here as the line above

	    * S4: There is expert-parallelism and tensor-slicing model parallelism (EP + MP):

		S4-A: if client initializes an mpu, then use it and don't do anything in groups API

            mpu.init()
            groups.initialize(ep_size=ep_size, mpu=mpu) <-- this will get the mp_group from mpu and create an ep_group
            model = my_model(args)
            engine = deepspeed.init_inference(model, mpu=mpu) <-- check if this mpu is same as mpu passed earlier

		S4-B: if client has no mpu, but want model parallelism (MP), use groups API to create group both MP and EP groups

            groups.initialize(ep_size=ep_size, mp_size=mp_size)
            model = my_model(args)
            engine = deepspeed.init_inference(model)

	    * S5: There is expert-parallelism, expert-slicing, and tensor-slicing model parallelism (EP + ES + MP):

		    -- Is S5 similar to S4 but the user needs to set es_size = world_size/ep_size -- Reza, please help me with this scenario

        64 GPUs, mp_size=8, dp_size=64/8=8, ep_size=dp_world_size, 64 experts,

        ep_size=128, -->  es_size=128/64

        Notes for implementation changes:
	    Note: https://github.com/microsoft/DeepSpeed/blob/df724e71e935414bb8e73b78f9f422baca344895/deepspeed/inference/engine.py#L90
            - current code is making an mp_size MP group if mpu is None but mp_size>1
            - _create_model_parallel_group
            - _create_ep_parallel_group  -- both ep_group and es_group are made here. We need to take this out and use inputs of ep_size and es_size to make these groups


    For Training
    ------------
        A. User Inputs:
            1) expert-parallel degree (ep_size)
            2) model-parallel degree (mp_size)
            3) expert-slicing degree (es_size) -- Does not work yet so ignore it
            4) number of GPUs (world_size)
            5) number of experts (num_experts) -- this should not be needed I think

        B. Scenarios:
        * S1: There is no expert parallelism or model parallelism, only data (D):

            model = my_model(args)
            engine = deepspeed.initialize(model) # initialize groups without mpu

        * S2: There is expert parallelism but no model parallelism (EP + DP)::

            deepspeed.utils.groups.initialize(ep_size) # groups will be initialized here
            model = my_model(args)
            engine = deepspeed.initialize(model)

        * S3: There is model parallelism but no expert parallelism (MP)::

            mpu.init() # client initializes it's model parallel unit
            model = my_model(args)
            engine = deepspeed.initialize(model, mpu=mpu) # init w. mpu but ep_size = dp_world_size

        * S4: There is model, data, and expert parallelism (EP + DP + MP)::

            mpu.init() # client initializes it's model parallel unit
            deepspeed.utils.groups.initialize(ep_size, mpu) # initialize expert groups wrt mpu
            model = my_model(args)
            engine = deepspeed.initialize(model, mpu=mpu) # passing mpu is optional in this case

"""
