# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

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
 Support different forms of parallelism in DeepSpeed using multiple process groups.
 Given that there are multiple scenarios and use-cases, this file is going to be updated
 frequently. For now, the group creation needed for the training scenario is being implemented.
 For inference and other new scenarios, the code will be either reused or added to this file.
"""

from deepspeed import comm as dist
from deepspeed.utils import log_dist
from deepspeed.utils.bwc import bwc_tensor_model_parallel_world_size, bwc_pipeline_parallel_world_size
from deepspeed.utils.exceptions import DeprecatedException
from deepspeed.accelerator import get_accelerator

# Expert parallel group that the current rank belongs to.
_EXPERT_PARALLEL_GROUP = {}
# Expert data parallel group that the current rank belongs to.
_EXPERT_DATA_PARALLEL_GROUP = {}
# dist world group needs to be cloned for some cases
_WORLD_GROUP = None
# ZeRO parameter  partitioning group that the current rank belongs to.
_ZERO_PARAM_INTRA_PARALLEL_GROUP = None
# global object to maintain mpu object if passed by a Megatron client
mpu = None
# global object that stores tensor parallel world size for experts
expert_tensor_parallel_world_size = 1
# All to All quantized graident communication groups
_ALL_TO_ALL_GROUP = {}

mesh_device = None


# Deprecated groups initialize function.
def initialize(ep_size=1, mpu=None):
    """ Deprecated function. Retained to inform the users."""
    raise DeprecatedException(
        "Please do not use the groups.initialize() API as it is deprecated. Instead, pass the desired ep_size to deepspeed.moe.layer.MoE(..,ep_size,..)"
    )


def _ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(numerator, denominator)


# ======== Start: Tensor Parallel Group Attributes ========

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None

# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None


def _init_tp_mesh_device(tensor_model_parallel_size=1, data_parallel_size=None):
    """Initialize model data parallel groups."""

    global _DATA_PARALLEL_GROUP
    global _MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GROUP

    if _TENSOR_MODEL_PARALLEL_GROUP is not None:
        return

    if data_parallel_size is None:
        data_parallel_size = dist.get_world_size() // tensor_model_parallel_size

    mesh_device = dist.initialize_mesh_device((data_parallel_size, tensor_model_parallel_size),
                                              ("data_parallel", "tensor_parallel"))
    _TENSOR_MODEL_PARALLEL_GROUP = mesh_device.get_group(mesh_dim="tensor_parallel")
    _DATA_PARALLEL_GROUP = mesh_device.get_group(mesh_dim="data_parallel")

    # They are always equal only in 2D (DP + TP) parallelism.
    # _MODEL_PARALLEL_GROUP is assigned the same value as _TENSOR_MODEL_PARALLEL_GROUP
    # to allow for future potential changes.
    _MODEL_PARALLEL_GROUP = _TENSOR_MODEL_PARALLEL_GROUP

    return _DATA_PARALLEL_GROUP, _MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""

    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, \
        'intra_layer_model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""

    assert _MODEL_PARALLEL_GROUP is not None, \
        'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP


def set_tensor_model_parallel_world_size(world_size):
    """Set the tensor model parallel size"""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return dist.get_world_size(group=get_tensor_model_parallel_group())


def get_model_parallel_world_size():
    return get_tensor_model_parallel_world_size()


def set_tensor_model_parallel_rank(rank):
    """Set tensor model parallel rank."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = rank


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return dist.get_rank(group=get_tensor_model_parallel_group())


def get_model_parallel_rank():
    return get_tensor_model_parallel_rank()


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = dist.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return dist.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return dist.get_rank(group=get_data_parallel_group())


# ======== End: Tensor Parallel Group Attributes ========


# Not currently used. Helper function to create a model (tensor) parallel group.
def _create_model_parallel(model_parallel_size_):
    """
    Initialize model data parallel groups.

    Arguments:
        model_parallel_size: number of GPUs used to parallelize model.

    Returns:
        Tuple of data parallel group and model parallel group

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
    log_dist(f'Creating model parallel group with size {model_parallel_size_}', ranks=[0])
    # Get world size and rank. Ensure some consistencies.
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    model_parallel_size = min(model_parallel_size_, world_size)
    _ensure_divisibility(world_size, model_parallel_size)
    rank = dist.get_rank()

    _DATA_PARALLEL_GROUP = None
    _MODEL_PARALLEL_GROUP = None
    # Build the data parallel groups.
    for i in range(model_parallel_size):
        ranks = range(i, world_size, model_parallel_size)
        group = dist.new_group(ranks)
        if i == (rank % model_parallel_size):
            _DATA_PARALLEL_GROUP = group

    # Build the model parallel groups.
    for i in range(world_size // model_parallel_size):
        ranks = range(i * model_parallel_size, (i + 1) * model_parallel_size)
        group = dist.new_group(ranks)
        if i == (rank // model_parallel_size):
            _MODEL_PARALLEL_GROUP = group

    return _DATA_PARALLEL_GROUP, _MODEL_PARALLEL_GROUP


def _create_expert_and_data_parallel(expert_parallel_size_, use_data_before_expert_parallel_=False):
    """
        Create expert and data parallel groups.

        Note: Caller of this function is responsible to check if the groups already exist.

        Example - E + D parallel
        world_size = 16
        expert_parallel_size = 2 # number of experts in same group
        expert_data_parallel_group = [0,2,4,6,8,10,12,14], [1,3,5,7,9,11,13,15] - all reduce is only on MoE params
        expert_parallel_group = [0, 1], [2,3], [4,5], [6,7], [8,9] - no all reduce, but all to all
        data_parallel_group = [0,1,...,15] - all reduce is only on non-MoE
        use_data_before_expert_parallel_ (bool): Use the D + E instead of E + D topology
    """
    assert dist.is_initialized()

    log_dist(f'Creating expert and data parallel groups with size {expert_parallel_size_}', ranks=[0])
    world_size = dist.get_world_size()
    pp_world_size = 1 if mpu is None else bwc_pipeline_parallel_world_size(mpu)
    rank = dist.get_rank()

    pp_stride = world_size // pp_world_size
    _ensure_divisibility(pp_stride, expert_parallel_size_)

    group_name = f"ep_size_{expert_parallel_size_}"

    # Build the expert data parallel groups.
    global _EXPERT_DATA_PARALLEL_GROUP

    ep_stride = pp_stride // expert_parallel_size_

    # Only create group if it does not already exist
    if group_name not in _EXPERT_DATA_PARALLEL_GROUP:
        for pp_stage_start in range(0, world_size, pp_stride):
            for i in range(expert_parallel_size_):
                if use_data_before_expert_parallel_:
                    ranks = range(pp_stage_start + i * ep_stride, pp_stage_start + (i + 1) * ep_stride)
                else:
                    ranks = range(pp_stage_start + i, pp_stage_start + pp_stride, expert_parallel_size_)
                group = dist.new_group(ranks)
                log_dist(
                    f'Creating expert data parallel process group named {group_name} '
                    f'with ranks: {list(ranks)}', [0])
                if rank in ranks:
                    _EXPERT_DATA_PARALLEL_GROUP[group_name] = group

    # Build the expert parallel groups.
    global _EXPERT_PARALLEL_GROUP

    # Only create group if it does not already exist
    if group_name not in _EXPERT_PARALLEL_GROUP:
        if use_data_before_expert_parallel_:
            for pp_stage_start in range(0, world_size, pp_stride):
                for i in range(ep_stride):
                    ranks = range(pp_stage_start + i, pp_stage_start + pp_stride, ep_stride)
                    group = dist.new_group(ranks)
                    log_dist(
                        f'creating expert parallel process group named {group_name} '
                        f'with ranks: {list(ranks)}', [0])
                    if rank in ranks:
                        _EXPERT_PARALLEL_GROUP[group_name] = group
        else:
            for i in range(world_size // expert_parallel_size_):
                ranks = range(i * expert_parallel_size_, (i + 1) * expert_parallel_size_)
                group = dist.new_group(ranks)
                log_dist(f'creating expert parallel process group named {group_name} '
                         f'with ranks: {list(ranks)}', [0])
                if rank in ranks:
                    _EXPERT_PARALLEL_GROUP[group_name] = group


def _get_expert_parallel_ranks(world_size,
                               tensor_parallel_size_,
                               expert_parallel_size_,
                               pipeline_parallel_size_=1,
                               use_data_before_expert_parallel_=False):
    """Generate expert parallel and expert data parallel group ranks list.

        Example - E + M + D parallel
        world_size = 16
        model_degree = 2
        expert_degree = 4 # number of experts in same group
        mp_group = [0, 1], [2,3], [4,5] ...
        data_parallel_group =[0,2,4,6,8,10, 12,14],                 [1,3,5,7,9,11,13,15]
        expert_parallel_group = [0,2,4,6], [8,10,12,14]             [1,3,5,7], [9,11,13,15]
        expert_data_parallel_group = [0,8],[2,10],[4,12],[6,14],    [1,9],[3,11],[5,13],[7,15]

    Args:
        world_size (int): Distributed world size.
        tensor_parallel_size_ (int): Tensor parallel group size.
        expert_parallel_size_ (int): Expert parallel group size.
        pipeline_parallel_size_ (int): Pipeline parallel group size
        use_data_before_expert_parallel_ (bool): Use the D + E instead of E + D topology
    Returns:
        Expert parallel group ranks and Expert data parallel group ranks list.
    """
    _ensure_divisibility(world_size, tensor_parallel_size_ * pipeline_parallel_size_)
    dp_world_size = world_size // (tensor_parallel_size_ * pipeline_parallel_size_)
    _ensure_divisibility(dp_world_size, expert_parallel_size_)

    # Generate data parallel groups
    data_parallel_groups = []
    dp_group_size = tensor_parallel_size_
    pp_stride = world_size // pipeline_parallel_size_

    if use_data_before_expert_parallel_:
        dp_stride = world_size // expert_parallel_size_ // tensor_parallel_size_ // pipeline_parallel_size_
        for pp_stage_start in range(0, world_size, pp_stride):
            pp_stage_next = pp_stage_start + pp_stride
            for i in range(dp_group_size):
                data_parallel_groups.append(list())
                for ds in range(dp_stride):
                    # [0, 4, 8, 12, 16, 20, 24, 28, 2, 6, 10, 14, 18, 22, 26, 30]
                    # [1, 5, 9, 13, 17, 21, 25, 29, 3, 7, 11, 15, 19, 23, 27, 31]
                    data_parallel_groups[-1].extend(
                        list(
                            range(pp_stage_start + i + ds * tensor_parallel_size_, pp_stage_next,
                                  dp_stride * tensor_parallel_size_)))
    else:
        for pp_stage_start in range(0, world_size, pp_stride):
            pp_stage_next = pp_stage_start + pp_stride
            for i in range(dp_group_size):
                data_parallel_groups.append(list(range(pp_stage_start + i, pp_stage_next, dp_group_size)))

    expert_parallel_groups = []
    expert_data_parallel_groups = []
    for dp_ranks in data_parallel_groups:
        # partition of expert parallel groups, e.g. [0,2,4,6], [8,10,12,14]
        part_ep_groups = []
        for i in range(0, dp_world_size, expert_parallel_size_):
            part_ep_groups.append(dp_ranks[i:i + expert_parallel_size_])
        expert_parallel_groups.extend(part_ep_groups)

        # zip part_ep_groups get expert data parallel ranks, e.g [0,8],[2,10],[4,12],[6,14]
        for expert_dp_ranks in zip(*part_ep_groups):
            expert_data_parallel_groups.append(list(expert_dp_ranks))

    return expert_parallel_groups, expert_data_parallel_groups


def _create_expert_data_and_model_parallel(expert_parallel_size_, mpu, use_data_before_expert_parallel_=False):
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
        expert_data_parallel_group = [0,8],[2,10],[4,12],[6,14],    [1,9],[3,11],[5,13],[7,15]
    """
    assert dist.is_initialized(), "dist is not initialized"
    tensor_parallel_size_ = bwc_tensor_model_parallel_world_size(mpu)

    global expert_tensor_parallel_world_size
    expert_tensor_parallel_world_size = tensor_parallel_size_

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    dp_world_size = mpu.get_data_parallel_world_size()
    pp_world_size = 1 if mpu is None else bwc_pipeline_parallel_world_size(mpu)

    _ensure_divisibility(world_size, tensor_parallel_size_)
    _ensure_divisibility(dp_world_size, expert_parallel_size_)

    log_dist(
        f"Creating deepspeed groups with model parallel size {tensor_parallel_size_}, "
        f"pipeline parallel size {pp_world_size}, expert parallel size {expert_parallel_size_}, "
        f"world size {world_size}, dp world size {dp_world_size}", [0])

    global _EXPERT_PARALLEL_GROUP, _EXPERT_DATA_PARALLEL_GROUP

    group_name = f"ep_size_{expert_parallel_size_}"

    # Only create groups if they don't already exist
    # Need to check conditions outside the group creation loop because of the way torch.dist group creation works
    if group_name not in _EXPERT_DATA_PARALLEL_GROUP and group_name not in _EXPERT_PARALLEL_GROUP:
        expert_parallel_groups, expert_data_parallel_groups = _get_expert_parallel_ranks(
            world_size, tensor_parallel_size_, expert_parallel_size_, pp_world_size, use_data_before_expert_parallel_)
        for ranks in expert_parallel_groups:
            group = dist.new_group(ranks)
            if rank in list(ranks):
                _EXPERT_PARALLEL_GROUP[group_name] = group

        for ranks in expert_data_parallel_groups:
            group = dist.new_group(ranks)
            if rank in list(ranks):
                _EXPERT_DATA_PARALLEL_GROUP[group_name] = group


def _get_max_expert_size():
    """Get the maximum ep_size from all the created groups."""
    assert _EXPERT_PARALLEL_GROUP is not None, "Warning! Process group not initialized"
    keylist = []
    for key in _EXPERT_PARALLEL_GROUP.keys():
        # index 2 is ep_size in the group name: ep_size_<ep_size>
        index = 2
        keylist.append(int(key.split('_')[index]))
    return max(keylist) if len(keylist) > 0 else None


def _get_max_expert_size_name():
    """Get the name of the group with max. ep_size"""
    return f'ep_size_{_get_max_expert_size()}'


def _get_max_expert_parallel_group():
    """Get the max expert parallel size."""
    return _get_expert_parallel_group(_get_max_expert_size_name())


def _get_expert_parallel_group(group_name):
    """Get the expert parallel group the caller rank belongs to."""
    assert group_name in _EXPERT_PARALLEL_GROUP, \
        'expert parallel group is not initialized'
    return _EXPERT_PARALLEL_GROUP[group_name]


def _get_expert_parallel_group_dict():
    """Get the expert parallel group dict."""
    return _EXPERT_PARALLEL_GROUP


def _get_expert_data_parallel_group(group_name):
    """Get the expert data parallel group the caller rank belongs to."""
    assert group_name in _EXPERT_DATA_PARALLEL_GROUP, \
        'expert data parallel group is not initialized'
    return _EXPERT_DATA_PARALLEL_GROUP[group_name]


def _get_expert_data_parallel_group_dict():
    """Get the expert data parallel group dict."""
    return _EXPERT_DATA_PARALLEL_GROUP


def _clone_world_group():
    """Create a clone of the world group
        Note: We need to clone the dist world group because we
        use dist.get_global_rank() utility function in DeepSpeed at many places.
        As that function does not work on dist.group.WORLD, we
        need to keep a clone of it.
    """
    assert dist.is_initialized(), "dist is not initialized"
    global _WORLD_GROUP
    if _WORLD_GROUP is None:
        # If not cloned already, clone the world group
        _WORLD_GROUP = dist.new_group(ranks=range(dist.get_world_size()))
    return _WORLD_GROUP


def _get_local_all_to_all_group():
    assert dist.is_initialized(), 'dist is not initialized'
    global _ALL_TO_ALL_GROUP
    device_per_node = get_accelerator().device_count()
    num_local = dist.get_world_size() // device_per_node
    if num_local == 0 and dist.get_world_size() > 0:
        assert dist.get_world_size() >= 1, 'num_gpus must >=1, cannot initialize All-To-All'
        cur_rank = []
        for i in range(dist.get_world_size()):
            cur_rank.append(i)
        _ALL_TO_ALL_GROUP['local_0'] = dist.new_group(ranks=cur_rank)
    elif num_local == 1:
        assert dist.get_world_size(
        ) == device_per_node, 'num_gpus not equal to device per node, cannot initialize All-To-All'
        _ALL_TO_ALL_GROUP['local_0'] = dist.new_group(ranks=[i for i in range(device_per_node)])
    else:
        assert dist.get_world_size() > device_per_node, 'num_nodes<2 cannot initialize All-To-All'
        for i in range(num_local):
            local_rank = [j + device_per_node * i for j in range(device_per_node)]
            _ALL_TO_ALL_GROUP[f"local_{i}"] = dist.new_group(ranks=local_rank)

        for i in range(device_per_node):
            cur_rank = []
            for j in range(num_local):
                cur_rank.append(i + j * device_per_node)
            _ALL_TO_ALL_GROUP[f"global_{i}"] = dist.new_group(ranks=cur_rank)
    return _ALL_TO_ALL_GROUP


def _get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert dist.is_initialized(), 'dist is not initialized'
    global mpu
    if mesh_device is not None:
        return mesh_device.get_group(mesh_dim="data_parallel")
    if mpu is not None:
        return mpu.get_data_parallel_group()

    # Return the clone of dist world group
    return _clone_world_group()


def _get_broadcast_src_rank():
    return dist.get_global_rank(_get_sequence_data_parallel_group(), 0)


def _get_expert_broadcast_src_rank(group_name):
    return dist.get_global_rank(_get_expert_data_parallel_group(group_name), 0)


def _get_expert_parallel_world_size(group_name):
    """Return world size for the expert parallel group."""
    return dist.get_world_size(group=_get_expert_parallel_group(group_name))


def _get_expert_data_parallel_world_size(group_name):
    """Return world size for the expert data parallel group."""
    return dist.get_world_size(group=_get_expert_data_parallel_group(group_name))


def _get_expert_parallel_rank(group_name):
    """Return my rank for the expert parallel group."""
    return dist.get_rank(group=_get_expert_parallel_group(group_name))


def _get_expert_parallel_src_rank(group_name):
    """Calculate the global rank corresponding to a local rank zero
    in the expert parallel group."""
    global_rank = dist.get_rank()
    local_world_size = _get_expert_parallel_world_size(group_name)
    return (global_rank // local_world_size) * local_world_size


def _get_expert_data_parallel_rank(group_name):
    """Return my rank for the expert data parallel group."""
    return dist.get_rank(group=_get_expert_data_parallel_group(group_name))


def _get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    if mesh_device is not None:
        return dist.get_world_size(mesh_device.get_group(mesh_dim="data_parallel"))
    global mpu
    if mpu is not None:
        return mpu.get_data_parallel_world_size()
    return dist.get_world_size(group=_get_data_parallel_group())


def _get_model_parallel_world_size():
    """Return world size for the model parallel group."""
    global mpu
    if mpu is not None:
        return mpu.get_model_parallel_world_size()
    return 1


def _get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return dist.get_rank(group=_get_data_parallel_group())


def _get_sequence_parallel_world_size():
    """Return world size for the model parallel group."""
    global mpu
    if mesh_device is not None:
        return dist.get_world_size(mesh_device.get_group(mesh_dim="sequence_parallel"))
    if mpu is not None and hasattr(mpu, 'get_sequence_parallel_world_size'):
        return mpu.get_sequence_parallel_world_size()
    return 1


def _get_sequence_parallel_rank():
    """Return my rank for the data parallel group."""
    global mpu
    if mpu is not None and hasattr(mpu, 'get_sequence_parallel_rank'):
        return mpu.get_sequence_parallel_rank()
    if mesh_device is not None:
        return dist.get_rank(mesh_device.get_group(mesh_dim="sequence_parallel"))
    return 0


def _get_sequence_parallel_group():
    global mpu
    if mpu is None or not hasattr(mpu, 'get_sequence_parallel_group'):
        if mesh_device is None:
            raise KeyError("No sequence parallel group found")
        return mesh_device.get_group(mesh_dim="sequence_parallel")
    return mpu.get_sequence_parallel_group()


def _get_sequence_data_parallel_world_size():
    """Return world size for the model parallel group."""
    global mpu
    if mpu is not None and hasattr(mpu, 'get_sequence_data_parallel_world_size'):
        return mpu.get_sequence_data_parallel_world_size()
    return _get_data_parallel_world_size()


def _get_sequence_data_parallel_rank():
    """Return my rank for the data parallel group."""
    global mpu
    if mpu is not None and hasattr(mpu, 'get_sequence_data_parallel_rank'):
        return mpu.get_sequence_data_parallel_rank()
    return _get_data_parallel_rank()


def _get_sequence_data_parallel_group():
    global mpu
    # When sequence parallelism is enabled, the process group for zero sharding and
    # gradient allreduce must be across both dimensions of data and sequence parallelism.
    if mpu is not None and hasattr(mpu, 'get_sequence_data_parallel_group'):
        return mpu.get_sequence_data_parallel_group()
    return _get_data_parallel_group()


def _get_expert_model_parallel_world_size():
    global expert_tensor_parallel_world_size
    return expert_tensor_parallel_world_size


def _create_zero_param_parallel_group(group_size):
    """
        Create parameter partitioning group within ZeRO data parallel groups.

        Example - ZP + D parallel
        world_size = 16
        zero_hpz_partition_size = 2 # number of ranks with replicated params (dual partitioning)
        zero_param_intra_parallel_group = [0, 1], [2,3], [4,5], [6,7], [8,9] - segmented (subgroup) with rep partition
        data_parallel_group = [0,1,...,15] - all reduce is on ZeRO model
    """
    assert dist.is_initialized()
    global _ZERO_PARAM_INTRA_PARALLEL_GROUP
    # Only create group if it does not already exist
    assert _ZERO_PARAM_INTRA_PARALLEL_GROUP is None, \
        'ZeRO parameter intra parallel group is already initialized'

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    zero_param_parallel_size_ = min(group_size, world_size)
    _ensure_divisibility(world_size, zero_param_parallel_size_)

    # Build the ZeRO param intra parallel groups.
    for i in range(world_size // zero_param_parallel_size_):
        ranks = range(i * zero_param_parallel_size_, (i + 1) * zero_param_parallel_size_)
        group = dist.new_group(ranks)
        if i == (rank // zero_param_parallel_size_):
            _ZERO_PARAM_INTRA_PARALLEL_GROUP = group


def _get_zero_param_intra_parallel_group():
    """Get the ZeRO parameter partitioning intra parallel group the caller rank belongs to."""
    #assert _ZERO_PARAM_INTRA_PARALLEL_GROUP is not None, \
    #    'ZeRO parameter partitioning group is not initialized'
    #TODO: Add warning
    return _ZERO_PARAM_INTRA_PARALLEL_GROUP


def _zero_param_parallel_is_initialized():
    """Check if ZeRO data parallel with parameter partititioning groups are initialized."""
    ###TODO: assert that MPU is not set
    if _ZERO_PARAM_INTRA_PARALLEL_GROUP is None and _DATA_PARALLEL_GROUP is None:
        return False


def _get_zero_param_intra_parallel_rank_in_mygroup():
    """Return my rank for the ZeRO parameter inter parallel group."""
    return dist.get_rank(group=_get_zero_param_intra_parallel_group())


def _get_zero_param_intra_parallel_group_world_size():
    """Return world size for the ZeRO parameter parallel group."""
    return dist.get_world_size(group=_get_zero_param_intra_parallel_group())


def _get_zero_param_intra_parallel_group_ranks():
    """Return all ranks for the ZeRO parameter intra parallel group."""
    return dist.get_all_ranks_from_group(group=_get_zero_param_intra_parallel_group())
