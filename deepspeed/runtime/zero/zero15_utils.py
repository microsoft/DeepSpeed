import os
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch import Tensor

from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import logger


def _log_rank0(msg):
    if dist.get_rank() == 0:
        logger.info(msg)


@torch.jit.script
def scale_tensors(tensors: List[Tensor], scale: int):
    for t in tensors:
        t.div_(scale)

class Zero15_CommGroups:
    param_zero15_shard_group=None
    param_zero15_shard_size=-1
    param_zero15_shard_rank=-1


    param_zero15_replica_group=None
    param_zero15_replica_size=-1
    param_zero15_replica_rank=-1


def _generate_zero15_config(world_size,zero15_shard_size,pp_size=1):
    config={}

    zero15_shard_group=np.arange(world_size).reshape(-1,zero15_shard_size)

    zero15_replica_group=[]

    for i in range(zero15_shard_size):
        same_shard_ranks=zero15_shard_group[:,i].tolist()
        n_ranks=len(same_shard_ranks)
        replicate_size = n_ranks // pp_size
        zero15_replica_group.extend([same_shard_ranks[j:j + replicate_size] for j in range(0, n_ranks, replicate_size)])
    config['zero15_replica_group']=zero15_replica_group
    config['zero15_shard_group']=zero15_shard_group

    return config

def create_zero15_comm_groups(
    zero15_shard_size,
    dp_group,
    hi_allreduce=False,
    mpu=None,
):
    groups=Zero15_CommGroups()

    world_size=dist.get_world_size()

    global_rank=dist.get_rank()

    config=_generate_zero15_config(world_size,zero15_shard_size,1)

    ranks_of_zero15_shard_group=config['zero15_shard_group']
    ranks_of_zero15_replica_group=config['zero15_replica_group']

    global_rank=dist.get_rank()

    for zero15_shard_rank in ranks_of_zero15_shard_group:
        _group=dist.new_group(zero15_shard_rank)
        if global_rank in zero15_shard_rank:
            groups.param_zero15_shard_group=_group
            groups.param_zero15_shard_rank=dist.get_rank(group=_group)
            groups.param_zero15_shard_size=len(zero15_shard_rank)

    
    for zero15_replica_rank in ranks_of_zero15_replica_group:
        _group=dist.new_group(zero15_replica_rank)
        if global_rank in zero15_replica_rank:
            groups.param_zero15_replica_group=_group
            groups.param_zero15_replica_rank=dist.get_rank(group=_group)
            groups.param_zero15_replica_size=len(zero15_replica_rank)

    return groups