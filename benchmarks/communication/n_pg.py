import torch
#import torch.distributed as dist
import deepspeed.comm as dist
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=-1)
args = parser.parse_args()

#dist.init_process_group(backend='nccl')
deepspeed.init_distributed(dist_backend='nccl', use_deepspeed=True)

torch.cuda.set_device(args.local_rank)
device = torch.device("cuda", args.local_rank)

world_size = dist.get_world_size()
model_parallel_size = 4

_DATA_PARALLEL_GROUP = None
_MODEL_PARALLEL_GROUP = None

rank = dist.get_rank()
for i in range(model_parallel_size):
    ranks = range(i, world_size, model_parallel_size)
    group = dist.new_group(ranks)
    if i == (rank % model_parallel_size):
        _DATA_PARALLEL_GROUP = group

for i in range(world_size // model_parallel_size):
    ranks = range(i * model_parallel_size, (i + 1) * model_parallel_size)
    group = dist.new_group(ranks)
    if i == (rank // model_parallel_size):
        _MODEL_PARALLEL_GROUP = group


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, \
        'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP


def get_model_parallel_rank():
    """Return my rank for the model parallel group."""
    return dist.get_rank(group=get_model_parallel_group())


def ag_test():
    src_rank = get_model_parallel_rank()
    mats = []
    for _ in range(dist.get_world_size(get_data_parallel_group())):
        mats.append(
            torch.rand(1,
                       268 * 1024 * 1024 //
                       dist.get_world_size(get_data_parallel_group()),
                       device=device))
    dist.all_gather(mats,
                    mats[dist.get_rank(get_data_parallel_group())],
                    group=get_data_parallel_group())


for _ in range(100):
    ag_test()
