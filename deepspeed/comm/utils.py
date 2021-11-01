import os
import enum
import torch


class ReduceOp(enum.Enum):
    SUM = 0
    PRODUCT = 1
    MIN = 2
    MAX = 3
    BAND = 4
    BOR = 5
    BXOR = 6
    AVG = 7
    UNUSED = 8


def older_torch():
    '''
        Helper to lookup torch version. For versions less than 1.8, torch.dist
        used torch.distributed.group.WORLD as the default group argument instead of None.
        See more details at: https://github.com/pytorch/pytorch/pull/48767
    '''
    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])
    if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
        return True
    else:
        return False


def get_local_rank_from_launcher():

    # DeepSpeed launcher will set it so get from there
    rank = os.environ.get('LOCAL_RANK')

    if rank is None:
        rank = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK')

    # Make it a single process job and set rank to 0
    if rank is None:
        rank = 0

    return int(rank)


def get_world_rank_from_launcher():

    # DeepSpeed launcher will set it so get from there
    rank = os.environ.get('RANK')

    if rank is None:
        rank = os.environ.get('OMPI_COMM_WORLD_RANK')

    # Make it a single process job and set rank to 0
    if rank is None:
        rank = 0

    return int(rank)


def get_world_size_from_launcher():
    # DeepSpeed launcher will set it so get from there
    size = os.environ.get('WORLD_SIZE')

    if size is None:
        size = os.environ.get('OMPI_COMM_WORLD_SIZE')

    # Make it a single process job and set size to 1
    if size is None:
        size = 1

    print(f"set world size to {size}")
    return int(size)
