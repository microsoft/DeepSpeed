'''
Copyright 2019 The Microsoft DeepSpeed Team
'''

import pickle
import typing

import torch
import torch.distributed as dist

_groups = None
_grid = None


#initializes adjacent process groups
#run this only after torch.distributed.init_process_group() has been called
def init_process_groups(grid):
    global _groups, _grid
    _grid = grid

    assert _grid.pipe_parallel_size > 1, "There is no pipeline parallelism"


def _is_valid_send_recv(src_stage, dest_stage):
    first_stage = 0
    last_stage = _grid.pipe_parallel_size - 1
    assert abs(src_stage-dest_stage) == 1 or \
        (src_stage == first_stage and dest_stage == last_stage) or \
        (src_stage == last_stage and dest_stage == first_stage), \
    "Functionality currently limited to send and receive between adjacent ranks only"


def send(tensor, dest_stage, async_op=False):
    global _groups
    assert async_op == False, "Doesnt support async_op true"
    src_stage = _grid.get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)

    dest_rank = _grid.stage_to_global(stage_id=dest_stage)
    return dist.send(tensor, dest_rank)


def recv(tensor, src_stage, async_op=False):
    global _groups
    assert async_op == False, "Doesnt support async_op true"
    dest_stage = _grid.get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)

    src_rank = _grid.stage_to_global(stage_id=src_stage)
    return dist.recv(tensor, src_rank)


def send_obj(msg: typing.Any, dest: int):
    """Send an arbitrary python object to ``dest``.

    Note: ``msg`` must be pickleable.

    WARN: This incurs a CPU -> GPU transfer and should be used sparingly
    for performance reasons.

    Args:
        msg (typing.Any): The object to send.
        dest (int): Destination rank.
    """
    # serialize the message
    msg = pickle.dumps(msg)
    # construct a tensor to send
    msg = torch.ByteTensor(torch.ByteStorage.from_buffer(msg)).cuda()

    # Send meta and message
    length_tensor = torch.tensor([len(msg)], dtype=torch.long).cuda()
    dist.send(length_tensor, dst=dest)
    dist.send(msg, dst=dest)


def recv_obj(sender: int) -> typing.Any:
    """Receive an arbitrary python object from ``sender``.

    WARN: This incur a CPU <-> GPU transfers and should be used sparingly
    for performance reasons.

    Args:
        sender (int): The rank sending the message.
    """
    # Get message meta
    length = torch.tensor([0], dtype=torch.long).cuda()
    dist.recv(length, src=sender)

    # Receive and deserialize
    msg = torch.empty(length.item(), dtype=torch.uint8).cuda()
    dist.recv(msg, src=sender)

    msg = pickle.loads(msg.cpu().numpy().tobytes())

    def _to(x):
        """Recursively move to the current device."""
        if torch.is_tensor(x):
            return x.cuda()
        if isinstance(x, (tuple, list)):
            ret = [_to(x_) for x_ in x]
            if isinstance(x, tuple):
                ret = tuple(ret)
            return ret
        # handle kwargs
        if isinstance(x, dict):
            ret = dict()
            for key, val in x.items():
                ret[_to(key)] = _to(val)
            return ret

        # Anything else is a no-op
        return x

    msg = _to(msg)
    return msg
