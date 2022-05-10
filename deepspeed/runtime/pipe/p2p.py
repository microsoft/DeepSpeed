'''
Copyright 2019 The Microsoft DeepSpeed Team
'''

import pickle
from . import p2p_util
from .p2p_util import Type
import torch
import torch.distributed as dist
import typing

# To query whether we have send/recv support
from packaging.version import Version
from deepspeed.git_version_info import torch_info

_groups = None
_grid = None
_async = []


def can_send_recv() -> bool:
    torch_version = Version(torch_info['version'])
    sendrecv_min = Version('1.8')
    return torch_version >= sendrecv_min


#initializes adjacent process groups
#run this only after torch.distributed.init_process_group() has been called
def init_process_groups(grid):
    global _groups, _grid
    _grid = grid

    assert _grid.pipe_parallel_size > 1, "There is no pipeline parallelism"

    if not can_send_recv():
        _groups = [dist.new_group(ranks=group) for group in _grid.p2p_groups]


def _is_valid_send_recv(src_stage, dest_stage):
    first_stage = 0
    last_stage = _grid.pipe_parallel_size - 1
    assert abs(src_stage-dest_stage) == 1 or \
        (src_stage == first_stage and dest_stage == last_stage) or \
        (src_stage == last_stage and dest_stage == first_stage), \
    "Functionality currently limited to send and receive between adjacent ranks only"


def send(tensor, dest_stage, async_op=False):
    global _groups
    src_stage = _grid.get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)

    dest_rank = _grid.stage_to_global(stage_id=dest_stage)
    if async_op:
        global _async
        op = dist.isend(tensor, dest_rank)
        _async.append(op)
    else:
        if can_send_recv():
            return dist.send(tensor, dest_rank)
        else:
            group = _get_send_recv_group(src_stage, dest_stage)
            src_rank = _grid.stage_to_global(stage_id=src_stage)
            return dist.broadcast(tensor, src_rank, group=group, async_op=async_op)


def recv(tensor, src_stage, async_op=False):
    global _groups
    dest_stage = _grid.get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)

    src_rank = _grid.stage_to_global(stage_id=src_stage)

    if async_op:
        global _async
        op = dist.irecv(tensor, src_rank)
        _async.append(op)
    else:
        if can_send_recv():
            return dist.recv(tensor, src_rank)
        else:
            group = _get_send_recv_group(src_stage, dest_stage)
            return dist.broadcast(tensor, src_rank, group=group, async_op=async_op)


def wait():
    global _async
    for op in _async:
        op.wait()
    _async = []

    torch.cuda.synchronize()


def new_send_obj(msg: typing.Any, dest: int, async_op=False):
    """send an python object to dest. the object could be tuples, lists,
       dictionaries and tensors."""

    # metadata: [msg_len, msg_type, max_dim]
    metadata = torch.empty((3), dtype=torch.long, device="cuda")
    metadata[0] = len(msg)
    tensors = []

    if isinstance(msg, tuple):
        metadata[1] = Type.TUPLE.value
        tensors = list(msg)
    elif isinstance(msg, list):
        metadata[1] = Type.LIST.value
        tensors = msg
    elif torch.is_tensor(msg):
        metadata[0] = 1
        metadata[1] = Type.TENSOR.value
        tensors[0] = msg
    else:
        raise Exception("Currently, send_obj only supports tuple, list and tensor type.")

    #assume each tensor has the same number of dims. This can be revisited later.
    max_dim = max([tensor.dim() for tensor in tensors])
    metadata[2] = max_dim

    stage_id = _grid.get_stage_id()
    if async_op:
        #send metadata
        promises = []
        promises.append(send(metadata, dest, True))
        #element_type_shape: [dtype, dim, shape, padding...]
        #if _grid.get_global_rank() % 8 == 0:
        #    ic(stage_id, metadata, "async send")

        for tensor in tensors:
            element_type_shape = torch.empty((max_dim + 2),
                                             dtype=torch.long,
                                             device="cuda")
            element_type_shape[0] = p2p_util.encode_element_type(tensor)
            element_type_shape[1] = tensor.dim()
            for i in range(tensor.dim()):
                element_type_shape[i + 2] = tensor.size(i)

            promises.append(send(element_type_shape, dest, True))
            promises.append(send(tensor, dest, True))
            #if _grid.get_global_rank() % 8 == 0:
            #    ic(stage_id, element_type_shape, tensor.shape, "async send")

        return promises
    else:
        dist.send(metadata, dest)
        for tensor in tensors:
            element_type_shape = torch.empty((max_dim + 2),
                                             dtype=torch.long,
                                             device="cuda")
            element_type_shape[0] = p2p_util.encode_element_type(tensor)
            element_type_shape[1] = tensor.dim()
            for i in range(tensor.dim()):
                element_type_shape[i + 2] = tensor.size(i)

            dist.send(element_type_shape, dest)
            dist.send(tensor, dest)


def new_recv_obj(sender: int, async_op=False) -> typing.Any:
    metadata = torch.empty((3), dtype=torch.long, device="cuda")
    msg = []

    stage_id = _grid.get_stage_id()

    if async_op:
        recv(metadata, sender, is_async=False)
        msg_len = metadata[0]
        msg_type = metadata[1]
        max_dim = metadata[2]
        for i in range(msg_len):
            element_type_shape = torch.empty((max_dim + 2),
                                             dtype=torch.long,
                                             device="cuda")
            recv(element_type_shape, sender, is_async=False)
            element_type = p2p_util.decode_element_type(element_type_shape[0].item())
            dim = element_type_shape[1].item()
            shape = element_type_shape[2:dim + 2].tolist()
            data = torch.empty(shape, dtype=element_type, device="cuda")
            recv(data, sender, is_async=False)
            msg.append(data)
        if msg_type == Type.TENSOR.value:
            return msg[0]
        elif msg_type == Type.TUPLE.value:
            return tuple(msg)
        elif msg_type == Type.LIST.value:
            return msg
        else:
            raise Exception('Message type is not supported:', msg_type)
    else:
        dist.recv(metadata, sender)
        msg_len = metadata[0]
        msg_type = metadata[1]
        max_dim = metadata[2]
        for i in range(msg_len):
            element_type_shape = torch.empty((max_dim + 2),
                                             dtype=torch.long,
                                             device="cuda")
            dist.recv(element_type_shape, sender)
            element_type = p2p_util.decode_element_type(element_type_shape[0].item())
            dim = element_type_shape[1].item()
            shape = element_type_shape[2:dim + 2].tolist()
            data = torch.empty(shape, dtype=element_type, device="cuda")
            dist.recv(data, sender)
            msg.append(data)
        if msg_type == Type.TENSOR.value:
            return msg[0]
        elif msg_type == Type.TUPLE.value:
            return tuple(msg)
        elif msg_type == Type.LIST.value:
            return msg
        else:
            raise Exception('Message type is not supported:', msg_type)


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


def _get_send_recv_group(src_stage, dest_stage):
    '''the group id is always the smaller rank unless its a wrap around'''

    stage_id = None

    first_stage = 0
    last_stage = _grid.pipe_parallel_size - 1

    if (src_stage == first_stage and dest_stage == last_stage
            or dest_stage == first_stage and src_stage == last_stage):
        stage_id = last_stage
    elif src_stage > dest_stage:
        stage_id = dest_stage
    else:
        stage_id = src_stage
    '''group_id corresponds to group of [group_id, group_id+1]
     unless group_id is the rank of the last stage
     in which case group_id corresponds to group[group_id-num_stages+1, group_id]
     '''
    group_id = _grid.stage_to_global(stage_id=stage_id)

    return _groups[group_id]
