"""
Test command:
python3 -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 <this-file>
"""

import argparse

import torch
import math
from torch._C import device
from torch.autograd.grad_mode import F
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group, _pg_names
from deepspeed.ops.op_builder import CommunicationBuilder
import time
import numpy as np

ds_coll_comm = CommunicationBuilder().load()


def sizeof_dtype(dtype):
    if dtype == torch.half:
        return 2
    elif dtype == torch.float:
        return 4
    else:
        return None


def prepare_tensor(partition_sizes, world_size, device, dtype=torch.half):
    # transformer layer structure with partitioned onto 8 GPUs

    output_tensors = []
    input_tensors = []

    for _size in partition_sizes:
        std = 1 / math.sqrt(_size)
        input_t = torch.empty(_size,
                              dtype=dtype,
                              device=device).view(-1).uniform_(-std,
                                                               std)
        output_t = torch.empty(_size * world_size,
                               dtype=dtype,
                               device=device).view(-1).uniform_(-std,
                                                                std)
        input_tensors.append(input_t)
        output_tensors.append(output_t)
    return output_tensors, input_tensors


def _torch_allgather_once(output_tensors,
                          input_tensors,
                          partition_sizes,
                          rank,
                          world_size):
    """"""
    s = torch.cuda.Stream()
    handles = []
    for part_idx, part_size in enumerate(partition_sizes):
        output_t = output_tensors[part_idx]
        input_t = input_tensors[part_idx]

        output_list = []
        for i in range(world_size):
            out_tensor = output_t.narrow(0, i * part_size, part_size)
            output_list.append(out_tensor)

        h = dist.all_gather(output_list, input_t, async_op=True)
        handles.append(h)

    torch.cuda.synchronize()


def print_bw_rank0(partition_sizes, time_costs, dtype):
    if dist.get_rank() == 0:
        elem_size = sizeof_dtype(dtype)  # in bytes
        assert elem_size != None

        numel = sum(partition_sizes)

        avg_t = np.mean(time_costs)
        bw = numel * elem_size * (dist.get_world_size() - 1) / 1e9 / avg_t
        print(f'avg time {avg_t * 1e3} ms, bw {bw} GB/s')


def bench_torch_allgather(output_tensors,
                          input_tensors,
                          partition_sizes,
                          rank,
                          world_size,
                          warm_up=5,
                          repeat=10):
    ts = []
    for i in range(warm_up + repeat):
        s = time.time()
        _torch_allgather_once(output_tensors,
                              input_tensors,
                              partition_sizes,
                              rank,
                              world_size)
        e = time.time()

        if i >= warm_up:
            ts.append(e - s)

    print_bw_rank0(partition_sizes, ts, input_tensors[0].dtype)


def _custom_allgather_once(output_tensors, input_tensors, comm_stream):
    default_pg = _get_default_group()
    pg_name = _pg_names[default_pg]

    with torch.cuda.stream(comm_stream):
        res = ds_coll_comm._inplace_allgather(output_tensors,
                                              input_tensors,
                                              default_pg,
                                              pg_name)
    comm_stream.synchronize()
    return res


def bench_custom_allgather(output_tensors,
                           input_tensors,
                           partition_sizes,
                           rank,
                           world_size,
                           warm_up=5,
                           repeat=10):
    """"""
    comm_stream = torch.cuda.Stream(rank % torch.cuda.device_count())

    ts = []
    for i in range(warm_up + repeat):
        s = time.time()
        _custom_allgather_once(output_tensors, input_tensors, comm_stream)
        e = time.time()

        if i >= warm_up:
            ts.append(e - s)

    print_bw_rank0(partition_sizes, ts, input_tensors[0].dtype)


def print_rank0(msg):
    if (dist.get_rank() == 0):
        print(msg)


def main():
    """"""
    dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    local_size = torch.cuda.device_count()
    device_id = rank % local_size
    world_size = dist.get_world_size()
    torch.cuda.set_device(device_id)

    partition_sizes = [
        2457600,
        960,
        819200,
        320,
        320,
        320,
        3276800,
        1280,
        3276800,
        320,
        320,
        320
    ]
    output_tensors, input_tensors = prepare_tensor(partition_sizes,
                         dist.get_world_size(), f'cuda:{device_id}',
                            torch.half)
    print_rank0('Using torch.distributed.allgather')
    bench_torch_allgather(output_tensors,
                          input_tensors,
                          partition_sizes,
                          rank,
                          world_size)

    combined_tensor_torch = torch.cat(output_tensors)
    out_sum_torch = combined_tensor_torch.sum()
    print_rank0(f'output tensors sum {out_sum_torch}')

    # clean output tensors
    for t in output_tensors:
        t.zero_()
    print_rank0('Using customized allgather')
    bench_custom_allgather(output_tensors,
                           input_tensors,
                           partition_sizes,
                           rank,
                           world_size)
    combined_tensor_custom = torch.cat(output_tensors)
    out_sum_custom = combined_tensor_custom.sum()
    print_rank0(f'output tensor sum {out_sum_custom}')

    print_rank0(
        f'allgather results are close {combined_tensor_custom.allclose(combined_tensor_torch)}'
    )


if __name__ == '__main__':
    main()
