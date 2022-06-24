import torch
import os
import math

global dist


def init_torch_distributed(backend):
    global dist
    import torch.distributed as dist
    torch.distributed.init_process_group(backend)
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)


def init_deepspeed_comm(backend):
    global dist
    import deepspeed
    import deepspeed.comm as dist
    deepspeed.init_distributed(dist_backend=backend)
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)


def init_processes(local_rank, args):
    if args.dist == 'deepspeed':
        init_deepspeed_comm(args.backend)
    elif args.dist == 'torch':
        init_torch_distributed(args.backend)
    else:
        print_rank_0(f"distributed framework {args.dist} not supported")
        exit(0)


def print_rank_0(message):
    if dist.get_rank() == 0:
        print(message)


def print_header(args, comm_op):
    if comm_op == 'pt2pt':
        world_size = 2
    else:
        world_size = dist.get_world_size()
    tput = f'Throughput ({args.bw_unit})'
    busbw = f'BusBW ({args.bw_unit})'
    header = f"\n---- Performance of {comm_op} on {world_size} devices ---------------------------------------------------------\n"
    header += f"{'Size (Bytes)':20s} {'Description':25s} {'Duration':20s} {tput:20s} {busbw:20s}\n"
    header += "----------------------------------------------------------------------------------------------------"
    print_rank_0(header)


def get_bw(comm_op, size, duration, args):
    n = dist.get_world_size()
    tput = 0
    busbw = 0
    if comm_op == "alltoall":
        tput = (size / duration)
        busbw = (size / duration) * ((n - 1) / n)
    elif comm_op == "allgather":
        size *= n
        tput = (size / duration)
        busbw = (size / duration) * ((n - 1) / n)
    elif comm_op == "allreduce":
        tput = (size * 2 / duration)
        busbw = (size / duration) * (2 * (n - 1) / n)
    elif comm_op == "pt2pt":
        tput = (size / duration)
        busbw = tput
    else:
        print_rank_0("wrong comm_op specified")
        exit(0)

    if args.bw_unit == 'Gbps':
        tput *= 8
        busbw *= 8

    return tput, busbw


def get_metric_strings(args, tput, busbw, duration):
    duration_ms = duration * 1e3
    duration_us = duration * 1e6
    if args.bw_unit == 'Gbps':
        tput = f'{tput / 1e9:.3f}'
        busbw = f'{busbw /1e9:.3f}'
    elif args.bw_unit == 'GBps':
        tput = f'{tput/8 / 1e9:.3f}'
        busbw = f'{busbw/8 /1e9:.3f}'

    if duration_us < 1e3:
        duration = f'{duration_us:.3f} us'
    else:
        duration = f'{duration_ms:.3f} ms'
    return tput, busbw, duration


def sync_all():
    torch.cuda.synchronize()
    dist.barrier()


def max_numel(comm_op, dtype, mem_factor, local_rank, args):
    dtype_size = torch._utils._element_size(args.dtype)
    max_memory_per_gpu = torch.cuda.get_device_properties(
        local_rank).total_memory * mem_factor
    if comm_op == 'allreduce' or comm_op == 'pt2pt':
        elements_per_gpu = int(max_memory_per_gpu // dtype_size)
    elif comm_op == 'allgather':
        # all_gather performance is lower for non-powers of two, and the output buffer size scales with world size
        # Therefore, divide by world size and round down to nearest power of 2
        elements_per_gpu = int(max_memory_per_gpu // dtype_size // dist.get_world_size())
        elements_per_gpu = int(pow(2, int(math.log(elements_per_gpu, 2))))
    elif comm_op == 'alltoall':
        # Number of elements must be divisible by world_size
        # all_to_all performance is lower for non-powers of two. Round down like allgather.
        elements_per_gpu = int(max_memory_per_gpu // dtype_size)
        elements_per_gpu = int(dist.get_world_size() *
                               round(elements_per_gpu / dist.get_world_size()))
        elements_per_gpu = int(pow(2, int(math.log(elements_per_gpu, 2))))
    else:
        print(f"This communication operation: {comm_op} is not supported yet")
        exit(0)
    return elements_per_gpu
