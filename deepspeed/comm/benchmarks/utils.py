import torch
import os

global dist


def init_torch_distributed(backend):
    global dist
    import torch.distributed as dist
    import deepspeed
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


def print_header(args, collective):
    tput = f'Throughput ({args.bw_unit})'
    busbw = f'BusBW ({args.bw_unit})'
    header = f"\n---- Performance of {collective} on {dist.get_world_size()} devices ---------------------------------------------------------\n"
    header += f"{'Size (Bytes)':20s} {'Description':25s} {'Duration':20s} {tput:20s} {busbw:20s}\n"
    header += "----------------------------------------------------------------------------------------------------"
    print_rank_0(header)


def get_bw(collective, size, duration, args):
    n = dist.get_world_size()
    tput = 0
    busbw = 0
    if collective == "alltoall":
        tput = (size / duration) * 8
        busbw = (size / duration) * ((n - 1) / n) * 8
    elif collective == "allgather":
        size *= n
        tput = (size / duration) * 8
        busbw = (size / duration) * ((n - 1) / n) * 8
    elif collective == "allreduce":
        tput = (size * 2 / duration) * 8
        busbw = (size / duration) * (2 * (n - 1) / n) * 8
    else:
        print_rank_0("wrong collective specified")
        exit(0)

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
    dist.barrier
