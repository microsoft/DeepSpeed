import torch
import utils
import deepspeed
from utils import *
from constants import *

import time
import argparse
import os


# Run allgather and print metrics
def timed_allgather(input, output, args):
    if args.dist == 'torch':
        import torch.distributed as dist
    elif args.dist == 'deepspeed':
        import deepspeed.comm as dist

    sync_all()
    # Warmup, establish connections, etc.
    for i in range(args.warmup):
        dist.allgather_fn(output, input, group=None, async_op=args.async_op)
    sync_all()

    # time the actual collective trials times and average it
    pre = time.perf_counter()
    for i in range(args.trials):
        dist.allgather_fn(output, input, group=None, async_op=args.async_op)
    sync_all()
    duration = time.perf_counter() - pre

    # maintain and clean performance data
    avg_duration = duration / args.trials
    size = input.element_size() * input.nelement()
    n = dist.get_world_size()
    tput, busbw = get_bw('allgather', size, avg_duration, n)
    tput_str, busbw_str, duration_str = get_metric_strings(args, tput, busbw, avg_duration)
    desc = f'{input.nelement()}x{input.element_size()}'

    print_rank_0(
        f"{size:<20} {desc:25s} {duration_str:20s} {tput_str:20s} {busbw_str:20s}")


def run_allgather_scan(local_rank, args):
    if args.dist == 'torch':
        import torch.distributed as dist
    elif args.dist == 'deepspeed':
        import deepspeed.comm as dist
    world_size = dist.get_world_size()
    # Create list of message sizes
    M_LIST = []
    for x in (2**p for p in range(1, args.maxsize)):
        M_LIST.append(x)

    sync_all()
    # Prepare benchmark header
    print_header(args, 'allgather')
    # loop over various tensor sizes
    for M in M_LIST:
        global_rank = dist.get_rank()
        mat = torch.ones(world_size, M, dtype=torch.float32).cuda(local_rank)
        sync_all()
        input = ((mat.mul_(float(global_rank))).view(-1))
        output = torch.cat([(mat.clone().view(-1))
                            for _ in range(dist.get_world_size())])
        sync_all()
        timed_allgather(input, output, args)

        del mat
        torch.cuda.empty_cache()
        sync_all()


def run_allgather_single(local_rank, args):
    if args.dist == 'torch':
        import torch.distributed as dist
    elif args.dist == 'deepspeed':
        import deepspeed.comm as dist

    print_header(args, 'allgather')
    global_rank = dist.get_rank()
    #ALLGATHER_M =
    #ALLGATHER_N =
    mat = torch.ones(ALLGATHER_N, ALLGATHER_M, dtype=DEFAULT_TYPE).cuda(local_rank)
    input = ((mat.mul_(float(global_rank))).view(-1))
    output = torch.cat([(mat.clone().view(-1)) for _ in range(dist.get_world_size())])

    sync_all()
    timed_allgather(input, output, args)

    del mat, input, output
    torch.cuda.empty_cache()
    sync_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--maxsize", type=int, default=24)
    parser.add_argument("--async-op", action="store_true")
    parser.add_argument("--bw-unit", type=str, default='Gbps')
    parser.add_argument("--backend", type=str, default='nccl')
    parser.add_argument("--dist", type=str, default='deepspeed')
    parser.add_argument("--scan", action="store_true")
    args = parser.parse_args()
    rank = args.local_rank
    init_processes(local_rank=rank, args=args)
    if args.dist == 'torch':
        import torch.distributed as dist
    elif args.dist == 'deepspeed':
        import deepspeed.comm as dist
    if args.scan:
        run_allgather_scan(local_rank=rank, args=args)
    else:
        run_allgather_single(local_rank=rank, args=args)
