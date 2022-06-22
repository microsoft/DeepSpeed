import torch
import utils
import deepspeed
from utils import *
from constants import *

import time
import argparse
import os
import math


def timed_allreduce(input, args):
    if args.dist == 'torch':
        import torch.distributed as dist
    elif args.dist == 'deepspeed':
        import deepspeed.comm as dist

    sync_all()
    # Warmup, establish connections, etc.
    for i in range(args.warmup):
        dist.all_reduce(input, async_op=args.async_op)
    sync_all()

    # time the actual collective trials times and average it
    pre = time.perf_counter()
    for i in range(args.trials):
        dist.all_reduce(input, async_op=args.async_op)
    sync_all()
    duration = time.perf_counter() - pre

    # maintain and clean performance data
    avg_duration = duration / args.trials
    size = input.element_size() * input.nelement()
    n = dist.get_world_size()
    tput, busbw = get_bw('allreduce', size, avg_duration, n)
    tput_str, busbw_str, duration_str = get_metric_strings(args, tput, busbw, avg_duration)
    desc = f'{input.nelement()}x{input.element_size()}'

    print_rank_0(
        f"{size:<20} {desc:25s} {duration_str:20s} {tput_str:20s} {busbw_str:20s}")


def run_allreduce_scan(local_rank, args):
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
    print_header(args, 'allreduce')
    # loop over various tensor sizes
    for M in M_LIST:
        global_rank = dist.get_rank()
        mat = torch.ones(world_size, M, dtype=args.dtype).cuda(local_rank)
        sync_all()
        input = ((mat.mul_(float(global_rank))).view(-1))
        sync_all()
        timed_allreduce(input, args)


def run_allreduce_single(local_rank, args):
    if args.dist == 'torch':
        import torch.distributed as dist
    elif args.dist == 'deepspeed':
        import deepspeed.comm as dist

    print_header(args, 'allreduce')
    global_rank = dist.get_rank()
    # Send the biggest message size our GPUs can fit. If you're facing OOM errors, reduce the mem_factor
    elements_per_gpu = max_numel(collective='allreduce',
                                 dtype=args.dtype,
                                 mem_factor=.8,
                                 local_rank=local_rank,
                                 args=args)
    mat = torch.ones(elements_per_gpu, dtype=args.dtype).cuda(local_rank)
    input = ((mat.mul_(float(global_rank))).view(-1))
    sync_all()
    timed_allreduce(input, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--trials",
                        type=int,
                        default=DEFAULT_TRIALS,
                        help='Number of timed iterations')
    parser.add_argument("--warmup",
                        type=int,
                        default=DEFAULT_WARMUPS,
                        help='Number of warmup (non-timed) iterations')
    parser.add_argument("--maxsize",
                        type=int,
                        default=24,
                        help='Max message size as a power of 2')
    parser.add_argument("--async-op",
                        action="store_true",
                        help='Enables non-blocking collectives')
    parser.add_argument("--bw-unit",
                        type=str,
                        default=DEFAULT_UNIT,
                        choices=['Gbps',
                                 'GBps'])
    parser.add_argument("--backend",
                        type=str,
                        default=DEFAULT_BACKEND,
                        choices=['nccl'],
                        help='Communication library to use')
    parser.add_argument("--dist",
                        type=str,
                        default=DEFAULT_DIST,
                        choices=['deepspeed',
                                 'torch'],
                        help='Distributed DL framework to use')
    parser.add_argument("--scan",
                        action="store_true",
                        help='Enables scanning all message sizes')
    parser.add_argument("--dtype",
                        type=str,
                        default=DEFAULT_TYPE,
                        help='PyTorch tensor dtype')
    args = parser.parse_args()
    rank = args.local_rank
    init_processes(local_rank=rank, args=args)
    if args.dist == 'torch':
        import torch.distributed as dist
    elif args.dist == 'deepspeed':
        import deepspeed.comm as dist
    if args.scan:
        run_allreduce_scan(local_rank=rank, args=args)
    else:
        run_allreduce_single(local_rank=rank, args=args)
