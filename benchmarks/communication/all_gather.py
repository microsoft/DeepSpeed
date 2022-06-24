import torch
import utils
from utils import *
from constants import *

import time
import argparse
import os

import math


# Run allgather and print metrics
def timed_allgather(input, output, args):
    if args.dist == 'torch':
        import torch.distributed as dist
    elif args.dist == 'deepspeed':
        import deepspeed.comm as dist

    sync_all()
    # Warmup, establish connections, etc.
    for i in range(args.warmup):
        # use all_gather_base if available
        if args.dist == 'torch':
            if hasattr(torch.distributed, "_all_gather_base"):
                dist._all_gather_base(output, input, group=None, async_op=args.async_op)
            else:
                output_tensors = list(
                    torch.chunk(output_tensor,
                                cdb.get_world_size(group)))
                dist.all_gather(output_tensors, input_tensor, group=group, async_op=True)
        elif args.dist == 'deepspeed':
            dist.allgather_fn(output, input, group=None, async_op=args.async_op)
    sync_all()

    # time the actual comm op trials times and average it
    pre = time.perf_counter()
    for i in range(args.trials):
        # use all_gather_base if available
        if args.dist == 'torch':
            if hasattr(torch.distributed, "_all_gather_base"):
                dist._all_gather_base(output, input, group=None, async_op=args.async_op)
            else:
                output_tensors = list(
                    torch.chunk(output_tensor,
                                cdb.get_world_size(group)))
                dist.all_gather(output_tensors, input_tensor, group=group, async_op=True)
        elif args.dist == 'deepspeed':
            dist.allgather_fn(output, input, group=None, async_op=args.async_op)
    sync_all()
    duration = time.perf_counter() - pre

    # maintain and clean performance data
    avg_duration = duration / args.trials
    size = input.element_size() * input.nelement()
    n = dist.get_world_size()
    tput, busbw = get_bw('allgather', size, avg_duration, args)
    tput_str, busbw_str, duration_str = get_metric_strings(args, tput, busbw, avg_duration)
    desc = f'{input.nelement()}x{input.element_size()}'

    print_rank_0(
        f"{size:<20} {desc:25s} {duration_str:20s} {tput_str:20s} {busbw_str:20s}")


def run_allgather(local_rank, args):
    if args.dist == 'torch':
        import torch.distributed as dist
    elif args.dist == 'deepspeed':
        import deepspeed.comm as dist

    # Prepare benchmark header
    print_header(args, 'allgather')
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    if args.scan:
        # Create list of message sizes
        M_LIST = []
        for x in (2**p for p in range(1, args.maxsize)):
            M_LIST.append(x)

        sync_all()
        # loop over various tensor sizes
        for M in M_LIST:
            global_rank = dist.get_rank()
            mat = torch.ones(world_size, M, dtype=args.dtype).cuda(local_rank)
            sync_all()
            input = ((mat.mul_(float(global_rank))).view(-1))
            # Delete original mat to avoid OOM
            del mat
            torch.cuda.empty_cache()
            output = torch.zeros(input.nelement() * world_size,
                                 dtype=args.dtype).cuda(local_rank)
            sync_all()
            timed_allgather(input, output, args)
    else:
        # all_gather_base saves memory
        if (args.dist == 'torch'
                and hasattr(torch.distributed,
                            "_all_gather_base")) or (args.dist == 'deepspeed'
                                                     and dist.has_allgather_base):
            mem_factor = 0.6
        else:
            mem_factor = 0.4
        # Send the biggest message size our GPUs can fit. If you're facing OOM errors, reduce the mem_factor
        elements_per_gpu = max_numel(comm_op='allgather',
                                     dtype=args.dtype,
                                     mem_factor=mem_factor,
                                     local_rank=local_rank,
                                     args=args)
        mat = torch.ones(elements_per_gpu, dtype=args.dtype).cuda(local_rank)
        # multiply each GPU's tensor by the rank to ease debugging
        input = ((mat.mul_(float(global_rank))).view(-1))
        # Delete original mat to avoid OOM
        del mat
        torch.cuda.empty_cache()
        output = torch.zeros(elements_per_gpu * world_size,
                             dtype=args.dtype).cuda(local_rank)

        sync_all()
        timed_allgather(input, output, args)


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
                        default=DEFAULT_MAXSIZE,
                        help='Max message size as a power of 2')
    parser.add_argument("--async-op",
                        action="store_true",
                        help='Enables non-blocking communication')
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
    run_allgather(local_rank=rank, args=args)
