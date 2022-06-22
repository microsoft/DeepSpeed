import torch
#from deepspeed.ops.comm import initialize_nccl

import time
import argparse
import os

# dist is global and functions can set them to switch between deepspeed and torch
dist = None

DEBUG = False


def print_rank_0(message):
    if dist.get_rank() == 0:
        print(message)


def collective_fn(collective, input, output, async_op):
    if collective == "alltoall":
        dist.all_to_all_single(output, input, async_op=async_op)
    elif collective == "allreduce":
        dist.all_reduce(input, async_op=async_op)
    elif collective == "allgather":
        dist.allgather_fn(output, input, group=None, async_op=async_op)
        #pass
    else:
        print_rank_0(f"collective {collective} not supported yet")
        exit(0)


def get_bw(collective, size, duration, n):
    tput = 0
    busbw = 0
    if collective == "alltoall" or collective == "allgather":
        tput = (size / duration) * 8
        busbw = (size / duration) * ((n - 1) / n) * 8
    elif collective == "allreduce":
        tput = (size * 2 / duration) * 8
        busbw = (size / duration) * (2 * (n - 1) / n) * 8
    else:
        print_rank_0("wrong collective specified")
        exit(0)
    return tput, busbw


def timed_benchmark(input, output, args, collective):
    dist.barrier()
    torch.cuda.synchronize()

    # Warmup, establish connections, etc.
    for i in range(args.warmup):
        collective_fn(collective, input, output, async_op=args.async_op)

    dist.barrier()
    torch.cuda.synchronize()

    # time the actual collective trials times and average it
    pre = time.perf_counter()
    for i in range(args.trials):
        collective_fn(collective, input, output, async_op=args.async_op)
    torch.cuda.synchronize()
    duration = time.perf_counter() - pre

    # maintain and clean performance data
    duration = duration / args.trials
    size = int(input.shape[0]) * 4
    n = dist.get_world_size()
    tput, busbw = get_bw(collective, size, duration, n)

    duration_ms = duration * 1e3
    duration_us = duration * 1e6

    desc = f'{input.shape[0]}x{4}'

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

    print_rank_0(f"{size:<20} {desc:25s} {duration:20s} {tput:20s} {busbw:20s}")


def test_correctness(input, output, args, collective):
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()

    for i in range(world_size):
        if i == global_rank:
            print(f"Before AllToAll Input List at rank {global_rank}: {input}")
        dist.barrier()

    collective_fn(collective, input, output, async_op=args.async_op)

    torch.cuda.synchronize()
    dist.barrier()

    for i in range(world_size):
        if i == global_rank:
            print(f"AllToAll Results at rank {global_rank}: {output}")
        dist.barrier()


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
    import os
    rank = os.environ.get('RANK')
    size = os.environ.get('WORLD_SIZE')
    ranks = [i for i in range(int(size))]
    import deepspeed.comm as dist
    deepspeed.init_distributed(dist_backend=backend)
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)


def init_processes(local_rank, args):
    if args.backend == 'deepspeed':
        # TODO. Put the right name in args.
        init_deepspeed_comm('nccl')
    elif args.backend == 'nccl':
        init_torch_distributed(args.backend)
    else:
        print(f"backend {args.backend} not supported")
        exit(0)

    N = dist.get_world_size()

    M_LIST = []
    for x in (2**p for p in range(1, args.maxsize)):
        M_LIST.append(x)

    # List of benchmarks
    collectives = ['allgather', 'allreduce', 'alltoall']

    # Run all collectives
    for collective in collectives:
        world_size = dist.get_world_size()
        dist.barrier()

        # Prepare benchmark header
        tput = f'Throughput ({args.bw_unit})'
        busbw = f'BusBW ({args.bw_unit})'

        header = f"\n---- Performance of {collective} on {dist.get_world_size()} devices ---------------------------------------------------------\n"
        header += f"{'Size (Bytes)':20s} {'Description':25s} {'Duration':20s} {tput:20s} {busbw:20s}\n"
        header += "----------------------------------------------------------------------------------------------------"

        print_rank_0(header)

        # loop over various tensor sizes for each collective
        for M in M_LIST:
            global_rank = dist.get_rank()
            mat = torch.ones(N, M, dtype=torch.float32).cuda(local_rank)
            torch.cuda.synchronize()

            if collective == 'alltoall':
                # check needed for alltoall only
                assert mat.numel() % world_size == 0, f"tensor cannot be divided in {world_size} chunks"

            input = ((mat.mul_(float(global_rank))).view(-1))
            if collective == 'allgather':
                output = torch.cat([(mat.clone().view(-1))
                                    for _ in range(dist.get_world_size())])
            else:
                output = (mat.clone().view(-1))
            torch.cuda.synchronize()
            dist.barrier()
            timed_benchmark(input, output, args, collective)

            global DEBUG
            if DEBUG:
                test_correctness(input, output, args, collective)
        del output, mat
        torch.cuda.empty_cache()

        dist.barrier()
        print_rank_0("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--maxsize", type=int, default=24)
    parser.add_argument("--async-op", action="store_true")
    parser.add_argument("--bw-unit", type=str, default='Gbps')
    parser.add_argument("--backend", type=str, default='nccl')
    args = parser.parse_args()
    rank = args.local_rank
    init_processes(local_rank=rank, args=args)
