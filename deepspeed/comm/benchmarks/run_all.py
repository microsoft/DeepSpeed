import torch
from utils import init_processes
from all_reduce import run_allreduce
from all_gather import run_allgather
from all_to_all import run_alltoall
from pt2pt import run_pt2pt
from constants import *

import time
import argparse
import os

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
    parser.add_argument(
        "--mem-factor",
        type=float,
        default=.4,
        help='Proportion of max available GPU memory to use for single-size evals')
    args = parser.parse_args()
    rank = args.local_rank

    init_processes(local_rank=rank, args=args)

    for comm_op in ['allreduce', 'allgather', 'alltoall', 'pt2pt']:
        if comm_op == 'allreduce':
            args.mem_factor = .8
            run_allreduce(local_rank=rank, args=args)
        if comm_op == 'allgather':
            run_allgather(local_rank=rank, args=args)
        if comm_op == 'alltoall':
            run_alltoall(local_rank=rank, args=args)
        if comm_op == 'pt2pt':
            args.mem_factor = .8
            run_pt2pt(local_rank=rank, args=args)
