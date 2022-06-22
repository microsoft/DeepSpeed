import torch
from utils import init_processes
from all_reduce_new import run_allreduce_scan

import time
import argparse
import os

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
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    rank = args.local_rank

    init_processes(local_rank=rank, args=args)

    for collective in ['allreduce', 'allgather', 'alltoall']:
        if collective == 'allreduce':
            run_allreduce_scan(local_rank=rank, args=args)
        if collective == 'allgather':
            run_allgather_scan(local_rank=rank, args=args)
        #run_collective(collective, args)
