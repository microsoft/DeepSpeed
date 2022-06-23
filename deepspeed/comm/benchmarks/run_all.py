import torch
from utils import init_processes
from all_reduce_new import run_allreduce_scan
from all_gather_new import run_allgather_scan
from all_to_all_new import run_alltoall_scan
from p2p_new import run_pt2pt_scan

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

    for comm_op in ['allreduce', 'allgather', 'alltoall', 'pt2pt']:
        if comm_op == 'allreduce':
            run_allreduce_scan(local_rank=rank, args=args)
        if comm_op == 'allgather':
            run_allgather_scan(local_rank=rank, args=args)
        if comm_op == 'alltoall':
            run_allgather_scan(local_rank=rank, args=args)
        if comm_op == 'pt2pt':
            run_pt2pt_scan(local_rank=rank, args=args)
        #run_comm_op(collective, args)
