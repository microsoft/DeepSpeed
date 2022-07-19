import torch
from benchmarks.communication.utils import *
from benchmarks.communication.constants import *

import time
import argparse
import os

import math


# Run allgather and print metrics
def timed_allgather(input, output, pg1, pg2, args):
    if args.dist == 'torch':
        import torch.distributed as dist
    elif args.dist == 'deepspeed':
        import deepspeed.comm as dist

    #comm_id = int(dist.get_rank()/8)
    comm_id = int(dist.get_rank()/(dist.get_world_size()/2))
    #comm_id = dist.get_rank()
    #print(f'comm_id: {comm_id}')

    sync_all()
    # Warmup, establish connections, etc.
    for i in range(args.warmup):
        if dist.get_rank() in pg1.ranks:
            group = pg1
        if dist.get_rank() in pg2.ranks:
            group = pg2
        #else:
        #    group = None
        #print(f'!!!BEFORE!!!RANK {dist.get_rank()} INPUT: {input} OUTPUT: {output}')
        # use all_gather_base if available
        if args.dist == 'torch':
            if hasattr(torch.distributed, "_all_gather_base"):
                dist._all_gather_base(output, input, group=None, async_op=args.async_op, comm_id=comm_id)
            else:
                output_tensors = list(
                    torch.chunk(output_tensor,
                                cdb.get_world_size(group)))
                dist.all_gather(output_tensors, input_tensor, group=group, async_op=True)
        elif args.dist == 'deepspeed':
            dist.allgather_fn(output, input, None, args.async_op, comm_id)
        sync_all()
        #print(f'!!!AFTER!!!RANK {dist.get_rank()} INPUT: {input} OUTPUT: {output}')
    sync_all()
    

    # time the actual comm op trials times and average it
    pre = time.perf_counter()
    for i in range(args.trials):
        if dist.get_rank() in pg1.ranks:
            group = pg1
        if dist.get_rank() in pg2.ranks:
            group = pg2
        print(f'!!!BEFORE!!!RANK {dist.get_rank()} INPUT: {input} OUTPUT: {output}')
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
            dist.allgather_fn(output, input, group, args.async_op)
        print(f'!!!AFTER!!!RANK {dist.get_rank()} INPUT: {input} OUTPUT: {output}')
        sync_all()
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
        f"{convert_size(size):<20} {desc:25s} {duration_str:20s} {tput_str:20s} {busbw_str:20s}"
    )


def run_allgather(local_rank, args):
    if args.dist == 'torch':
        import torch.distributed as dist
    elif args.dist == 'deepspeed':
        import deepspeed.comm as dist

    # Prepare benchmark header
    print_header(args, 'allgather')
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    group1 = [0,1,2,3,4,5,6,7]
    group2 = [8,9,10,11,12,13,14,15]

    #group1 = [0,1]
    #group2 = [2,3]

    #exit()

    #dist.create_comm_group(group1, global_rank, 0, 0)
    #dist.create_comm_group(group2, global_rank, 1, 1)

    #dist.test_set()
    #exit(0)

    pg1 = dist.new_group(group1)
    #print(pg1)
    #dist.barrier()
    pg2 = dist.new_group(group2)
    #print(pg2)



    #exit()

    if args.scan:
        # Create list of message sizes
        M_LIST = []
        for x in (2**p for p in range(1, args.maxsize)):
            M_LIST.append(x)

        sync_all()
        # loop over various tensor sizes
        for M in M_LIST:
            global_rank = dist.get_rank()
            try:
                mat = torch.ones(world_size,
                                 M,
                                 dtype=getattr(torch,
                                               args.dtype)).cuda(local_rank)
                sync_all()
                input = ((mat.mul_(float(global_rank))).view(-1))
                # Delete original mat to avoid OOM
                del mat
                torch.cuda.empty_cache()
                output = torch.zeros(input.nelement() * world_size,
                                     dtype=getattr(torch,
                                                   args.dtype)).cuda(local_rank)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    if dist.get_rank() == 0:
                        print('WARNING: Ran out of GPU memory. Exiting comm op.')
                    sync_all()
                    break
            sync_all()
            timed_allgather(input, output, pg1, pg2, args)
    else:
        # all_gather_base saves memory
        if (args.dist == 'torch'
                and hasattr(torch.distributed,
                            "_all_gather_base")) or (args.dist == 'deepspeed'
                                                     and dist.has_allgather_base):
            mem_factor = args.mem_factor + 0.2
        else:
            mem_factor = args.mem_factor
        # Send the biggest message size our GPUs can fit. If you're facing OOM errors, reduce the mem_factor
        sync_all()
        elements_per_gpu = max_numel(comm_op='allgather',
                                     dtype=getattr(torch,
                                                   args.dtype),
                                     mem_factor=mem_factor,
                                     local_rank=local_rank,
                                     args=args)
        try:
            mat = torch.ones(elements_per_gpu,
                             dtype=getattr(torch,
                                           args.dtype)).cuda(local_rank)
            # multiply each GPU's tensor by the rank to ease debugging
            input = ((mat.mul_(float(global_rank))).view(-1))
            # Delete original mat to avoid OOM
            del mat
            torch.cuda.empty_cache()
            output = torch.zeros(elements_per_gpu * world_size,
                                 dtype=getattr(torch,
                                               args.dtype)).cuda(local_rank)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                if dist.get_rank() == 0:
                    print(
                        'WARNING: Ran out of GPU memory. Try to reduce the --mem-factor argument!'
                    )
                sync_all()
                return

        sync_all()
        timed_allgather(input, output, pg1, pg2, args)


if __name__ == "__main__":
    args = benchmark_parser().parse_args()
    rank = args.local_rank
    init_processes(local_rank=rank, args=args)
    run_allgather(local_rank=rank, args=args)