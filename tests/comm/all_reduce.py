import torch
import deepspeed.comm as dist
import time
import argparse
import os
import deepspeed

TRIALS = 50

N = 50000
M = 4000


def timed_allreduce(mat):
    torch.cuda.synchronize()
    pre = time.perf_counter()
    dist.all_reduce(mat)
    #print('ignore me', mat[0][0])  # required due to lazy evaluation
    torch.cuda.synchronize()
    duration = time.perf_counter() - pre
    #print("duration: %f sec" % duration)
    tput = ((M * N * 4 * 2) / duration) * 8
    #print("algo throughput: %f bps, %f Gbps" % (tput, tput/1e9))
    size = M * N * 4
    n = dist.get_world_size()
    busbw = (size / duration) * (2 * (n - 1) / n) * 8
    #print("busbw: %f Gbps" % (busbw / 1e9))
    return tput, busbw


def run(local_rank):
    global_rank = dist.get_rank()
    if global_rank == 0:
        print(global_rank, "data size:", M * N * 4 / 1e9, "GB")
    mat = torch.rand(N, M, dtype=torch.float32).cuda(local_rank)

    tputs = []
    busbws = []
    for trial in range(TRIALS):
        tput, busbw = timed_allreduce(mat)
        if trial > 2:
            tputs.append(tput)
            busbws.append(busbw)

    local_avg = sum(tputs) / len(tputs)
    local_avg_bb = sum(busbws) / len(busbws)
    t = torch.tensor([local_avg / 1e9, local_avg_bb / 1e9], device='cuda')
    dist.all_reduce(t)
    tput_avg = t[0] / dist.get_world_size()
    busbw_avg = t[1] / dist.get_world_size()
    if dist.get_rank() == 0:
        print('tput_avg (Gbps):', tput_avg.item(), 'busbw_avg (Gbps):', busbw_avg.item())


def init_processes(fn, backend='nccl', use_deepspeed=False):
    #dist.init_process_group(backend=backend, use_deepspeed=use_deepspeed)
    dist.init_distributed(backend)
    print(f'local rank = {dist.get_local_rank()}')
    torch.cuda.set_device(dist.get_local_rank())
    fn(dist.get_local_rank())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--use_deepspeed", action="store_true")
    args = parser.parse_args()
    rank = args.local_rank
    use_deepspeed = args.use_deepspeed
    #print("local_rank: %d" % rank)
    init_processes(fn=run, use_deepspeed=use_deepspeed)
