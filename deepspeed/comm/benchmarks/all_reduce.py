import torch
import deepspeed.comm as dist
import time
import argparse
import os
import deepspeed

DEFAULT_WARMUPS = 5
DEFAULT_TRIALS = 50
DEFAULT_MIN = 1
DEFAULT_MAX = 134217728
DEFAULT_TYPE = torch.float32
DEFAULT_STEP = 2
DEFAULT_BACKEND = 'nccl'

N = 50000
M = 4000


# Helper function to pretty-print message sizes
def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def calc_bw(msg_size, lat):
    n = dist.get_world_size()
    algbw = ((msg_size * 8 * 2) / lat) / 1e6
    busbw = algbw * ((n - 1) / n)
    return algbw, busbw


def timed_allreduce(mat):
    torch.cuda.synchronize()
    pre = time.perf_counter()
    dist.all_reduce(mat)
    torch.cuda.synchronize()
    duration = time.perf_counter() - pre
    return calc_bw(mat.element_size() * mat.nelement(), duration)


def run_bench(local_rank,
              backend=DEFAULT_BACKEND,
              trials=DEFAULT_TRIALS,
              warmups=DEFAULT_WARMUPS,
              min_size=DEFAULT_MIN,
              max_size=DEFAULT_MAX,
              dtype=DEFAULT_TYPE,
              step=DEFAULT_STEP):
    #
    #global_rank = dist.get_rank()
    #if global_rank == 0:
    #    print(global_rank, "data size:", M * N * 4 / 1e9, "GB")
    #mat = torch.rand(N, M, dtype=torch.float32).cuda(local_rank)

    dist.init_distributed(backend)
    print(f'local rank = {dist.get_local_rank()}')
    torch.cuda.set_device(dist.get_local_rank())
    #dist.start_profiling_comms()
    #dist.configure(enabled=True)

    #for trial in range(trials + warmups):
    for size in (step**p for p in range(int(min_size / step), int(max_size / step))):
        tputs = []
        busbws = []
        t = torch.rand(size, dtype=dtype).cuda(local_rank)
        for trial in range(trials):
            tput, busbw = timed_allreduce(t, msg_range)
            if trial > warmups:
                tputs.append(tput)
                busbws.append(busbw)
            #dist.log_summary(['all'])

        local_avg = sum(tputs) / len(tputs)
        local_avg_bb = sum(busbws) / len(busbws)
        t = torch.tensor([local_avg / 1e9, local_avg_bb / 1e9], device='cuda')
        dist.all_reduce(t)
        tput_avg = t[0] / dist.get_world_size()
        busbw_avg = t[1] / dist.get_world_size()
        if dist.get_rank() == 0:
            print("Size: " + convert_size(mat.element_size() * mat.nelement()))
            print('tput_avg (Gbps):',
                  tput_avg.item(),
                  'busbw_avg (Gbps):',
                  busbw_avg.item())
            #dist.log_summary_new()


#def init_processes(fn, backend='nccl', trials=DEFAULT_TRIALS, warmups=DEFAULT_WARMUPS, min_size=DEFAULT_MIN, max_size=DEFAULT_MAX, dtype=DEFAULT_TYPE, step=DEFAULT_STEP):
#    dist.init_distributed(backend)
#    print(f'local rank = {dist.get_local_rank()}')
#    torch.cuda.set_device(dist.get_local_rank())
#    dist.start_profiling_comms()
#    dist.configure(enabled=True)
#    fn(dist.get_local_rank(), trials=trials, warmups=warmups, min_size=min_size, max_size=max_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--warmups',
                        '-w',
                        type=int,
                        default=DEFAULT_WARMUPS,
                        dest='warmups')
    parser.add_argument('--trials',
                        '-i',
                        type=int,
                        default=DEFAULT_TRIALS,
                        dest='trials')
    parser.add_argument('--min', '-min', type=int, default=DEFAULT_MIN, dest='min_size')
    parser.add_argument('--max',
                        '-max',
                        type=int,
                        default=DEFAULT_MAX,
                        dest='max_size')  # 128MB
    parser.add_argument('--step', '-s', type=int, default=DEFAULT_STEP, dest='step')
    parser.add_argument('--type',
                        '-t',
                        type=torch.dtype,
                        default=torch.float32,
                        dest='dtype')
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    run_bench(local_rank=args.local_rank,
              trials=args.trials,
              warmups=args.warmups,
              min_size=args.min_size,
              max_size=args.max_size,
              dtype=args.dtype,
              step=args.step)
