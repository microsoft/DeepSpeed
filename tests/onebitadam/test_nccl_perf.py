from mpi4py import MPI
import time
import torch
import torch.distributed as dist
import numpy as np
import deepspeed

from deepspeed.runtime.comm.nccl import NcclBackend

# Configure wall clock timer
from deepspeed.utils.timer import SynchronizedWallClockTimer

from statistics import mean 

timers = SynchronizedWallClockTimer()

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#TODO: Detect the hostname we are running on automatically
torch.distributed.init_process_group(backend='nccl',
                                     init_method='tcp://worker-0:2245',
                                     world_size=size,
                                     rank=rank)

backend = NcclBackend()

device = torch.device('cuda', rank % torch.cuda.device_count())

tensor_size = 300 * 2**20
server_size = int(tensor_size / size)
if tensor_size % (8 * size) != 0:
    right_tensor_size = tensor_size + (8 * size - (tensor_size % (8 * size)))
else:
    right_tensor_size = tensor_size
right_server_size = right_tensor_size // size

# Adding bias to the initialization of the gradient we are communicating
# In order to get rid of the case where some elements in the gradient are too small
a = (torch.rand(tensor_size, device=device) - 0.5) + 0.01 * rank

worker_error = torch.zeros(right_tensor_size, device=device)
server_error = torch.zeros(right_server_size, device=device)

warmup = 10
iters = 10

local_rank = rank % torch.cuda.device_count()

# Warmup
for i in range(warmup):
    backend.compressed_allreduce(a, worker_error, server_error, local_rank)

time_list = []

a_sign = a.sign().add_(1).bool().float().add_(-0.5).mul_(2.0)
scale = a.norm() / np.sqrt(a.numel())
a_compressed = scale * a_sign
print(a_compressed.shape)

for i in range(iters):
    timers('compressed_allreduce').start()
    backend.compressed_allreduce(a, worker_error, server_error, local_rank)
    #torch.distributed.all_reduce(a_compressed)
    timers('compressed_allreduce').stop()
    time_list.append(timers('compressed_allreduce').elapsed())

timer_names = ['compressed_allreduce']
timers.log(names=timer_names, normalizer=1, memory_breakdown=None)

places = 2
convert = 1e3
float_size = 4

if rank == 0:
    for i in range(iters):
        lat = time_list[i]
        print("latency = ", lat * convert)

minlat = round(min(time_list) * convert)
maxlat = round(max(time_list) * convert)
meanlat = round(mean(time_list) * convert, places)
print("min, max, and mean = {} ms, {} ms, {} ms".format(minlat, maxlat, meanlat))
print("tensor shape", a.shape)
duration=meanlat/1e3
tput = ((tensor_size*4)/duration)
print("algo throughput: %f Bytes/s, %f GB/s" % (tput, tput/1e9))
size = tensor_size * 4
n = dist.get_world_size()
busbw = (size / duration) * (2 * (n - 1) / n)
print("busbw: %f GB/s" % (busbw / 1e9))
