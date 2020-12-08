from mpi4py import MPI
import time
import torch
import torch.distributed as dist
import numpy as np
import deepspeed

from deepspeed.runtime.comm.nccl import NcclBackend

# Configure wall clock timer
from deepspeed.utils.timer import SynchronizedWallClockTimer

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

tensor_size = 100 * 2**20
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

iters = 100

local_rank = rank % torch.cuda.device_count()

# Warmup
for i in range(iters):
    backend.compressed_allreduce(a, worker_error, server_error, local_rank)

time_list = []

for i in range(iters):
    timers('compressed_allreduce').start()
    backend.compressed_allreduce(a, worker_error, server_error, local_rank)
    timers('compressed_allreduce').stop()
    time_list += timers('compressed_allreduce').elapsed()

timer_names = ['compressed_allreduce']
timers.log(names=timer_names, normalizer=iters, memory_breakdown=None)

print(time_list)
