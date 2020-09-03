from mpi4py import MPI
import time
import torch
import torch.distributed as dist
import numpy as np
import deepspeed
from deepspeed.runtime.fp16.onebit_adam import OnebitAdam

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

torch.distributed.init_process_group(backend='nccl',
                                     init_method='tcp://worker-1:2345',
                                     world_size=size,
                                     rank=rank)

dummy_model = [torch.nn.Parameter(torch.ones(10))]
dummy_optim = OnebitAdam(dummy_model)

device = torch.device('cuda', rank % torch.cuda.device_count())


def torch_sim(a):
    a_sign = a.sign().add_(1).bool().float().add_(-0.5).mul_(2.0)
    #a_sign = a.sign()
    scale = a.norm() / np.sqrt(a.numel())
    a_compressed = scale * a_sign
    a_sign = None
    worker_error = a - a_compressed
    dist.all_reduce(a_compressed)
    a_compressed.mul_(1 / dist.get_world_size())
    a_server_sign = a_compressed.sign().add_(1).bool().float().add_(-0.5).mul_(2.0)
    #a_server_sign = a_compressed.sign()
    a_list = torch.chunk(a_compressed, chunks=dist.get_world_size())
    server_scale = [chunk_a.norm() / np.sqrt(chunk_a.numel()) for chunk_a in a_list]
    a_sign_list = torch.chunk(a_server_sign, dist.get_world_size())
    a_server_compressed = torch.cat(
        [server_scale[i] * a_sign_list[i] for i in range(dist.get_world_size())])
    rank = dist.get_rank()
    server_error = a_list[rank] - server_scale[rank] * a_sign_list[rank]
    return a_server_compressed, worker_error, server_error


tensor_size = 200 * 2**20
server_size = int(tensor_size / size)

# a = -torch.ones(tensor_size, device=device)
a = torch.randn(tensor_size, device=device)
#if rank == 0:
#   print('a is: ',a)
worker_error = torch.zeros_like(a)
server_error = torch.zeros(server_size, device=device)
a_torch, worker_error_torch, server_error_torch = torch_sim(a)
torch.cuda.empty_cache()
local_rank = rank % torch.cuda.device_count()
a_after = dummy_optim.Compressed_Allreduce(a,
                                           worker_error,
                                           server_error,
                                           rank,
                                           size,
                                           comm,
                                           local_rank)
#print('a becomes ',a)
#if rank == 0:
if True:
    print('Rank is {} =============================================='.format(rank))
    print('Diff is: ', torch.norm(a_after - a_torch))
    #print('Original Norm is: ', torch.norm(a_after))
    #print('Compressed_addreduce gives: ', a_after[0:10])
    print('Worker error diff is: ', torch.norm(worker_error - worker_error_torch))
    print('Server error is: ', torch.norm(server_error - server_error_torch))
    print('+++++++++++++++++++++++++++++++')
    #print('torch sim gives: ', a_torch[0:10])
