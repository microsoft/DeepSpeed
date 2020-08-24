from mpi4py import MPI
import time
import torch
import torch.distributed as dist
import numpy as np
import deepspeed

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

torch.distributed.init_process_group(backend='nccl', init_method='tcp://worker-0:2345', world_size=size, rank=rank)

dummy_model = torch.nn.Linear(10,10)
dummy_optim = deepspeed.OnebitAdam(dummy_model)

device = torch.device('cuda',rank)

def torch_sim(a):
    a_sign = a.sign()
    scale = a.norm() / a_sign.norm()
    a_compressed = scale * a_sign
    dist.all_reduce(a_compressed)
    a_compressed.mul_(1/dist.get_world_size())
    a_server_sign = a_compressed.sign()
    a_list = torch.chunk(a_compressed,chunks=dist.get_world_size())
    server_scale = [chunk_a.norm()/np.sqrt(chunk_a.numel()) for chunk_a in a_list]
    a_sign_list = torch.chunk(a_server_sign,dist.get_world_size())
    a_server_compressed = torch.cat([ server_scale[i] * a_sign_list[i] for i in range(dist.get_world_size()) ])
    return a_server_compressed

tensor_size = 8 * size * 4
server_size = int(tensor_size/size)

# a = -torch.ones(tensor_size, device=device)
a = torch.randn(tensor_size, device=device)
if rank == 0:
    print('a is: ',a)
worker_error = torch.zeros_like(a)
server_error = torch.zeros(server_size, device=device)
a_torch = torch_sim(a)


a_after = dummy_optim.Compressed_Allreduce(a, worker_error, server_error, rank, size, comm)
print('a becomes ',a)
if rank == 0:
    print('a_after is: ', a_after)
    print('torch sim gives: ', a_torch)

