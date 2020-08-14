from hank_compressed_scatter_gather import *
from mpi4py import MPI
import time
import torch
import torch.distributed as dist
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

torch.distributed.init_process_group(backend='nccl', init_method='tcp://worker-0:2345', world_size=size, rank=rank)

device = torch.device('cuda',rank)
def myIgather(rank, size, comm, sendbuf, recbuf, root):
    req = []
    if rank == root:
        for idx in range(size):
            if idx != rank:
                req.append(comm.Irecv(recbuf[idx], source=idx))
            else:
                recbuf[rank] = sendbuf
    else:
        req.append(comm.Isend(sendbuf, dest=root))

    return req

def Compressed_Allreduce(buffer_m: torch.tensor, worker_error, server_error, rank, world_size, comm):
    #print("----------------------------- callling mpi ----------------")
    all_start_time = time.time()
    original_size = buffer_m.numel()
    cupy.cuda.Device(rank % torch.cuda.device_count()).use()
    if torch.numel(buffer_m) != torch.numel(worker_error):
        empty_tensor = torch.zeros(torch.numel(worker_error) - torch.numel(buffer_m), device=buffer_m.device)
        buffer_m = torch.cat([buffer_m, empty_tensor])
    buffer_m.add_(worker_error)
    worker_scale = torch.norm(buffer_m) / np.sqrt(torch.numel(buffer_m))
    buffer_m_bk = buffer_m
    worker_error_bk = worker_error
    partition = buffer_m_bk.numel() // 8
    for i in range(8):
        start = i * partition
        buffer_m = buffer_m_bk.narrow(0, start, partition)
        worker_error = worker_error_bk.narrow(0,start, partition)
        worker_error.set_(buffer_m - worker_scale * buffer_m.sign())

    compensated_buffer_m = buffer_m_bk
    compensated_buffer_m.sign_()
    compensated_buffer_m = compensated_buffer_m.add_(1).bool()
    cupy_worker_scale = torch2cupy(worker_scale)
    cupy_compensated_buffer_m = torch2cupy(compensated_buffer_m)
    compensated_buffer_m = None
    # del compensated_buffer_m
    # del buffer_m
    # print(cupy_compensated_buffer_m)

    cupy_sign_list_packed = compress_by_chunk(cupy_compensated_buffer_m, world_size)
    cupy_compensated_buffer_m = None
    # del cupy_compensated_buffer_m

    cupy_recvbuf_sign = cupy.zeros([world_size, cupy_sign_list_packed[rank].size],
                                   dtype=cupy_sign_list_packed[0].dtype)
    cupy_recvbuf_scale = cupy.zeros([world_size, 1], dtype=cupy_worker_scale.dtype)
    requests = []

    #print("calling igather")

    gather_start = time.time()
    for idx in range(world_size):
        cupy_sign_list_packed[idx] = cupy.asnumpy(cupy_sign_list_packed[idx])
    cupy.cuda.get_current_stream().synchronize() 
    
    #print("cupy completed")

    for idx in range(world_size):
        cupy_recvbuf_sign = cupy.asnumpy(cupy_recvbuf_sign)
        cupy.cuda.get_current_stream().synchronize()
        print ("igather 1 start")
        req_sign = myIgather(rank, world_size, comm, cupy_sign_list_packed[idx], cupy_recvbuf_sign, root=idx)
        requests += req_sign
    cupy.cuda.get_current_stream().synchronize()
    MPI.Request.Waitall(requests)

    print("igather 1 complete")

    cupy.cuda.get_current_stream().synchronize()
    cupy_worker_scale = cupy.asnumpy(cupy_worker_scale)
    cupy_recvbuf_scale = cupy.asnumpy(cupy_recvbuf_scale)
    cupy.cuda.get_current_stream().synchronize()
    requests = []
    for idx in range(world_size):
        #print ("igather 2 start")
        req_scale = myIgather(rank, world_size, comm, cupy_worker_scale, cupy_recvbuf_scale, root=idx)
        requests += req_scale
    MPI.Request.Waitall(requests)
    gather_end = time.time()
    cupy.cuda.get_current_stream().synchronize()
    #print("gather 2 completed")
    cupy_sign_list_packed[idx] = cupy.array(cupy_sign_list_packed[idx])
    cupy_recvbuf_sign = cupy.array(cupy_recvbuf_sign)
    cupy_worker_scale = cupy.array(cupy_worker_scale)
    cupy_recvbuf_scale = cupy.array(cupy_recvbuf_scale)
    cupy.cuda.get_current_stream().synchronize()

    print("copy from numpy to cupy completed")

    cupy_unpacked_sign = (cupy.unpackbits(cupy_recvbuf_sign.flatten())).reshape(world_size, -1)
    cupy_recvbuf_sign = None
    # del cupy_recvbuf_sign
    unpacked_sign = cupy2torch(cupy_unpacked_sign).float()
    cupy_unpacked_sign = None
    # del cupy_unpacked_sign
    unpacked_sign = unpacked_sign.add_(-0.5).mul_(2.0)
    worker_scale = cupy2torch(cupy_recvbuf_scale).mul_(1/world_size)
    compensated_server_m = unpacked_sign.mul_(worker_scale).sum(0)
    unpacked_sign = None
    # del unpacked_sign
    compensated_server_m.add_(server_error)
    server_scale = torch.norm(compensated_server_m) / np.sqrt(compensated_server_m.numel())
    server_error.set_(compensated_server_m - server_scale * compensated_server_m.sign())
    compensated_server_m.sign_()
    compensated_server_m = compensated_server_m.add_(1).bool()
    cupy_server_scale = torch2cupy(server_scale)
    cupy_compensated_server_m = torch2cupy(compensated_server_m)
    compensated_server_m = None
    # del compensated_server_m

    cupy_server_sign_packed = compress_by_chunk(cupy_compensated_server_m, 1)

    cupy_recvbuf_sign_server = cupy.zeros([world_size, cupy_server_sign_packed[0].size],
                                          dtype=cupy_sign_list_packed[0].dtype)
    cupy_recvbuf_scale_server = cupy.zeros([world_size, 1], dtype=cupy_worker_scale.dtype)

    allgather_start = time.time()
    comm.Allgather(cupy_server_sign_packed[0], cupy_recvbuf_sign_server)
    comm.Allgather(cupy_server_scale, cupy_recvbuf_scale_server)
    allgather_end = time.time()

    cupy_server_unpacked_sign = (cupy.unpackbits(cupy_recvbuf_sign_server.flatten())).reshape(world_size, -1)
    cupy_recvbuf_sign_server = None
    # del cupy_recvbuf_sign_server
    server_unpacked_sign = cupy2torch(cupy_server_unpacked_sign)
    cupy_server_unpacked_sign = None
    # del cupy_server_unpacked_sign
    server_unpacked_sign = server_unpacked_sign.float().add_(-0.5).mul_(2.0)
    server_scale = cupy2torch(cupy_recvbuf_scale_server)
    buffer_m = server_unpacked_sign.mul_(server_scale).flatten()[0:original_size]

    cupy._default_memory_pool.free_all_blocks()

    return buffer_m
    
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

tensor_size = 2 * 28 #8 * size * 4
server_size = int(tensor_size/size)

# a = -torch.ones(tensor_size, device=device)
a = torch.randn(tensor_size, device=device)

#if rank == 0:
#    print('a is: ', a)

worker_error = torch.zeros_like(a)
server_error = torch.zeros(server_size, device=device)

a_torch = torch_sim(a)

#a_after = Hank_cupy_compression_com_reduce(a, worker_error, server_error, rank, size, comm)
a_after = Compressed_Allreduce(a, worker_error, server_error, rank, size, comm)

#print('a becomes ', a)
#if rank == 0:
#    print('a_after is: ', a_after)
#    print('torch sim gives: ', a_torch)

