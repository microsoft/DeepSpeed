import torch
import cupy
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import math
import time
import torch.distributed as dist
from mpi4py import MPI

import pdb
import numpy as np



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

def torch2cupy(tensor):
    return cupy.fromDlpack(to_dlpack(tensor))

def cupy2torch(cupy_tensor):
    return from_dlpack(cupy_tensor.toDlpack())

def compress_by_chunk(cupy_bool_tensor, num_chunks):
    packed_sign = cupy.packbits(cupy_bool_tensor)
    sign_list_packed = cupy.split(packed_sign, num_chunks)
    return sign_list_packed

def hank_compress_by_chunk(cupy_bool_tensor, num_chunks):
    packed_sign = cupy.packbits(cupy_bool_tensor)
    if num_chunks > 1:
        sign_list_packed = cupy.split(packed_sign, num_chunks)
    else:
        sign_list_packed = packed_sign
    return sign_list_packed


def Reza_compress_by_chunk(tensor, num_chunk, packer):
    (norm, sign_packed) = packer.Pack(tensor)
    # if num_chunk > 1 :
    #     sign_list_packed = torch.chunk(sign_packed, chunks= num_chunk)
    # else:
    #     sign_list_packed = sign_packed
    return norm, sign_packed

def Hank_cupy_compression_com_reduce(buffer_m: torch.tensor, worker_error, server_error, rank, world_size, comm):

    all_start_time = time.time()
    cupy.cuda.Device(rank % torch.cuda.device_count()).use()
    if torch.numel(buffer_m) != torch.numel(worker_error):
        empty_tensor = torch.zeros(torch.numel(worker_error) - torch.numel(buffer_m), device = buffer_m.device)
        buffer_m = torch.cat([buffer_m, empty_tensor])
    buffer_m.add_(worker_error)
    compensated_buffer_m = buffer_m
    worker_scale = torch.norm(compensated_buffer_m)/np.sqrt(torch.numel(compensated_buffer_m))
    worker_error.set_(compensated_buffer_m - worker_scale * compensated_buffer_m.sign())

    compensated_buffer_m.sign_()
    compensated_buffer_m = compensated_buffer_m.add_(1).bool()
    cupy_worker_scale = torch2cupy(worker_scale)
    cupy_compensated_buffer_m = torch2cupy(compensated_buffer_m)
    compensated_buffer_m = None
    # print(cupy_compensated_buffer_m)

    cupy_sign_list_packed = hank_compress_by_chunk(cupy_compensated_buffer_m, world_size)
    # if rank == 0:
    #     print('cupy_compensated_buffer_m is: ',cupy_compensated_buffer_m)
    cupy_compensated_buffer_m = None

    cupy_recvbuf_sign = cupy.zeros([world_size, cupy_sign_list_packed[rank].size], dtype=cupy_sign_list_packed[0].dtype)
    cupy_recvbuf_scale = cupy.zeros([world_size, 1], dtype=cupy_worker_scale.dtype)
    requests = []

    gather_start = time.time()
    for idx in range(world_size):
        req_sign = myIgather(rank, world_size, comm, cupy_sign_list_packed[idx], cupy_recvbuf_sign, root=idx)
        requests += req_sign
    for idx in range(world_size):
        req_scale = myIgather(rank, world_size, comm, cupy_worker_scale, cupy_recvbuf_scale, root=idx)
        requests += req_scale
    MPI.Request.Waitall(requests)
    gather_end = time.time()

    cupy_unpacked_sign = (cupy.unpackbits(cupy_recvbuf_sign.flatten())).reshape(world_size, -1)
    # if rank == 0:
    #     print('cupy_unpacked_sign is: ',cupy_unpacked_sign)
    cupy_recvbuf_sign = None
    unpacked_sign = cupy2torch(cupy_unpacked_sign).float().add_(-0.5).mul_(2.0)
    cupy_unpacked_sign = None
    worker_scale = cupy2torch(cupy_recvbuf_scale).mul_(1/world_size)
    compensated_server_m = unpacked_sign.mul_(worker_scale).sum(0).add_(server_error)
    unpacked_sign = None
    server_scale = torch.norm(compensated_server_m)/np.sqrt(compensated_server_m.numel())
    server_error.set_(compensated_server_m - server_scale * compensated_server_m.sign())

    compensated_server_m.sign_()
    compensated_server_m = compensated_server_m.add_(1).bool()
    cupy_server_scale = torch2cupy(server_scale)
    cupy_compensated_server_m = torch2cupy(compensated_server_m)
    compensated_server_m = None

    cupy_server_sign_packed = hank_compress_by_chunk(cupy_compensated_server_m, 1)

    cupy_recvbuf_sign_server = cupy.zeros([world_size, cupy_server_sign_packed.size], dtype=cupy_sign_list_packed[0].dtype)
    cupy_recvbuf_scale_server = cupy.zeros([world_size, 1], dtype=cupy_worker_scale.dtype)

    allgather_start = time.time()
    comm.Allgather(cupy_server_sign_packed, cupy_recvbuf_sign_server)
    comm.Allgather(cupy_server_scale, cupy_recvbuf_scale_server)
    allgather_end = time.time()

    cupy_server_unpacked_sign = (cupy.unpackbits(cupy_recvbuf_sign_server.flatten())).reshape(world_size, -1)
    # if rank == 0:
    #     print(cupy_server_unpacked_sign)
    cupy_recvbuf_sign_server = None
    server_unpacked_sign = cupy2torch(cupy_server_unpacked_sign).float().add_(-0.5).mul_(2.0)
    # if rank == 0:
    #     print(server_unpacked_sign)
    cupy_server_unpacked_sign = None
    server_scale = cupy2torch(cupy_recvbuf_scale_server)
    buffer_m = server_unpacked_sign.mul_(server_scale).flatten()

    # cupy_server_final_m = (cupy_recvbuf_scale_server * cupy_server_unpacked_sign).flatten()

    # buffer_m.set_(cupy2torch(cupy_server_final_m)[0:buffer_m.numel()])
    # exit()

    all_end_time = time.time()

    return buffer_m




def Hank_com_reduce(buffer_m: torch.tensor, worker_error, server_error, rank, world_size, comm, packer, printkey):
    from mpi4py import MPI
    all_start_time = time.time()
    cupy.cuda.Device(rank % torch.cuda.device_count()).use()
    if torch.numel(buffer_m) != torch.numel(worker_error):
        empty_tensor = torch.cuda.FloatTensor(torch.numel(worker_error) - torch.numel(buffer_m)).fill_(0)
        buffer_m = torch.cat([buffer_m, empty_tensor])
    compensated_buffer_m = buffer_m + worker_error

    norm, sign_packed = Reza_compress_by_chunk(compensated_buffer_m, world_size, packer)
    torch.cuda.synchronize()
    cupy_sign_list_packed = cupy.split(cupy.fromDlpack(to_dlpack(sign_packed)), world_size)
    # print('Norm is on device: {}'.format(norm.device))
    scale = norm/np.sqrt(buffer_m.numel())
    cupy_scale_list = [cupy.fromDlpack(to_dlpack(scale)) for i in range(world_size)]

    worker_error.set_(compensated_buffer_m - scale * compensated_buffer_m.sign())

    cupy_recvbuf_sign = cupy.zeros([world_size, cupy_sign_list_packed[rank].size], dtype='int8')
    cupy_recvbuf_scale = cupy.zeros([world_size, 1], dtype = 'float32')
    requests = []

    gather_start = time.time()
    for idx in range(world_size):
        req_sign = myIgather(rank, world_size, comm, cupy_sign_list_packed[idx], cupy_recvbuf_sign, root=idx)
        requests += req_sign
    for idx in range(world_size):
        req_scale = myIgather(rank, world_size, comm, cupy_scale_list[idx], cupy_recvbuf_scale, root=idx)
        requests += req_scale
    MPI.Request.Waitall(requests)
    gather_end = time.time()

    recvbuf_sign = from_dlpack(cupy_recvbuf_sign.toDlpack())
    req_scale = from_dlpack(cupy_recvbuf_scale.toDlpack())

    flatten_sign = packer.Unpack(recvbuf_sign.flatten())[0].view(world_size, -1)
    server_m = flatten_sign.mul_(req_scale).sum(0)
    compensated_server_m = server_m.add_(server_error)

    server_norm, server_sign_packed = Reza_compress_by_chunk(compensated_server_m, 1, packer)
    torch.cuda.synchronize()
    cupy_server_sign_packed = cupy.fromDlpack(to_dlpack(server_sign_packed))
    server_scale = server_norm / np.sqrt(server_m.numel())
    cupy_server_scale = cupy.fromDlpack(to_dlpack(server_scale))
    server_error.set_(compensated_server_m - server_scale * compensated_server_m.sign())

    cupy_recvbuf_sign_server = cupy.zeros([world_size, cupy_server_sign_packed.size], dtype='int8')
    cupy_recvbuf_scale_server = cupy.zeros([world_size, 1], dtype='float32')

    allgather_start = time.time()
    comm.Allgather(cupy_server_sign_packed, cupy_recvbuf_sign_server)
    comm.Allgather(cupy_server_scale, cupy_recvbuf_scale_server)
    allgather_end = time.time()

    recvbuf_sign_server = from_dlpack(cupy_recvbuf_sign_server.toDlpack())
    recvbuf_scale_server = from_dlpack(cupy_recvbuf_scale_server.toDlpack())
    flatten_server_sign = packer.Unpack(recvbuf_sign_server.flatten())[0].view(world_size, -1)
    buffer_m.set_(flatten_server_sign.mul(recvbuf_scale_server).flatten()[0:buffer_m.numel()])
    # exit()

    all_end_time = time.time()
    # cupy.cuda.get_current_stream().synchronize()
    # torch.cuda.synchronize

    return gather_end - gather_start, allgather_end - allgather_start, all_end_time - all_start_time


def com_reduce(buffer_m: torch.tensor, worker_error, server_error, rank, world_size, comm, printkey):
    # print('rank is :', rank)

    all_start_time = time.time()
    chunk_size = torch.numel(server_error)
    flatten_buffer_m = buffer_m.flatten()

    cupy.cuda.Device(rank%torch.cuda.device_count()).use()

    if torch.numel(flatten_buffer_m) != torch.numel(worker_error):
        empty_tensor = torch.cuda.FloatTensor(torch.numel(worker_error) - torch.numel(flatten_buffer_m)).fill_(0)
        flatten_buffer_m = torch.cat([flatten_buffer_m, empty_tensor])

    compensated_buffer_m = flatten_buffer_m + worker_error

    compensated_buffer_m_cupy = cupy.fromDlpack(to_dlpack(compensated_buffer_m))
    # print('rank is {}, and cupy device is {}'.format(rank,compensated_buffer_m_cupy.device) )
    # if compensated_buffer_m_cupy.device is not cupy.cuda.Device(rank):
    #     print('mismatch of the cupy device')


    # sign_list_packed, scale_list = compress_by_chunk(compensated_buffer_m_cupy, world_size, chunk_size, worker_error)

    scale_list = [cupy.fromDlpack(to_dlpack(torch.norm(tensor)))/np.sqrt(tensor.numel()) for tensor in torch.chunk(flatten_buffer_m,chunks=world_size)]
    sign_list_packed = hank_compress_by_chunk(compensated_buffer_m_cupy, world_size, chunk_size, worker_error)

    # First round of communication
    recvbuf_sign = cupy.zeros([world_size, sign_list_packed[rank].size], dtype=sign_list_packed[rank].dtype)
    # if rank == 0:
    #     print('#parameters in one block for calling allgather is {}'.format(len(sign_list_packed[0] )*world_size ))
    recvbuf_scale = cupy.zeros([world_size, 1], dtype=scale_list[rank].dtype)
    # cupy.cuda.get_current_stream().synchronize()

    requests = []

    gather_start = time.time()
    for idx in range(world_size):
        req_sign = myIgather(rank, world_size, comm, sign_list_packed[idx], recvbuf_sign, root=idx)
        requests += req_sign
    for idx in range(world_size):
        req_scale = myIgather(rank, world_size, comm, scale_list[idx], recvbuf_scale, root=idx)
        requests += req_scale

    MPI.Request.Waitall(requests)

    gather_end = time.time()

    flattened_sign = recvbuf_sign.flatten()
    unpacked_sign = cupy.unpackbits(flattened_sign).astype(cupy.float32)
    local_uncompressed = cupy.zeros_like(unpacked_sign)

    # _decompress_kernel(unpacked_sign, recvbuf_scale, chunk_size, local_uncompressed)
    numBlocks_ = (local_uncompressed.size + block_size - 1) // block_size
    _decompress_kernel_binary((numBlocks_,), (block_size,),
                              (unpacked_sign, recvbuf_scale, chunk_size, local_uncompressed))
    # cupy.cuda.get_current_stream().synchronize()

    local_reduced_chunk = cupy.zeros(chunk_size, dtype=cupy.float32)
    _avg_chunks(local_uncompressed, chunk_size, world_size, local_reduced_chunk)

    # add server error
    server_error_cupy = cupy.fromDlpack(to_dlpack(server_error))
    # local_reduced_chunk_compansated = cupy.zeros_like(local_reduced_chunk)

    # cupy_add(local_reduced_chunk, server_error_cupy, local_reduced_chunk_compansated)
    local_reduced_chunk += server_error_cupy

    # sign_list_packed_server, scale_list_server = compress_by_chunk(local_reduced_chunk_compansated, 1, chunk_size,
    #                                                                server_error)
    scale_list_server = [cupy.linalg.norm(local_reduced_chunk)/np.sqrt(chunk_size)]
    sign_list_packed_server = hank_compress_by_chunk(local_reduced_chunk, 1, chunk_size, server_error)

    # prepare buffer
    recvbuf_sign_server = cupy.zeros([world_size, sign_list_packed[0].size], dtype=sign_list_packed_server[0].dtype)
    recvbuf_scale_server = cupy.zeros([world_size, 1], dtype=scale_list_server[0].dtype)
    # cupy.cuda.get_current_stream().synchronize()

    allgather_start = time.time()

    req_server_sign = comm.Allgather(sign_list_packed_server[0], recvbuf_sign_server)
    # # print('Internal: the size of allgather is {}'.format(len(sign_list_packed_server[0]) * world_size))
    req_server_scale = comm.Allgather(scale_list_server[0], recvbuf_scale_server)
    #
    # MPI.Request.Waitall([req_server_sign, req_server_scale])

    allgather_end = time.time()

    flattened_sign_server = recvbuf_sign_server.flatten()
    unpacked_sign_server = cupy.unpackbits(flattened_sign_server).astype(cupy.float32)
    server_uncompressed = cupy.zeros_like(unpacked_sign_server)

    # _decompress_kernel(unpacked_sign_server, recvbuf_scale_server, chunk_size, server_uncompressed)
    numBlocks_ = (server_uncompressed.size + block_size - 1) // block_size
    _decompress_kernel_binary((numBlocks_,), (block_size,),
                              (unpacked_sign_server, recvbuf_scale_server, chunk_size, server_uncompressed))

    aggregated_m_tensor = from_dlpack(server_uncompressed.toDlpack())

    aggregated_m_tensor = aggregated_m_tensor[0:torch.numel(buffer_m)]
    buffer_m.set_(aggregated_m_tensor.type(buffer_m.dtype).view_as(buffer_m))

    all_end_time = time.time()

    return gather_end - gather_start, allgather_end - allgather_start, all_end_time - all_start_time


