'''
Copyright 2020 The Microsoft DeepSpeed Team
'''

import torch
import cupy
import numpy as np
from mpi4py import MPI

class MpiBackend(object):
    def __init__(self, cuda_aware):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.cuda_aware = cuda_aware

    def my_igather(self, rank, size, comm, sendbuf, recbuf, root):
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

    def gather_cuda(self,
                    rank,
                    world_size,
                    comm,
                    cupy_sign_list_packed,
                    cupy_recvbuf_sign,
                    cupy_worker_scale,
                    cupy_recvbuf_scale):
        # We do in-place operations on cupy buffers so we do not return any buffers
        requests = []
        for idx in range(world_size):
            req_sign = my_igather(rank,
                                  world_size,
                                  comm,
                                  cupy_sign_list_packed[idx],
                                  cupy_recvbuf_sign,
                                  root=idx)
            requests += req_sign

        for idx in range(world_size):
            req_scale = my_igather(rank,
                                   world_size,
                                   comm,
                                   cupy_worker_scale,
                                   cupy_recvbuf_scale,
                                   root=idx)
            requests += req_scale

        MPI.Request.Waitall(requests)

    def gather_host(rank,
                    world_size,
                    comm,
                    cupy_sign_list_packed,
                    cupy_recvbuf_sign,
                    cupy_worker_scale,
                    cupy_recvbuf_scale):

        # In-place operations are not possible for newly created cupy arrays
        # so we need to return the new buffers
        numpy_recvbuf_sign = np.zeros([world_size,
                                       cupy_sign_list_packed[rank].size],
                                      dtype=cupy_sign_list_packed[0].dtype)
        numpy_recvbuf_scale = np.zeros([world_size, 1], dtype=cupy_worker_scale.dtype)

        # 1. convert from cupy to numpy
        numpy_sign_list_packed = cupy_sign_list_packed

        for idx in range(world_size):
            numpy_sign_list_packed[idx] = cupy.asnumpy(cupy_sign_list_packed[idx])

        numpy_worker_scale = cupy.asnumpy(cupy_worker_scale)
        numpy_recvbuf_scale = cupy.asnumpy(cupy_recvbuf_scale)

        cupy.cuda.get_current_stream().synchronize()

        # 2. use numpy buffers for communication
        requests = []

        for idx in range(world_size):
            req_sign = my_igather(rank,
                                  world_size,
                                  comm,
                                  numpy_sign_list_packed[idx],
                                  numpy_recvbuf_sign,
                                  root=idx)
            requests += req_sign

        for idx in range(world_size):
            req_scale = my_igather(rank,
                                   world_size,
                                   comm,
                                   numpy_worker_scale,
                                   numpy_recvbuf_scale,
                                   root=idx)
            requests += req_scale

        MPI.Request.Waitall(requests)

        # 3. Convert back from numpy to cupy
        cupy_recvbuf_sign = cupy.asarray(numpy_recvbuf_sign)
        for idx in range(world_size):
            cupy_sign_list_packed[idx] = cupy.asarray(numpy_sign_list_packed[idx])

        cupy_worker_scale = cupy.asarray(numpy_worker_scale)
        cupy_recvbuf_scale = cupy.asarray(numpy_recvbuf_scale)
        cupy.cuda.get_current_stream().synchronize()

        return cupy_sign_list_packed, cupy_recvbuf_sign, cupy_worker_scale, cupy_recvbuf_scale

    def allgather_cuda(comm,
                       cupy_server_sign_packed,
                       cupy_recvbuf_sign_server,
                       cupy_server_scale,
                       cupy_recvbuf_scale_server):
        comm.Allgather(cupy_server_sign_packed, cupy_recvbuf_sign_server)
        comm.Allgather(cupy_server_scale, cupy_recvbuf_scale_server)

    def allgather_host(comm,
                       cupy_server_sign_packed,
                       cupy_recvbuf_sign_server,
                       cupy_server_scale,
                       cupy_recvbuf_scale_server):

        # 1. Convert cupy to numpy
        numpy_recvbuf_sign_server = np.zeros(
            [comm.Get_size(),
             cupy_server_sign_packed.size],
            dtype=cupy_server_sign_packed.dtype)
        numpy_recvbuf_scale_server = np.zeros([comm.Get_size(),
                                               1],
                                              dtype=cupy_server_scale.dtype)

        numpy_server_sign_packed = cupy.asnumpy(cupy_server_sign_packed)
        numpy_recvbuf_sign_server = cupy.asnumpy(cupy_recvbuf_sign_server)
        numpy_server_scale = cupy.asnumpy(cupy_server_scale)
        numpy_recvbuf_scale_server = cupy.asnumpy(cupy_recvbuf_scale_server)
        cupy.cuda.get_current_stream().synchronize()

        # 2. Communicate numpy buffers
        comm.Allgather(numpy_server_sign_packed, numpy_recvbuf_sign_server)
        comm.Allgather(numpy_server_scale, numpy_recvbuf_scale_server)
        comm.Barrier()

        # 3. Convert numpy back to cupy
        cupy_server_sign_packed = cupy.asarray(numpy_server_sign_packed)
        cupy_recvbuf_sign_server = cupy.asarray(numpy_recvbuf_sign_server)
        cupy_server_scale = cupy.asarray(numpy_server_scale)
        cupy_recvbuf_scale_server = cupy.asarray(numpy_recvbuf_scale_server)
        cupy.cuda.get_current_stream().synchronize()

        return cupy_server_sign_packed, cupy_recvbuf_sign_server, cupy_server_scale, cupy_recvbuf_scale_server

    def compressed_allreduce(self,
                             buffer_m: torch.tensor,
                             worker_error,
                             server_error,
                             local_rank):

        all_start_time = time.time()
        original_size = buffer_m.numel()
        cupy.cuda.Device(local_rank).use()

        if torch.numel(buffer_m) != torch.numel(worker_error):
            empty_tensor = torch.zeros(torch.numel(worker_error) - torch.numel(buffer_m),
                                       device=buffer_m.device)
            buffer_m = torch.cat([buffer_m, empty_tensor])

        buffer_m.add_(worker_error)
        worker_scale = torch.norm(buffer_m) / np.sqrt(torch.numel(buffer_m))
        sign_buffer_m = buffer_m.sign().add_(1).bool()
        sign_buffer_m = sign_buffer_m.float()
        sign_buffer_m.add_(-0.5).mul_(2.0)
        worker_error.set_((buffer_m - worker_scale * sign_buffer_m))
        sign_buffer_m = None

        compensated_buffer_m = buffer_m
        compensated_buffer_m.sign_()
        compensated_buffer_m = compensated_buffer_m.add_(1).bool()
        cupy_worker_scale = self.torch2cupy(worker_scale)
        cupy_compensated_buffer_m = self.torch2cupy(compensated_buffer_m)
        compensated_buffer_m = None

        cupy_sign_list_packed = self.compress_by_chunk(cupy_compensated_buffer_m,
                                                       self.size)
        cupy_compensated_buffer_m = None

        cupy_recvbuf_sign = cupy.zeros(
            [self.size,
             cupy_sign_list_packed[self.rank].size],
            dtype=cupy_sign_list_packed[0].dtype)
        cupy_recvbuf_scale = cupy.zeros([self.size, 1], dtype=cupy_worker_scale.dtype)

        # Communication Phase 1
        gather_start = time.time()
        if self.cuda_aware:
            gather_cuda(self.rank,
                        self.size,
                        self.comm,
                        cupy_sign_list_packed,
                        cupy_recvbuf_sign,
                        cupy_worker_scale,
                        cupy_recvbuf_scale)
        else:
            cupy_sign_list_packed, cupy_recvbuf_sign, cupy_worker_scale, cupy_recvbuf_scale = gather_host(self.rank,
               self.size,
               self.comm,
               cupy_sign_list_packed,
               cupy_recvbuf_sign,
               cupy_worker_scale,
               cupy_recvbuf_scale)
        gather_end = time.time()

        cupy_unpacked_sign = (cupy.unpackbits(cupy_recvbuf_sign.flatten())).reshape(
            self.size,
            -1)
        cupy_recvbuf_sign = None
        unpacked_sign = self.cupy2torch(cupy_unpacked_sign).float()
        cupy_unpacked_sign = None
        unpacked_sign = unpacked_sign.add_(-0.5).mul_(2.0)
        worker_scale = self.cupy2torch(cupy_recvbuf_scale).mul_(1 / self.size)
        compensated_server_m = unpacked_sign.mul_(worker_scale).sum(0)
        unpacked_sign = None

        compensated_server_m.add_(server_error)
        server_scale = torch.norm(compensated_server_m) / np.sqrt(
            compensated_server_m.numel())
        sign_server_m = compensated_server_m.sign().add_(1).bool()
        sign_server_m = sign_server_m.float()
        sign_server_m.add_(-0.5).mul_(2.0)
        server_error.set_(compensated_server_m - server_scale * sign_server_m)
        sign_server_m = None

        compensated_server_m.sign_()
        compensated_server_m = compensated_server_m.add_(1).bool()
        cupy_server_scale = self.torch2cupy(server_scale)
        cupy_compensated_server_m = self.torch2cupy(compensated_server_m)
        compensated_server_m = None

        cupy_server_sign_packed = self.compress_by_chunk(cupy_compensated_server_m, 1)

        cupy_recvbuf_sign_server = cupy.zeros(
            [self.size,
             cupy_server_sign_packed[0].size],
            dtype=cupy_sign_list_packed[0].dtype)
        cupy_recvbuf_scale_server = cupy.zeros([self.size,
                                                1],
                                               dtype=cupy_worker_scale.dtype)

        # Communication Phase 2
        if self.cuda_aware:
            allgather_cuda(self.comm,
                           cupy_server_sign_packed[0],
                           cupy_recvbuf_sign_server,
                           cupy_server_scale,
                           cupy_recvbuf_scale_server)
        else:
            cupy_server_sign_packed[0], cupy_recvbuf_sign_server, cupy_server_scale, cupy_recvbuf_scale_server = allgather_host(self.comm,
                  cupy_server_sign_packed[0],
                  cupy_recvbuf_sign_server,
                  cupy_server_scale,
                  cupy_recvbuf_scale_server)

        cupy_server_unpacked_sign = (cupy.unpackbits(
            cupy_recvbuf_sign_server.flatten())).reshape(self.size,
                                                         -1)
        cupy_recvbuf_sign_server = None

        server_unpacked_sign = self.cupy2torch(cupy_server_unpacked_sign)
        cupy_server_unpacked_sign = None

        server_unpacked_sign = server_unpacked_sign.float().add_(-0.5).mul_(2.0)
        server_scale = self.cupy2torch(cupy_recvbuf_scale_server)
        buffer_m = server_unpacked_sign.mul_(server_scale).flatten()[0:original_size]

        return buffer_m
