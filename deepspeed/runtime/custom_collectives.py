from mpi4py import MPI
import numpy as np
import cupy


def my_igather(rank, size, comm, sendbuf, recbuf, root):
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


def gather_cuda(rank,
                world_size,
                comm,
                cupy_sign_list_packed,
                cupy_recvbuf_sign,
                cupy_worker_scale,
                cupy_recvbuf_scale):
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

    cupy_worker_scale = cupy.array(numpy_worker_scale)
    cupy_recvbuf_scale = cupy.array(numpy_recvbuf_scale)
    cupy.cuda.get_current_stream().synchronize()


def gather(rank,
           world_size,
           comm,
           cupy_sign_list_packed,
           cupy_recvbuf_sign,
           cupy_worker_scale,
           cupy_recvbuf_scale):
    cuda_aware = False
    if cuda_aware:
        gather_cuda(rank,
                    world_size,
                    comm,
                    cupy_sign_list_packed,
                    cupy_recvbuf_sign,
                    cupy_worker_scale,
                    cupy_recvbuf_scale)
    else:
        gather_host(rank,
                    world_size,
                    comm,
                    cupy_sign_list_packed,
                    cupy_recvbuf_sign,
                    cupy_worker_scale,
                    cupy_recvbuf_scale)


def allgather(comm,
              cupy_server_sign_packed,
              cupy_recvbuf_sign_server,
              cupy_server_scale,
              cupy_recvbuf_scale_server):
    cuda_aware = False
    if cuda_aware:
        comm.Allgather(cupy_server_sign_packed, cupy_recvbuf_sign_server)
        comm.Allgather(cupy_server_scale, cupy_recvbuf_scale_server)
    else:
        # 1. Convert cupy to numpy
        numpy_recvbuf_sign_server = np.zeros(
            [comm.Get_size(),
             cupy_server_sign_packed.size],
            dtype=cupy_server_sign_packed.dtype)
        numpy_recvbuf_scale_server = np.zeros([comm.Get_size(),
                                               1],
                                              dtype=cupy_server_scale.dtype)

        numpy_server_sign_packed = cupy.asnumpy(cupy_server_sign_packed[0])
        numpy_recvbuf_sign_server = cupy.asnumpy(cupy_recvbuf_sign_server)
        numpy_server_scale = cupy.asnumpy(cupy_server_scale)
        numpy_recvbuf_scale_server = cupy.asnumpy(cupy_recvbuf_scale_server)
        cupy.cuda.get_current_stream().synchronize()

        # 2. Communicate numpy buffers
        comm.Allgather(numpy_server_sign_packed, numpy_recvbuf_sign_server)
        comm.Allgather(numpy_server_scale, numpy_recvbuf_scale_server)
        comm.Barrier()

        # 3. Convert numpy back to cupy
        cupy_server_sign_packed = cupy.array(numpy_server_sign_packed)
        cupy_recvbuf_sign_server = cupy.array(numpy_recvbuf_sign_server)
        cupy_server_scale = cupy.array(numpy_server_scale)
        cupy_recvbuf_scale_server = cupy.array(numpy_recvbuf_scale_server)
        cupy.cuda.get_current_stream().synchronize()
