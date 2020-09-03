from mpi4py import MPI


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


def myIgather_new(rank, size, comm, sendbuf, recbuf, root):
    req = []
    if rank == root:
        for idx in range(size):
            if idx != rank:
                req.append(comm.Irecv(recbuf[idx], source=idx))
    else:
        comm.Send(sendbuf, dest=root)
    return req


def myGather(rank, size, comm, sendbuf, recbuf, root):
    #req = []
    req = 1
    if rank == root:
        for idx in range(size):
            if idx != rank:
                #req.append(comm.Irecv(recbuf[idx], source=idx))
                req = comm.Irecv(recbuf[idx], source=idx)
                req.wait()
            else:
                recbuf[rank] = sendbuf
    else:
        #req.append(comm.Isend(sendbuf, dest=root))
        #req = comm.Isend(sendbuf, dest=root)
        comm.Send(sendbuf, dest=root)
        #req.wait()

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
        req_sign = myIgather(rank,
                             world_size,
                             comm,
                             cupy_sign_list_packed[idx],
                             cupy_recvbuf_sign,
                             root=idx)
        requests += req_sign

    for idx in range(world_size):
        req_scale = myIgather(rank,
                              world_size,
                              comm,
                              cupy_worker_scale,
                              cupy_recvbuf_scale,
                              root=idx)
        requests += req_scale

    MPI.Request.Waitall(requests)

    #return cupy_sign_list_packed, cupy_recvbuf_sign, cupy_worker_scale, cupy_recvbuf_scale


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

    #print ("calling igather at rank ", rank, flush=True)

    # 2. use numpy buffers for communication
    requests = []

    for idx in range(world_size):
        #print ("igather1 queued at idx, rank, sizes,", idx, rank, cupy_sign_list_packed[idx].size, cupy_recvbuf_sign.size, flush=True)
        req_sign = myIgather(rank,
                             world_size,
                             comm,
                             numpy_sign_list_packed[idx],
                             numpy_recvbuf_sign,
                             root=idx)
        requests += req_sign

    for idx in range(world_size):
        req_scale = myIgather(rank,
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

    #print("igather and conversion completed at rank ", rank, flush=True)

    #return cupy_sign_list_packed, cupy_recvbuf_sign, cupy_worker_scale, cupy_recvbuf_scale


def gather(rank,
           world_size,
           comm,
           cupy_sign_list_packed,
           cupy_recvbuf_sign,
           cupy_worker_scale,
           cupy_recvbuf_scale):
    cuda_aware = True
    if cuda_aware:
        #cupy_sign_list_packed, cupy_recvbuf_sign, cupy_worker_scale, cupy_recvbuf_scale =
        gather_cuda(rank,
                    world_size,
                    comm,
                    cupy_sign_list_packed,
                    cupy_recvbuf_sign,
                    cupy_worker_scale,
                    cupy_recvbuf_scale)
    else:
        #cupy_sign_list_packed, cupy_recvbuf_sign, cupy_worker_scale, cupy_recvbuf_scale =
        gather_host(rank,
                    world_size,
                    comm,
                    cupy_sign_list_packed,
                    cupy_recvbuf_sign,
                    cupy_worker_scale,
                    cupy_recvbuf_scale)

    #return cupy_sign_list_packed, cupy_recvbuf_sign, cupy_worker_scale, cupy_recvbuf_scale


def allgather(comm,
              cupy_server_sign_packed,
              cupy_recvbuf_sign_server,
              cupy_server_scale,
              cupy_recvbuf_scale_server):
    cuda_aware = True
    if cuda_aware:
        comm.Allgather(cupy_server_sign_packed, cupy_recvbuf_sign_server)
        comm.Allgather(cupy_server_scale, cupy_recvbuf_scale_server)
    else:
        # 1. Convert cupy to numpy
        numpy_recvbuf_sign_server = np.zeros([world_size,
                                              cupy_server_sign_packed.size],
                                             dtype=cupy_sign_list_packed.dtype)
        numpy_recvbuf_scale_server = np.zeros([world_size,
                                               1],
                                              dtype=cupy_worker_scale.dtype)

        numpy_server_sign_packed = cupy.asnumpy(cupy_server_sign_packed[0])
        numpy_recvbuf_sign_server = cupy.asnumpy(cupy_recvbuf_sign_server)
        numpy_server_scale = cupy.asnumpy(cupy_server_scale)
        numpy_recvbuf_scale_server = cupy.asnumpy(cupy_recvbuf_scale_server)
        cupy.cuda.get_current_stream().synchronize()

        # 2. Communicate numpy buffers

        #print("allgather 1 called: ", flush=True)
        comm.Allgather(numpy_server_sign_packed, numpy_recvbuf_sign_server)
        #print("allgather 2 called", flush=True)
        comm.Allgather(numpy_server_scale, numpy_recvbuf_scale_server)
        #print("allgather 2 finished", flush=True)
        comm.Barrier()

        # 3. Convert numpy back to cupy
        cupy_server_sign_packed = cupy.array(numpy_server_sign_packed)
        cupy_recvbuf_sign_server = cupy.array(numpy_recvbuf_sign_server)
        cupy_server_scale = cupy.array(numpy_server_scale)
        cupy_recvbuf_scale_server = cupy.array(numpy_recvbuf_scale_server)
        cupy.cuda.get_current_stream().synchronize()

    #return cupy_server_sign_packed, cupy_recvbuf_sign_server, cupy_server_scale, cupy_recvbuf_scale_server
