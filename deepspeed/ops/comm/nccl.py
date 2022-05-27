import torch
from deepspeed.ops import op_builder

nccl_cpp_module = None


def build_op():
    global nccl_cpp_module
    builder = op_builder.CommBuilder()
    try:
        nccl_cpp_module = builder.load()
        print(f'DeepSpeed {builder.absolute_name()} built successfully')
        return nccl_cpp_module
    except Exception as inst:
        # if comm cannot be built, use torch.dist.
        print(f"Failed to build {builder.absolute_name()}. Full error: {inst}")
        exit(0)


def get_nccl_id():
    return nccl_cpp_module.getNcclId()


def initialize_mpi():
    nccl_cpp_module.initialize_mpi()


def finalize_mpi():
    nccl_cpp_module.finalize_mpi()


def finalize_nccl():
    nccl_cpp_module.finalize_nccl()


def barrier():
    return nccl_cpp_module.barrier()


def nccl_send(tensor, rank, tag=0):
    nccl_cpp_module.nccl_send(tensor, rank, tag)


def nccl_recv(tensor, rank, tag=0):
    nccl_cpp_module.nccl_recv(tensor, rank, tag)


def all_reduce(tensor, op=None, group=None, async_op=False):
    return nccl_cpp_module.nccl_allreduce(tensor, False)


def nccl_alltoall(outputTensor, inputTensor, is_prof):
    return nccl_cpp_module.nccl_alltoall(outputTensor, inputTensor, is_prof)


def nccl_alltoall_list(outputTensors, inputTensors):
    return nccl_cpp_module.nccl_alltoall(outputTensors, inputTensors)


def mpi_send(tensor, rank, tag=0):
    nccl_cpp_module.mpi_send(tensor, rank, tag)


def mpi_recv(tensor, rank, tag=0):
    nccl_cpp_module.mpi_recv(tensor, rank, tag)


def mpi_isend(tensor, rank, tag=0, comm_index=0):
    return nccl_cpp_module.mpi_isend(tensor, rank, tag, comm_index)


def mpi_irecv(tensor, rank, tag=0, comm_index=0):
    return nccl_cpp_module.mpi_irecv(tensor, rank, tag, comm_index)


def mpi_allreduce(tensor, comm_index, is_prof):
    return nccl_cpp_module.mpi_allreduce(tensor, comm_index, is_prof)


def mpi_allgather(tensor, comm_index=0):
    return nccl_cpp_module.mpi_allgather(tensor, 0)


def mpi_gather(tensor, root_rank, comm_index=0):
    return nccl_cpp_module.mpi_gather(tensor, root_rank, comm_index=0)


#def mpi_scatter(tensor, root_rank, tag=0, comm_index=0):
#        return nccl_cpp_module.mpi_scatter(tensor, root_rank, tag=0, comm_index=0)


def mpi_reduce_scatter(tensor, comm_index=0):
    return nccl_cpp_module.mpi_reduce_scatter(tensor, comm_index=0)


def mpi_reduce(tensor, root_rank, comm_index=0):
    return nccl_cpp_module.mpi_reduce(tensor, root_rank, comm_index=0)


def mpi_bcast(tensor, root_rank, comm_index=0):
    return nccl_cpp_module.mpi_bcast(tensor, root_rank, comm_index=0)


def mpi_alltoall(outputTensor, inputTensor, comm_index=0, is_prof=0):
    return nccl_cpp_module.mpi_alltoall(outputTensor, inputTensor, comm_index, is_prof)


def mpi_alltoall_list(outputTensors, inputTensors, comm_index=0, is_prof=0):
    return nccl_cpp_module.mpi_alltoall(outputTensors, inputTensors, comm_index, is_prof)


def mpi_wait(req):
    nccl_cpp_module.mpi_wait(req)


def mpi_wait_multi(req, tensor):
    return nccl_cpp_module.mpi_wait_multi(req, tensor)


def create_comms(number=1):
    nccl_cpp_module.create_comms(number)


def mpi_cuda_sync():
    nccl_cpp_module.mpi_device_sync()


def print_comms():
    nccl_cpp_module.print_comm_number()
