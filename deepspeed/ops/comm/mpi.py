import torch
from deepspeed.ops import op_builder

mpi_cpp_module = None


def build_mpi_op():
    global mpi_cpp_module
    builder = op_builder.MPICommBuilder()
    try:
        mpi_cpp_module = builder.load()
        print(f'DeepSpeed {builder.absolute_name()} built successfully')
        return mpi_cpp_module
    except Exception as inst:
        # if comm cannot be built, use torch.dist.
        print(f"Failed to build {builder.absolute_name()}. Full error: {inst}")
        exit(0)


def initialize():
    mpi_cpp_module.initialize()

def finalize():
    mpi_cpp_module.finalize()

def barrier():
    return mpi_cpp_module.barrier()

def send(tensor, rank, tag=0):
    mpi_cpp_module.send(tensor, rank, tag)

def recv(tensor, rank, tag=0):
    mpi_cpp_module.recv(tensor, rank, tag)

def all_reduce(tensor, op=None, group=None, async_op=False):
    return mpi_cpp_module.allreduce(tensor, op, async_op)

def alltoall(outputTensor, inputTensor, is_prof):
    return mpi_cpp_module.alltoall(outputTensor, inputTensor, is_prof)

def alltoall_list(outputTensors, inputTensors):
    return mpi_cpp_module.alltoall(outputTensors, inputTensors)

def send(tensor, rank, tag=0):
    mpi_cpp_module.send(tensor, rank, tag)

def recv(tensor, rank, tag=0):
    mpi_cpp_module.recv(tensor, rank, tag)

def isend(tensor, rank, tag=0, comm_index=0):
    return mpi_cpp_module.isend(tensor, rank, tag, comm_index)

def irecv(tensor, rank, tag=0, comm_index=0):
    return mpi_cpp_module.irecv(tensor, rank, tag, comm_index)

def allreduce(tensor, comm_index, is_prof):
    return mpi_cpp_module.allreduce(tensor, comm_index, is_prof)

def allgather(tensor, comm_index=0):
    return mpi_cpp_module.allgather(tensor, 0)

def gather(tensor, root_rank, comm_index=0):
    return mpi_cpp_module.gather(tensor, root_rank, comm_index=0)

#def scatter(tensor, root_rank, tag=0, comm_index=0):
#        return mpi_cpp_module.scatter(tensor, root_rank, tag=0, comm_index=0)

def reduce_scatter(tensor, comm_index=0):
    return mpi_cpp_module.reduce_scatter(tensor, comm_index=0)

def reduce(tensor, root_rank, comm_index=0):
    return mpi_cpp_module.reduce(tensor, root_rank, comm_index=0)

def bcast(tensor, root_rank, comm_index=0):
    return mpi_cpp_module.bcast(tensor, root_rank, comm_index=0)

def alltoall(outputTensor, inputTensor, comm_index=0, is_prof=0):
    return mpi_cpp_module.alltoall(outputTensor, inputTensor, comm_index, is_prof)

def alltoall_list(outputTensors, inputTensors, comm_index=0, is_prof=0):
    return mpi_cpp_module.alltoall(outputTensors, inputTensors, comm_index, is_prof)

def wait(req):
    mpi_cpp_module.wait(req)

def wait_multi(req, tensor):
    return mpi_cpp_module.wait_multi(req, tensor)

def create_comms(number=1):
    mpi_cpp_module.create_comms(number)

def cuda_sync():
    mpi_cpp_module.device_sync()

def print_comms():
    mpi_cpp_module.print_comm_number()