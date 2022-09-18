import torch
from deepspeed.ops import op_builder

nccl_cpp_module = None


def build_nccl_op():
    global nccl_cpp_module
    builder = op_builder.NCCLCommBuilder()
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


def initialize():
    nccl_cpp_module.initialize()


def finalize():
    nccl_cpp_module.finalize()


def barrier():
    return nccl_cpp_module.barrier()


def send(tensor, rank, tag=0):
    nccl_cpp_module.send(tensor, rank, tag)


def recv(tensor, rank, tag=0):
    nccl_cpp_module.recv(tensor, rank, tag)


def all_reduce(tensor, op=None, group=None, async_op=False):
    return nccl_cpp_module.allreduce(tensor, op, async_op)


def all_to_all_single(outputTensor, inputTensor, async_op=False):
    return nccl_cpp_module.all_to_all(outputTensor, inputTensor, async_op)


def all_to_all(outputTensors, inputTensors):
    return nccl_cpp_module.all_to_all_list(outputTensors, inputTensors)
