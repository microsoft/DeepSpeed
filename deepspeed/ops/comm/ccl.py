#import torch
from deepspeed.ops import op_builder
from deepspeed.accelerator import get_accelerator

ccl_cpp_module = None


def build_ccl_op():
    global ccl_cpp_module
    builder = get_accelerator().create_op_builder("CCLCommBuilder")
    ccl_cpp_module = builder.load()
    print(f'DeepSpeed {builder.absolute_name()} built successfully')
    return ccl_cpp_module


def get_ccl_id():
    return ccl_cpp_module.getCclId()


def initialize():
    ccl_cpp_module.initialize()


def finalize():
    ccl_cpp_module.finalize()


def barrier():
    return ccl_cpp_module.barrier()


def all_reduce(tensor, op=None, group=None, async_op=False):
    return ccl_cpp_module.allreduce(tensor, op, async_op)


#def send(tensor, rank, tag=0):
#    ccl_cpp_module.send(tensor, rank, tag)

#def recv(tensor, rank, tag=0):
#    ccl_cpp_module.recv(tensor, rank, tag)

#def all_to_all_single(outputTensor, inputTensor, async_op=False):
#    return ccl_cpp_module.all_to_all(outputTensor, inputTensor, async_op)

#def all_to_all(outputTensors, inputTensors):
#    return ccl_cpp_module.all_to_all_list(outputTensors, inputTensors)
