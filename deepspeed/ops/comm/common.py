from enum import Enum
import torch
from ..op_builder import CommBuilder
import importlib

common_cpp_module = None


def build_op():
    global common_cpp_module
    builder = CommBuilder()
    try:
        common_cpp_module = builder.load()
        print(f'DeepSpeed {builder.absolute_name()} built successfully')
        return common_cpp_module
    except Exception as inst:
        # if comm cannot be built, use torch.dist.
        print(f"Failed to build {builder.absolute_name()}. Full error: {inst}")
        exit(0)


#if common_cpp_module is None:
#    common_cpp_module = CommBuilder().load()
#SUM = common_cpp_module.SUM

#class ReduceOp(Enum):
#    global common_cpp_module
#    if common_cpp_module is None:
#        common_cpp_module = CommBuilder().load()
#    print(dir(common_cpp_module))
#    #if common_cpp_module is not None:
#    SUM = common_cpp_module.ReduceOp.SUM
#    AVG = common_cpp_module.ReduceOp.AVG
#    PRODUCT = common_cpp_module.ReduceOp.PRODUCT
#    MIN = common_cpp_module.ReduceOp.MIN
#    MAX = common_cpp_module.ReduceOp.MAX
#    BAND = common_cpp_module.ReduceOp.BAND
#    BOR = common_cpp_module.ReduceOp.BOR
#    BXOR = common_cpp_module.ReduceOp.BXOR
#    UNUSED = common_cpp_module.ReduceOp.UNUSED
