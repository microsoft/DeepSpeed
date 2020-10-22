import torch
import warnings
from .builder import OpBuilder, command_exists


class SparseAttnBuilder(OpBuilder):
    BUILD_VAR = "DS_BUILD_SPARSE_ATTN"
    NAME = "sparse_attn"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.sparse_attention.{self.NAME}'

    def sources(self):
        return ['csrc/sparse_attention/utils.cpp']

    def cxx_args(self):
        return ['-O2', '-fopenmp']

    def is_compatible(self):
        # Check to see if llvm and cmake are installed since they are dependencies
        required_commands = ['llvm-config|llvm-config-9', 'cmake']
        command_status = list(map(command_exists, required_commands))
        compatible = False
        if not all(command_status):
            zipped_status = list(zip(required_commands, command_status))
            warnings.warn(
                f'Missing non-python requirements, please install the missing packages: {zipped_status}'
            )
            warnings.warn(
                'Skipping sparse attention installation due to missing required packages'
            )

        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        if TORCH_MAJOR == 1 and TORCH_MINOR >= 5:
            compatible = True
        else:
            warnings.warn(
                f'Sparse attention requires a torch version >= 1.5 but detected {TORCH_MAJOR}.{TORCH_MINOR}'
            )

        return super().is_compatible() and compatible
