#NOTE: This work is still experimental. Please do not run custom NCCL/MPI deepspeed comm backends until they are fully supported.

import torch
from .builder import CUDAOpBuilder

MV2 = f'/opt/mvapich2/gdr/2.3.6/no-mpittool/no-openacc/cuda11.3/mofed5.4/mpirun/gnu9.3.0'
OMPI = f'/usr/local/mpi'

MPI_INCL = OMPI + '/include'
MPI_LIB = '-L' + OMPI + '/lib -lmpi'

CUDA = '/usr/local/cuda'
CUDA_INCL = CUDA + '/include'
CUDA_LIB = '-L' + CUDA + '/lib -lm -lcuda -lcudart'


class CommBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_COMM"
    NAME = "deepspeed_comm"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.comm.{self.NAME}_op'

    def sources(self):
        return ['csrc/comm/nccl.cpp']

    def include_paths(self):
        return [MPI_INCL, CUDA_INCL]

    def is_compatible(self, verbose=True):
        # TODO: add soft compatibility check for private binary release.
        #  a soft check, as in we know it can be trivially changed.
        return super().is_compatible(verbose)
