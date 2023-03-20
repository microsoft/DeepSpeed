'''Copyright The Microsoft DeepSpeed Team'''

from .builder import CPUOpBuilder, cpu_kernel_path, cpu_kernel_include


class CCLCommBuilder(CPUOpBuilder):
    BUILD_VAR = "DS_BUILD_CCL_COMM"
    NAME = "deepspeed_ccl_comm"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.comm.{self.NAME}_op'

    def sources(self):
        return [cpu_kernel_path('csrc/comm/ccl.cpp')]
        #return [cpu_kernel_path('csrc/foo.c')]

    def include_paths(self):
        includes = [cpu_kernel_include('csrc/includes')]
        return includes

    #if 0
    def is_compatible(self, verbose=True):
        # TODO: add soft compatibility check for private binary release.
        #  a soft check, as in we know it can be trivially changed.
        return super().is_compatible(verbose)

    def extra_ldflags(self):
        return ['-lccl']

    #endif
