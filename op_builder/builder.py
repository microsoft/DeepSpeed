import os
import torch
import importlib
from pathlib import Path
import subprocess
from abc import ABC


def command_exists(cmd):
    if '|' in cmd:
        cmds = cmd.split("|")
    else:
        cmds = [cmd]
    valid = False
    for cmd in cmds:
        result = subprocess.Popen(f'type {cmd}', stdout=subprocess.PIPE, shell=True)
        valid = valid or result.wait() == 0
    return valid


class OpBuilder(ABC):
    def __init__(self, name, name_prefix='', cuda=True):
        self.name = name
        self.name_prefix = name_prefix
        self.jit_mode = False
        self.cuda = cuda

    def absolute_name(self):
        return self.name_prefix + self.name

    def sources(self):
        '''
        Returns list of source files for your op, relative to root of deepspeed package (i.e., DeepSpeed/deepspeed)
        '''
        raise NotImplemented

    def include_paths(self):
        '''
        Returns list of include paths, relative to root of deepspeed package (i.e., DeepSpeed/deepspeed)
        '''
        return []

    def nvcc_args(self):
        '''
        Returns optional list of compiler flags to forward to nvcc when building CUDA sources
        '''
        return []

    def cxx_args(self):
        '''
        Returns optional list of compiler flags to forward to the build
        '''
        return []

    def verbose(self):
        '''
        Verbose logging of build process in JIT mode, on by default
        '''
        return True

    def is_compatible(self):
        '''
        Check if all non-python dependencies are satisfied to build this op
        '''
        return True

    def compute_capability_args(self, cross_compile_archs=['52', '60', '61', '70']):
        args = []
        if self.jit_mode:
            # Compile for underlying architecture since we know it at runtime
            CC_MAJOR, CC_MINOR = torch.cuda.get_device_capability()
            compute_capability = f"{CC_MAJOR}{CC_MINOR}"
            args.append('-gencode')
            args.append(
                f'arch=compute_{compute_capability},code=compute_{compute_capability}')
        else:
            # Cross-compile mode, compile for various architectures
            for compute_capability in cross_compile_archs:
                args.append('-gencode')
                args.append(
                    f'arch=compute_{compute_capability},code=compute_{compute_capability}'
                )
        return args

    def deepspeed_src_path(self, code_path):
        if os.path.isabs(code_path):
            return code_path
        else:
            return os.path.join(Path(__file__).parent.parent.absolute(), code_path)

    def builder(self):
        if self.cuda:
            return self.cuda_extension_builder()
        else:
            return self.cpp_extension_builder()

    def cuda_extension_builder(self):
        from torch.utils.cpp_extension import CUDAExtension
        return CUDAExtension(name=self.absolute_name(),
                             sources=self.sources(),
                             include_dirs=self.include_paths(),
                             extra_compile_args={
                                 'cxx': self.cxx_args(),
                                 'nvcc': self.nvcc_args()
                             })

    def cpp_extension_builder(self):
        from torch.utils.cpp_extension import CppExtension
        return CppExtension(name=self.absolute_name(),
                            sources=self.sources(),
                            include_dirs=self.include_paths(),
                            extra_compile_args={'cxx': self.cxx_args()})

    def load(self):
        from ...git_version_info import installed_ops
        if installed_ops[self.name]:
            return importlib.import_module(self.absolute_name())
        else:
            return self.jit_load()

    def jit_load(self):
        if not self.is_compatible():
            raise RuntimeError(
                f"Unable to JIT load the {self.name} op due to it not being compatible with the underlying hardware."
            )
        self.jit_mode = True
        from torch.utils.cpp_extension import load

        # Ensure directory exists to prevent race condition in some cases
        ext_path = os.path.join(os.environ['TORCH_EXTENSIONS_DIR'], self.name)
        os.makedirs(ext_path, exist_ok=True)

        return load(name=self.name,
                    sources=[self.deepspeed_src_path(path) for path in self.sources()],
                    extra_include_paths=[
                        self.deepspeed_src_path(path) for path in self.include_paths()
                    ],
                    extra_cflags=self.cxx_args(),
                    extra_cuda_cflags=self.nvcc_args(),
                    verbose=self.verbose())
