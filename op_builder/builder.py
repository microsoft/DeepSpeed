import os
import time
import torch
import importlib
from pathlib import Path
import subprocess
from abc import ABC, abstractmethod

YELLOW = '\033[93m'
END = '\033[0m'
WARNING = f"{YELLOW} [WARNING] {END}"

DEFAULT_TORCH_EXTENSION_PATH = "/tmp/torch_extensions"


class OpBuilder(ABC):
    def __init__(self, name):
        self.name = name
        self.jit_mode = False

    @abstractmethod
    def absolute_name(self):
        '''
        Returns absolute build path for cases where the op is pre-installed, e.g., deepspeed.ops.adam.cpu_adam
        will be installed as something like: deepspeed/ops/adam/cpu_adam.so
        '''
        pass

    @abstractmethod
    def sources(self):
        '''
        Returns list of source files for your op, relative to root of deepspeed package (i.e., DeepSpeed/deepspeed)
        '''
        pass

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

    def is_compatible(self):
        '''
        Check if all non-python dependencies are satisfied to build this op
        '''
        return True

    def command_exists(self, cmd):
        if '|' in cmd:
            cmds = cmd.split("|")
        else:
            cmds = [cmd]
        valid = False
        for cmd in cmds:
            result = subprocess.Popen(f'type {cmd}', stdout=subprocess.PIPE, shell=True)
            valid = valid or result.wait() == 0

        if not valid and len(cmds) > 1:
            print(
                f"{WARNING} {self.name} requires one of the following commands '{cmds}', but it does not exist!"
            )
        elif not valid and len(cmds) == 1:
            print(
                f"{WARNING} {self.name} requires the '{cmd}' command, but it does not exist!"
            )
        return valid

    def warning(self, msg):
        print(f"{WARNING} {msg}")

    def deepspeed_src_path(self, code_path):
        if os.path.isabs(code_path):
            return code_path
        else:
            return os.path.join(Path(__file__).parent.parent.absolute(), code_path)

    def builder(self):
        from torch.utils.cpp_extension import CppExtension
        return CppExtension(name=self.absolute_name(),
                            sources=self.sources(),
                            include_dirs=self.include_paths(),
                            extra_compile_args={'cxx': self.cxx_args()})

    def load(self, verbose=True):
        from ...git_version_info import installed_ops
        if installed_ops[self.name]:
            return importlib.import_module(self.absolute_name())
        else:
            return self.jit_load(verbose)

    def jit_load(self, verbose=True):
        if not self.is_compatible():
            raise RuntimeError(
                f"Unable to JIT load the {self.name} op due to it not being compatible due to hardware/software issue."
            )
        if not self.command_exists('ninja'):
            raise RuntimeError(
                f"Unable to JIT load the {self.name} op due to ninja not being installed."
            )

        self.jit_mode = True
        from torch.utils.cpp_extension import load

        # Ensure directory exists to prevent race condition in some cases
        ext_path = os.path.join(
            os.environ.get('TORCH_EXTENSIONS_DIR',
                           DEFAULT_TORCH_EXTENSION_PATH),
            self.name)
        os.makedirs(ext_path, exist_ok=True)

        start_build = time.time()
        op_module = load(
            name=self.name,
            sources=[self.deepspeed_src_path(path) for path in self.sources()],
            extra_include_paths=[
                self.deepspeed_src_path(path) for path in self.include_paths()
            ],
            extra_cflags=self.cxx_args(),
            extra_cuda_cflags=self.nvcc_args(),
            verbose=verbose)
        build_duration = time.time() - start_build
        if verbose:
            print(f"Time to load {self.name} op: {build_duration} seconds")
        return op_module


class CUDAOpBuilder(OpBuilder):
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

    def version_dependent_macros(self):
        # Fix from apex that might be relevant for us as well, related to https://github.com/NVIDIA/apex/issues/456
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        version_ge_1_1 = []
        if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 0):
            version_ge_1_1 = ['-DVERSION_GE_1_1']
        version_ge_1_3 = []
        if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
            version_ge_1_3 = ['-DVERSION_GE_1_3']
        version_ge_1_5 = []
        if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 4):
            version_ge_1_5 = ['-DVERSION_GE_1_5']
        return version_ge_1_1 + version_ge_1_3 + version_ge_1_5

    def is_compatible(self):
        return super().is_compatible() and self.command_exists('nvcc')

    def builder(self):
        from torch.utils.cpp_extension import CUDAExtension
        return CUDAExtension(name=self.absolute_name(),
                             sources=self.sources(),
                             include_dirs=self.include_paths(),
                             extra_compile_args={
                                 'cxx': self.cxx_args(),
                                 'nvcc': self.nvcc_args()
                             })
