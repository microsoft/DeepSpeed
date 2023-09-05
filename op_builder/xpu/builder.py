"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
import os
import time
import importlib
import shutil
from pathlib import Path
from deepspeed.ops.op_builder.builder import OpBuilder, TORCH_MAJOR, TORCH_MINOR

class SYCLOpBuilder(OpBuilder):
    def builder(self):
        try:
            from intel_extension_for_pytorch.xpu.cpp_extension import DPCPPExtension
        except ImportError:
            from intel_extension_for_pytorch.xpu.utils import DPCPPExtension

        print("dpcpp sources = {}".format(self.sources()))
        dpcpp_ext = DPCPPExtension(
            name=self.absolute_name(),
            sources=self.strip_empty_entries(self.sources()),
            include_dirs=self.strip_empty_entries(self.include_paths()),
            extra_compile_args={
                'cxx': self.strip_empty_entries(self.cxx_args()),
            },
            extra_link_args=self.strip_empty_entries(self.fixed_aotflags()))
        return dpcpp_ext

    def version_dependent_macros(self):
        # Fix from apex that might be relevant for us as well, related to https://github.com/NVIDIA/apex/issues/456
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

    def cxx_args(self):
        cxx_flags = ['-fsycl', '-fsycl-targets=spir64_gen', '-g', '-gdwarf-4', '-O3', '-std=c++17', '-fPIC', '-DMKL_ILP64', '-fno-strict-aliasing']
        if os.environ.get('USE_MKL_GEMM'):
            cxx_flags.append('-DUSE_MKL_GEMM')
        return cxx_flags

    def extra_ldflags(self):
        return ['-fPIC', '-fsycl', '-fsycl-targets=spir64_gen', '-fsycl-max-parallel-link-jobs=8', '-Xs "-options -cl-poison-unsupported-fp64-kernels,cl-intel-enable-auto-large-GRF-mode"', '-Xs "-device pvc"', '-Wl,-export-dynamic']

    def fixed_aotflags(self):
        return ['-fsycl', '-fsycl-targets=spir64_gen', '-fsycl-max-parallel-link-jobs=8', '-Xs', "-options -cl-poison-unsupported-fp64-kernels,cl-intel-enable-auto-large-GRF-mode", '-Xs', "-device pvc"]

    def load(self, verbose=True):
        from deepspeed.git_version_info import installed_ops, torch_info  # noqa: F401
        if installed_ops.get(self.name, False):
            return importlib.import_module(self.absolute_name())
        else:
            return self.jit_load(verbose)

    def jit_load(self, verbose=True):
        if not self.is_compatible(verbose):
            raise RuntimeError(
                f"Unable to JIT load the {self.name} op due to it not being compatible due to hardware/software issue. {self.error_log}"
            )
        try:
            import ninja  # noqa: F401
        except ImportError:
            raise RuntimeError(
                f"Unable to JIT load the {self.name} op due to ninja not being installed."
            )

        self.jit_mode = True
        from intel_extension_for_pytorch.xpu.cpp_extension import load

        start_build = time.time()
        # Recognize relative paths as absolute paths for jit load

        sources = [self.deepspeed_src_path(path) for path in self.sources()]
        extra_include_paths = [
            self.deepspeed_src_path(path) for path in self.include_paths()
        ]

        # Torch will try and apply whatever CCs are in the arch list at compile time,
        # we have already set the intended targets ourselves we know that will be
        # needed at runtime. This prevents CC collisions such as multiple __half
        # implementations. Stash arch list to reset after build.
        '''
        torch_arch_list = None
        if "TORCH_CUDA_ARCH_LIST" in os.environ:
            torch_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
            os.environ["TORCH_CUDA_ARCH_LIST"] = ""
        '''

        op_module = load(
            name=self.name,
            sources=self.strip_empty_entries(sources),
            extra_include_paths=self.strip_empty_entries(extra_include_paths),
            extra_cflags=self.strip_empty_entries(self.cxx_args()),
            # extra_cuda_cflags=self.strip_empty_entries(self.nvcc_args()),
            extra_ldflags=self.strip_empty_entries(self.extra_ldflags()),
            verbose=verbose)

        build_duration = time.time() - start_build
        if verbose:
            print(f"Time to load {self.name} op: {build_duration} seconds")
        '''
        # Reset arch list so we are not silently removing it for other possible use cases
        if torch_arch_list:
            os.environ["TORCH_CUDA_ARCH_LIST"] = torch_arch_list
        '''
        return op_module


def sycl_kernel_path(code_path):
    # Always return a path like "SYCL_KERNEL_PATH/..."
    SYCL_KERNEL_PATH = "third-party"
    abs_source_path = os.path.join(Path(__file__).parent.absolute(), code_path)
    rel_target_path = os.path.join(SYCL_KERNEL_PATH, code_path)

    # Jit_load mode require absolute path. Use abs path for copy
    # To get the absolute path of deepspeed
    # We use a non-abstract builder class instance to call deepspeed_src_path()
    # FusedAdamBuilder is one of such class instance
    from .fused_adam import FusedAdamBuilder
    abs_target_path = FusedAdamBuilder().deepspeed_src_path(rel_target_path)

    sycl_link_path = os.path.join(
        os.path.dirname(FusedAdamBuilder().deepspeed_src_path("")),
        SYCL_KERNEL_PATH)
    if not os.path.exists(sycl_link_path):
        # Create directory and link for sycl kernel:
        #   deepspeed/ops/SYCL_KERNEL_PATH-->../../SYCL_KERNEL_PATH
        sycl_dir_path = os.path.join(os.path.dirname(sycl_link_path),
                                     "../../" + SYCL_KERNEL_PATH)

        os.makedirs(sycl_dir_path, exist_ok=True)
        os.symlink("../../" + SYCL_KERNEL_PATH, sycl_link_path, True)
        print("Create directory and link for sycl kernel:{}-->{}".format(
            sycl_link_path,
            sycl_dir_path))

    import filecmp
    if (os.path.exists(abs_target_path) and filecmp.cmp(abs_target_path,
                                                        abs_source_path)):
        print("skip copy, {} and {} have the same content".format(
            abs_source_path,
            abs_target_path))
        return rel_target_path

    print("Copying SYCL kernel file from {} to {}".format(abs_source_path,
                                                          abs_target_path))
    os.makedirs(os.path.dirname(abs_target_path), exist_ok=True)
    shutil.copyfile(abs_source_path, abs_target_path)

    # Prebuild install mode require paths relative to the setup.py directory. Use the relative path.
    return rel_target_path


def sycl_kernel_include(code_path):
    import intel_extension_for_pytorch  # noqa: F401
    abs_path = os.path.join(Path(__file__).parent.absolute(), code_path)
    return abs_path
