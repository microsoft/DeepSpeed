# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import time
import importlib
import shutil
from pathlib import Path
import filecmp
import subprocess
from deepspeed.ops.op_builder.builder import OpBuilder, TORCH_MAJOR, TORCH_MINOR

def are_dirs_equal(dir1, dir2):
    # Check if the directories exist
    if not os.path.exists(dir1) or not os.path.exists(dir2):
        return False

    # Compare directory contents
    dir_cmp = filecmp.dircmp(dir1, dir2)

    # Check if the directory structures are the same
    if dir_cmp.left_only or dir_cmp.right_only or dir_cmp.diff_files:
        return False

    # Recursively check subdirectories
    for common_dir in dir_cmp.common_dirs:
        sub_dir1 = os.path.join(dir1, common_dir)
        sub_dir2 = os.path.join(dir2, common_dir)
        if not are_dirs_equal(sub_dir1, sub_dir2):
            return False

    return True

class SYCLOpBuilder(OpBuilder):

    def builder(self):
        try:
            from intel_extension_for_pytorch.xpu.cpp_extension import DPCPPExtension
        except ImportError:
            from intel_extension_for_pytorch.xpu.utils import DPCPPExtension

        print("dpcpp sources = {}".format(self.sources()))
        dpcpp_ext = DPCPPExtension(name=self.absolute_name(),
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
        cxx_flags = [
            '-fsycl', '-fsycl-targets=spir64_gen', '-g', '-gdwarf-4', '-O3', '-std=c++17', '-fPIC', '-DMKL_ILP64',
            '-fno-strict-aliasing'
        ]
        if os.environ.get('USE_MKL_GEMM'):
            cxx_flags.append('-DUSE_MKL_GEMM')
        return cxx_flags

    def extra_ldflags(self):
        return [
            '-fPIC', '-fsycl', '-fsycl-targets=spir64_gen', '-fsycl-max-parallel-link-jobs=8',
            '-Xs "-options -cl-poison-unsupported-fp64-kernels,cl-intel-enable-auto-large-GRF-mode"',
            '-Xs "-device pvc"', '-Wl,-export-dynamic'
        ]

    def fixed_aotflags(self):
        return [
            '-fsycl', '-fsycl-targets=spir64_gen', '-fsycl-max-parallel-link-jobs=8', '-Xs',
            "-options -cl-poison-unsupported-fp64-kernels,cl-intel-enable-auto-large-GRF-mode", '-Xs', "-device pvc"
        ]

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
            raise RuntimeError(f"Unable to JIT load the {self.name} op due to ninja not being installed.")

        self.jit_mode = True
        from intel_extension_for_pytorch.xpu.cpp_extension import load

        start_build = time.time()
        # Recognize relative paths as absolute paths for jit load

        sources = [self.deepspeed_src_path(path) for path in self.sources()]
        extra_include_paths = [self.deepspeed_src_path(path) for path in self.include_paths()]

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


class SYCLAutoOpBuilder(OpBuilder):

    def builder(self):
        try:
            from intel_extension_for_pytorch.xpu.cpp_extension import DPCPPExtension
        except ImportError:
            from intel_extension_for_pytorch.xpu.utils import DPCPPExtension

        sycl_include_paths, sycl_sources = self.update_sycl_code_path()
        print("dpcpp sources = {}".format(sycl_sources))
        dpcpp_ext = DPCPPExtension(name=self.absolute_name(),
                                   sources=self.strip_empty_entries(sycl_sources),
                                   include_dirs=self.strip_empty_entries(sycl_include_paths),
                                   extra_compile_args={
                                       'cxx': self.strip_empty_entries(self.cxx_args()),
                                   },
                                   extra_link_args=self.strip_empty_entries(self.fixed_aotflags()))
        return dpcpp_ext

    def _migrate_sycl_includes(self, ds_root_path, sycl_link_path, extra_args):
        for include_path in self.include_paths():
            ds_inc_path = os.path.join(ds_root_path, include_path)
            build_inc_path = os.path.join(ds_root_path, 'build', include_path)

            sycl_inc_path = os.path.join(sycl_link_path, include_path)
            extra_args += " --extra-arg=" + "\"" + "-I " + f'{build_inc_path}' + "\""

            if os.path.exists(build_inc_path) and are_dirs_equal(build_inc_path, ds_inc_path):
                continue
            elif os.path.exists(build_inc_path):
                shutil.rmtree(build_inc_path)

            os.makedirs(os.path.dirname(build_inc_path), exist_ok=True)
            shutil.copytree(ds_inc_path, build_inc_path)

        return extra_args

    def _copy_sources_to_build(self, ds_root_path):
        for source in self.sources():
            ori_source = os.path.join(ds_root_path, source)
            build_source = os.path.join(ds_root_path, 'build', source)
            if os.path.exists(build_source) and filecmp.cmp(ori_source, build_source):
                continue

            os.makedirs(os.path.dirname(build_source), exist_ok=True)
            shutil.copyfile(ori_source, build_source)

    def _migrate_cuda_file(self, ds_build_path, sycl_link_path, trans_cmd):
        need_post_process = False
        for source in self.sources():
            if '.cu' in source or '.cpp' in source:
                cuda_source = f' {os.path.join(ds_build_path, source)}'
                sycl_kernel_name = source.replace('.cu', '.dp.cpp')
                if os.path.exists(os.path.join(sycl_link_path, sycl_kernel_name)):
                    continue

                need_post_process = True
                process_cmd = trans_cmd + cuda_source
                p = subprocess.Popen(f'{process_cmd}', stdout=subprocess.PIPE, shell=True)
                p.wait()
        return need_post_process

    def sycl_extension(self):
        if self.is_sycl_enabled():
            c2s_cmd = 'c2s'

            # this is necessary for sylomatic
            # TODO: make an assert here
            cuda_inc_path = os.environ.get('CUDA_INC_PATH')
            cuda_inc_flag = " --cuda-include-path=" + f'{cuda_inc_path}'

            # get input and output folder
            ds_root_path = os.getcwd()
            sycl_ds_kernel_path = "deepspeed/third-party"
            sycl_link_path = os.path.join(ds_root_path, sycl_ds_kernel_path)

            extra_args = " --use-experimental-features=local-memory-kernel-scope-allocation "
            extra_args += " --use-experimental-features=free-function-queries "
            extra_args += " --change-cuda-files-extension-only "
            extra_args += " --extra-arg=" + "\"" + "-DBF16_AVAILABLE=1" + "\""

            # copy include dir to build folder and add flags to extra_args
            extra_args = self._migrate_sycl_includes(ds_root_path, sycl_link_path, extra_args)

            from intel_extension_for_pytorch.xpu.cpp_extension import get_pytorch_include_dir

            torch_includes = get_pytorch_include_dir()
            for path in torch_includes:
                extra_args += " --extra-arg=" + "\"" + "-I " + f'{path}' + "\""

            # find Python.h
            import sysconfig
            # Get the path to the include directory
            python_h_dir = sysconfig.get_paths()['include']
            extra_args += " --extra-arg=" + "\"" + "-I " + f'{python_h_dir}' + "\""

            out_root = " --out-root=" + f'{sycl_link_path}'
            in_root = " --in-root=" + f'{ds_root_path}/build'

            sources = ""
            processes_running = []

            # copy source code to build folder
            self._copy_sources_to_build(ds_root_path)

            # check if there is rule.YAML
            rule_file = os.path.join(ds_root_path, 'op_builder/xpu', 'rule.yaml')
            if os.path.exists(rule_file):
                extra_args += " --rule-file " + f'{rule_file}'

            # add pre_process and post_process cmd scripts
            pre_process_script = os.path.join(ds_root_path, 'op_builder/xpu', f'pre_process_{self.NAME}.sh')
            post_process_script = os.path.join(ds_root_path, 'op_builder/xpu', f'post_process_{self.NAME}.sh')

            if os.path.exists(pre_process_script):
                p = subprocess.Popen('source ' + f'{pre_process_script}', stdout=subprocess.PIPE, shell=True)
                p.wait()

            ds_build_path = os.path.join(ds_root_path, 'build')
            trans_cmd = c2s_cmd + cuda_inc_flag + extra_args + in_root + out_root
            need_post_process = self._migrate_cuda_file(ds_build_path, sycl_link_path, trans_cmd)

            if os.path.exists(post_process_script) and need_post_process:
                p = subprocess.Popen('source ' + f'{post_process_script}', stdout=subprocess.PIPE, shell=True)
                p.wait()

    def update_sycl_code_path(self):
        sycl_include_paths = []
        sycl_sources = []
        if self.is_sycl_enabled():

            ds_root_path = Path(__file__).parent.parent.parent.parent.parent.absolute()
            ds_csrc_path = os.path.join(ds_root_path, 'deepspeed/ops')

            sycl_ds_kernel_path = "deepspeed/third-party"
            sycl_link_path = os.path.join(ds_root_path, sycl_ds_kernel_path)

            for include_path in self.include_paths():
                if not os.path.exists(os.path.join(ds_csrc_path, include_path)):
                    ds_root_path = Path(__file__).parent.parent.parent.absolute()
                    sycl_link_path = os.path.join(ds_root_path, sycl_ds_kernel_path)
                sycl_inc_path = os.path.join(sycl_link_path, include_path)
                sycl_include_paths.append(sycl_inc_path)

            for source in self.sources():
                if '.cu' in source or '.cpp' in source:
                    sycl_kernel_name = source.replace('.cu', '.dp.cpp')
                    sycl_sources.append(os.path.join(sycl_link_path, sycl_kernel_name))
        return sycl_include_paths, sycl_sources

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
        cxx_flags = [
            '-fsycl', '-fsycl-targets=spir64_gen', '-g', '-gdwarf-4', '-O3', '-std=c++17', '-fPIC', '-DMKL_ILP64',
            '-fno-strict-aliasing'
        ]
        if os.environ.get('USE_MKL_GEMM'):
            cxx_flags.append('-DUSE_MKL_GEMM')
        return cxx_flags

    def extra_ldflags(self):
        return [
            '-fPIC', '-fsycl', '-fsycl-targets=spir64_gen', '-fsycl-max-parallel-link-jobs=8',
            '-Xs "-options -cl-poison-unsupported-fp64-kernels,cl-intel-enable-auto-large-GRF-mode"',
            '-Xs "-device pvc"', '-Wl,-export-dynamic'
        ]

    def fixed_aotflags(self):
        return [
            '-fsycl', '-fsycl-targets=spir64_gen', '-fsycl-max-parallel-link-jobs=8', '-Xs',
            "-options -cl-poison-unsupported-fp64-kernels,cl-intel-enable-auto-large-GRF-mode", '-Xs', "-device pvc"
        ]

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
            raise RuntimeError(f"Unable to JIT load the {self.name} op due to ninja not being installed.")

        self.jit_mode = True
        from intel_extension_for_pytorch.xpu.cpp_extension import load

        self.enable_bf16 = True
        cxx_args = self.strip_empty_entries(self.cxx_args())
        if isinstance(self, SYCLAutoOpBuilder):
            if self.enable_bf16:
                cxx_args.append("-DBF16_AVAILABLE")

        start_build = time.time()
        # Recognize relative paths as absolute paths for jit load

        extra_include_paths, sources = self.update_sycl_code_path()
        # sources = [self.deepspeed_src_path(path) for path in self.sources()]
        # extra_include_paths = [self.deepspeed_src_path(path) for path in self.include_paths()]

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
            extra_cflags=cxx_args,
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


