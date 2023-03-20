"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
import os
import shutil
from pathlib import Path
from deepspeed.ops.op_builder.builder import OpBuilder


class CPUOpBuilder(OpBuilder):
    def builder(self):
        from torch.utils.cpp_extension import CppExtension as ExtensionBuilder

        compile_args = {'cxx': self.strip_empty_entries(self.cxx_args())}

        cpp_ext = ExtensionBuilder(
            name=self.absolute_name(),
            sources=self.strip_empty_entries(self.sources()),
            include_dirs=self.strip_empty_entries(self.include_paths()),
            libraries=self.strip_empty_entries(self.libraries_args()),
            extra_compile_args=compile_args)

        return cpp_ext

    def cxx_args(self):
        return ['-O3', '-std=c++14', '-g', '-Wno-reorder']

    def libraries_args(self):
        return []


def cpu_kernel_path(code_path):
    # Always return a path like "CPU_KERNEL_PATH/..."
    CPU_KERNEL_PATH = "third-party"
    abs_source_path = os.path.join(Path(__file__).parent.absolute(), code_path)
    rel_target_path = os.path.join(CPU_KERNEL_PATH, code_path)

    # Jit_load mode require absolute path. Use abs path for copy
    # To get the absolute path of deepspeed
    # We use a non-abstract builder class instance to call deepspeed_src_path()
    # InferenceBuilder is one of such class instance
    from .transformer_inference import InferenceBuilder
    abs_target_path = InferenceBuilder().deepspeed_src_path(rel_target_path)

    cpu_link_path = os.path.join(
        os.path.dirname(InferenceBuilder().deepspeed_src_path("")),
        CPU_KERNEL_PATH)
    if not os.path.exists(cpu_link_path):
        # Create directory and link for cpu kernel:
        #   deepspeed/ops/CPU_KERNEL_PATH-->../../CPU_KERNEL_PATH
        cpu_dir_path = os.path.join(os.path.dirname(cpu_link_path),
                                    "../../" + CPU_KERNEL_PATH)

        os.mkdir(cpu_dir_path)
        os.symlink("../../" + CPU_KERNEL_PATH, cpu_link_path, True)
        print("Create directory and link for cpu kernel:{}-->{}".format(
            cpu_link_path,
            cpu_dir_path))

    import filecmp
    if (os.path.exists(abs_target_path) and filecmp.cmp(abs_target_path,
                                                        abs_source_path)):
        print("skip copy, {} and {} have the same content".format(
            abs_source_path,
            abs_target_path))
        return rel_target_path

    print("Copying CPU kernel file from {} to {}".format(abs_source_path,
                                                         abs_target_path))
    os.makedirs(os.path.dirname(abs_target_path), exist_ok=True)
    shutil.copyfile(abs_source_path, abs_target_path)

    # Prebuild install mode require paths relative to the setup.py directory. Use the relative path.
    return rel_target_path


def cpu_kernel_include(code_path):
    import intel_extension_for_pytorch  # noqa: F401
    abs_path = os.path.join(Path(__file__).parent.absolute(), code_path)
    return abs_path
