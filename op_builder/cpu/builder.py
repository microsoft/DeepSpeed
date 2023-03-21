"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
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
