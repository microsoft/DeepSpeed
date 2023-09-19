# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import SYCLOpBuilder


class TransformerBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_TRANSFORMER"
    NAME = "transformer"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.transformer.{self.NAME}_op'

    def extra_ldflags(self):
        return super().extra_ldflags()

    def sources(self):
        return [
            'csrc/xpu/transformer/onednn_wrappers.dp.cpp', 'csrc/xpu/transformer/ds_transformer_sycl.dp.cpp',
            'csrc/xpu/transformer/onemkl_wrappers.dp.cpp', 'csrc/xpu/transformer/transform_kernels.dp.cpp',
            'csrc/xpu/transformer/ds_gelu_sycl.dp.cpp', 'csrc/xpu/transformer/gelu_kernels.dp.cpp',
            'csrc/xpu/transformer/ds_dropout_sycl.dp.cpp', 'csrc/xpu/transformer/dropout_kernels.dp.cpp',
            'csrc/xpu/transformer/ds_feedforward_sycl.dp.cpp', 'csrc/xpu/transformer/ds_layer_reorder_sycl.dp.cpp',
            'csrc/xpu/transformer/ds_normalize_sycl.dp.cpp', 'csrc/xpu/transformer/normalize_kernels.dp.cpp',
            'csrc/xpu/transformer/ds_softmax_sycl.dp.cpp', 'csrc/xpu/transformer/softmax_kernels.dp.cpp',
            'csrc/xpu/transformer/ds_stridedbatchgemm_sycl.dp.cpp', 'csrc/xpu/transformer/general_kernels.dp.cpp'
        ]

    def include_paths(self):
        includes = ['csrc/xpu/includes', 'csrc/includes']
        return includes
