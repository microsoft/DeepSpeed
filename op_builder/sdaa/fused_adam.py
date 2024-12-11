# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import SDAAOpBuilder

try:
    import torch
except ImportError as e:
    pass


class SDAAFusedAdam:

    @staticmethod
    def multi_tensor_adam(chunk_size, noop_flag_buffer, tensor_lists, lr, beta1, beta2, epsilon, step, adam_w_mode,
                          bias_correction, weight_decay, *args):
        g_tensor_lis, p_tensor_lis, m_tensor_lis, v_tensor_lis = tensor_lists
        torch.ops.sdaa.fused_adam(g_tensor_lis, p_tensor_lis, m_tensor_lis, v_tensor_lis,
                                [], beta1, beta2, epsilon, lr, weight_decay, adam_w_mode, step, bias_correction)


class FusedAdamBuilder(SDAAOpBuilder):
    BUILD_VAR = "DS_BUILD_FUSED_ADAM"
    NAME = "fused_adam"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.adam.{self.NAME}_op'

    def sources(self):
        return []

    def include_paths(self):
        return []

    def load(self, verbose=True):
        return SDAAFusedAdam
