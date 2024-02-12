# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .cuda_accelerator import CUDA_Accelerator


class HIP_Accelerator(CUDA_Accelerator):

    def __init__(self):
        self._name = 'hip'
        self._communication_backend_name = 'nccl'

    def is_triton_supported(self):
        return False
