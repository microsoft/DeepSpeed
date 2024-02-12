# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .cuda_accelerator import CUDA_Accelerator
# During setup stage torch may not be installed, pass on no torch will
# allow op builder related API to be executed.
try:
    import torch.cuda
except ImportError:
    pass

# Delay import pynvml to avoid import error when CUDA is not available
pynvml = None


class HIP_Accelerator(CUDA_Accelerator):

    def __init__(self):
        self._name = 'hip'
        self._communication_backend_name = 'nccl'
        if pynvml is None:
            self._init_pynvml()

    def _init_pynvml(self):
        global pynvml
        try:
            import pynvml
        except ImportError:
            return
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError:
            pynvml = None
            return

    def is_triton_supported(self):
        return False
