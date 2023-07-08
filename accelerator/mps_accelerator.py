# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import pkgutil
import importlib

from .abstract_accelerator import DeepSpeedAccelerator
# During setup stage torch may not be installed, pass on no torch will
# allow op builder related API to be executed.
try:
    import torch.mps
except ImportError:
    pass

class MPS_Accelerator(DeepSpeedAccelerator):

    def __init__(self):
        self._name = 'mps'
        self._communication_backend_name = 'mccl'

    def is_synchronized_device(self):
        return False

    def device_name(self):
        return 'mps'

    def get_rng_state(self):
        return torch.mps.get_rng_state()

    def manual_seed(self, seed):
        torch.mps.manual_seed(seed)

    def seed(self):
        return torch.mps.seed()

    def set_rng_state(self, new_state):
        torch.mps.set_rng_state(new_state)

    def synchronize(self):
        return torch.mps.synchronize()

    def empty_cache(self):
        return torch.mps.empty_cache()

    def set_per_process_memory_fraction(self, fraction):
        return torch.mps.set_per_process_memory_fraction(fraction)
    
    def current_allocated_memory(self):
        return torch.mps.current_allocated_memory()

    def driver_allocated_memory(self):
        return torch.mps.driver_allocated_memory()

    def is_available(self):
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    def random(self):
        return torch.random

    def communication_backend_name(self):
        return self._communication_backend_name

    def pin_memory(self, tensor):
        return tensor.pin_memory()

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith('mps'):
            return True
        else:
            return False


