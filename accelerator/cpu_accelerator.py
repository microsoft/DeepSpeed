'''Copyright The Microsoft DeepSpeed Team'''

import torch
from deepspeed.accelerator.abstract_accelerator import DeepSpeedAccelerator
import oneccl_bindings_for_pytorch  # noqa: F401
import psutil


# accelerator for Intel CPU
class CPU_Accelerator(DeepSpeedAccelerator):

    def __init__(self):
        self._name = 'cpu'
        self._communication_backend_name = 'ccl'
        self.max_mem = psutil.Process().memory_info().rss

        def _check_and_mapping_mpich_env():
            # Workaround when using Intel MPI launcher
            # Model can check "RANK" and "WORLD_SIZE" for rank and world size
            import os
            if "RANK" not in os.environ and "PMI_RANK" in os.environ:
                os.environ['RANK'] = os.environ.get('PMI_RANK')
                print("mapping environment variable PMI_RANK to RANK")
            if "WORLD_SIZE" not in os.environ and "PMI_SIZE" in os.environ:
                os.environ['WORLD_SIZE'] = os.environ.get('PMI_SIZE')
                print("mapping environment variable PMI_SIZE to WORLD_SIZE")
            if "LOCAL_RANK" not in os.environ and "MPI_LOCALRANKID" in os.environ:
                os.environ['LOCAL_RANK'] = os.environ.get('MPI_LOCALRANKID')
                print("mapping environment variable MPI_LOCALRANKID to LOCAL_RANK")

        _check_and_mapping_mpich_env()

    def is_synchronized_device(self):
        return True

    # Device APIs
    def device_name(self, device_index=None):
        return 'cpu'

    def device(self, device_index=None):
        return None

    def set_device(self, device_index):
        return

    def current_device(self):
        return 0

    def current_device_name(self):
        return 'cpu'

    def device_count(self):
        return 1

    def synchronize(self, device_index=None):
        return

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        if device_index == None:
            return torch.set_rng_state(new_state)
        return torch.set_rng_state(new_state, device_index)

    def get_rng_state(self, device_index=None):
        return torch.get_rng_state()

    def manual_seed(self, seed):
        return torch.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch.manual_seed(seed)

    def initial_seed(self, seed):
        return torch.initial_seed(seed)

    def default_generator(self, device_index):
        return torch.default_generator

    # Streams/Events
    @property
    def Stream(self):
        return None

    def stream(self, stream):
        from deepspeed.runtime.utils import noop_decorator
        return noop_decorator

    def current_stream(self, device_index=None):
        return None

    def default_stream(self, device_index=None):
        return None

    @property
    def Event(self):
        return None

    # Memory management
    def empty_cache(self):
        return

    def get_rss(self):
        mem = psutil.Process().memory_info().rss
        if mem > self.max_mem:
            self.max_mem = mem
        return mem

    def reset_rss(self):
        mem = psutil.Process().memory_info().rss
        self.max_mem = mem
        return mem

    def memory_allocated(self, device_index=None):
        return self.get_rss()

    def max_memory_allocated(self, device_index=None):
        self.get_rss()
        return self.max_mem

    def reset_max_memory_allocated(self, device_index=None):
        self.reset_rss()
        return

    def memory_cached(self, device_index=None):
        return self.get_rss()

    def max_memory_cached(self, device_index=None):
        self.get_rss()
        return self.max_mem

    def reset_max_memory_cached(self, device_index=None):
        self.reset_rss()
        return

    def memory_stats(self, device_index=None):
        return self.get_rss()

    def reset_peak_memory_stats(self, device_index=None):
        self.reset_rss()
        return

    def memory_reserved(self, device_index=None):
        return self.get_rss()

    def max_memory_reserved(self, device_index=None):
        self.get_rss()
        return self.max_mem

    def total_memory(self, device_index=None):
        return psutil.virtual_memory().total

    # Misc
    def amp(self):
        return torch.cpu.amp

    def is_available(self):
        return True

    def range_push(self, msg):
        # TODO itt is currently not supported yet
        # return torch.profiler.itt.range_push(msg)
        return

    def range_pop(self):
        # TODO itt is currently not supported yet
        # return torch.profiler.itt.range_pop()
        return

    def lazy_call(self, callback):
        return callback()

    def communication_backend_name(self):
        return self._communication_backend_name

    # Data types
    def is_bf16_supported(self):
        return True

    def is_fp16_supported(self):
        return True

    # Tensor operations

    @property
    def BFloat16Tensor(self):
        return torch.BFloat16Tensor

    @property
    def ByteTensor(self):
        return torch.ByteTensor

    @property
    def DoubleTensor(self):
        return torch.DoubleTensor

    @property
    def FloatTensor(self):
        return torch.FloatTensor

    @property
    def HalfTensor(self):
        return torch.HalfTensor

    @property
    def IntTensor(self):
        return torch.IntTensor

    @property
    def LongTensor(self):
        return torch.LongTensor

    def pin_memory(self, tensor):
        return tensor.pin_memory(device=self.current_device_name())

    def op_builder_dir(self):
        try:
            # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
            # if successful this also means we're doing a local install and not JIT compile path
            from op_builder import __deepspeed__  # noqa: F401
            return "op_builder.cpu"
        except ImportError:
            return "deepspeed.ops.op_builder.cpu"

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith('cpu'):
            return True
        else:
            return False

    # create an instance of op builder and return, name specified by class_name
    def create_op_builder(self, op_name):
        builder_class = self.get_op_builder(op_name)
        if builder_class != None:
            return builder_class()
        return None

    # return an op builder class, name specified by class_name
    def get_op_builder(self, class_name):
        try:
            # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
            # if successful this also means we're doing a local install and not JIT compile path
            from op_builder import __deepspeed__  # noqa: F401
            from op_builder.cpu import InferenceBuilder, CCLCommBuilder, NotImplementedBuilder
        except ImportError:
            from deepspeed.ops.op_builder.cpu import InferenceBuilder, CCLCommBuilder, NotImplementedBuilder

        if class_name == "InferenceBuilder":
            return InferenceBuilder
        if class_name == "CCLCommBuilder":
            return CCLCommBuilder
        else:
            # return a NotImplementedBuilder to avoid get NoneType[Name] in unit tests
            return NotImplementedBuilder

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension
