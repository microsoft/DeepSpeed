# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import psutil
import os
import pkgutil
import importlib

from .abstract_accelerator import DeepSpeedAccelerator
# During setup stage torch may not be installed, pass on no torch will
# allow op builder related API to be executed.
try:
    import torch.cpu
except ImportError:
    pass


# accelerator for generic x86 cpus
class CPU_Accelerator(DeepSpeedAccelerator):

    def __init__(self):
        self._name = 'cpu'

    def is_synchronized_device(self):
        return True

    # Device APIs
    def device_name(self, device_index=None):
        if device_index is None:
            return 'cpu'
        return 'cpu:{}'.format(device_index)

    def device(self, device_index=None):
        return torch.device(device_index)

    def set_device(self, device_index):
        torch.cpu.set_device(device_index)

    def current_device(self):
        return torch.cpu.current_device()

    def current_device_name(self):
        return 'cpu:{}'.format(torch.cpu.current_device())

    def device_count(self):
        return torch.cpu.device_count()

    def synchronize(self, device_index=None):
        return torch.cpu.synchronize(device_index)

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        if device_index is None:
            return torch.set_rng_state(new_state)

        return torch.set_rng_state(new_state, device_index)

    def get_rng_state(self, device_index=None):
        if device_index is None:
            return torch.get_rng_state()

        return torch.get_rng_state(device_index)

    def manual_seed(self, seed):
        return torch.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch.manual_seed_all(seed)

    def initial_seed(self, seed):
        return torch.initial_seed(seed)

    def default_generator(self, device_index):
        return torch.default_generators[device_index]

    # Streams/Events
    @property
    def Stream(self):
        return torch.cpu.Stream

    def stream(self, stream):
        return torch.cpu.stream(stream)

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
        mem = self.get_rss()
        mem_stat = {}
        mem_stat['allocated_bytes.all.current'] = mem
        mem_stat['allocated_bytes.all.peak'] = self.max_mem
        return mem_stat

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

    def available_memory(self, device_index=None):
        return psutil.virtual_memory().available

    # Data types
    def is_bf16_supported(self):
        return True

    def is_fp16_supported(self):
        return True

    def supported_dtypes(self):
        supported_dtypes = [torch.float]
        if self.is_fp16_supported():
            supported_dtypes.append(torch.half)
        if self.is_bf16_supported():
            supported_dtypes.append(torch.bfloat16)
        return supported_dtypes

    # Misc
    def amp(self):
        return torch.cpu.amp

    def is_available(self):
        return torch.cpu.is_available()

    def range_push(self, msg):
        return

    def range_pop(self):
        return

    def lazy_call(self, callback):
        return callback()

    def communication_backend_name(self):
        return self._communication_backend_name

    def is_triton_supported(self):
        return False

    # Graph operations
    def create_graph(self):
        return None

    def capture_to_graph(self, graph, pool=None, stream=None):
        from deepspeed.runtime.utils import noop_context
        return noop_context()

    def replay_graph(self, graph):
        return

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

    def pin_memory(self, tensor, align_bytes=1):
        return tensor.pin_memory()

    def is_pinned(self, tensor):
        return tensor.is_pinned()

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith('cpu:'):
            return True
        else:
            return False

    def op_builder_dir(self):
        try:
            # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
            # if successful this also means we're doing a local install and not JIT compile path
            from op_builder import __deepspeed__  # noqa: F401 # type: ignore
            return "op_builder"
        except ImportError:
            return "deepspeed.ops.op_builder"

    # dict that holds class name <--> class type mapping i.e.
    # 'AsyncIOBuilder': <class 'op_builder.async_io.AsyncIOBuilder'>
    # this dict will be filled at init stage
    class_dict = None

    def _lazy_init_class_dict(self):
        if self.class_dict is not None:
            return
        else:
            self.class_dict = {}
            # begin initialize for create_op_builder()
            # put all valid class name <--> class type mapping into class_dict
            op_builder_dir = self.op_builder_dir()
            op_builder_module = importlib.import_module(op_builder_dir)
            op_builder_absolute_path = os.path.dirname(op_builder_module.__file__)
            for _, module_name, _ in pkgutil.iter_modules([op_builder_absolute_path]):
                # avoid self references,
                # skip sub_directories which contains ops for other backend(cpu, npu, etc.).
                if module_name != 'all_ops' and module_name != 'builder' and not os.path.isdir(
                        os.path.join(op_builder_absolute_path, module_name)):
                    module = importlib.import_module("{}.{}".format(op_builder_dir, module_name))
                    for member_name in module.__dir__():
                        if member_name.endswith(
                                'Builder'
                        ) and member_name != "OpBuilder" and member_name != "CUDAOpBuilder" and member_name != "TorchCPUOpBuilder":  # avoid abstract classes
                            if not member_name in self.class_dict:
                                self.class_dict[member_name] = getattr(module, member_name)
            # end initialize for create_op_builder()

    # create an instance of op builder and return, name specified by class_name
    def create_op_builder(self, class_name):
        self._lazy_init_class_dict()
        if class_name in self.class_dict:
            return self.class_dict[class_name]()
        else:
            return None

    # return an op builder class, name specified by class_name
    def get_op_builder(self, class_name):
        self._lazy_init_class_dict()
        if class_name in self.class_dict:
            return self.class_dict[class_name]
        else:
            return None

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension

    def export_envs(self):
        return []
