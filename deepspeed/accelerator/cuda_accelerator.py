from deepspeed.accelerator.abstract_accelerator import DeepSpeedAccelerator
try:
    import torch.cuda
except ImportError:
    pass


class CUDA_Accelerator(DeepSpeedAccelerator):
    def __init__(self):
        self._name = 'cuda'
        self._communication_backend_name = 'nccl'

    # Device APIs
    def device_name(self, device_index=None):
        if device_index == None:
            return 'cuda'
        return 'cuda:{}'.format(device_index)

    def device(self, device_index=None):
        return torch.cuda.device(device_index)

    def set_device(self, device_index):
        torch.cuda.set_device(device_index)

    def current_device(self):
        return torch.cuda.current_device()

    def current_device_name(self):
        return 'cuda:{}'.format(torch.cuda.current_device())

    def device_count(self):
        return torch.cuda.device_count()

    def synchronize(self, device_index=None):
        return torch.cuda.synchronize(device_index)

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        if device_index is None:
            return torch.cuda.set_rng_state(new_state)

        return torch.cuda.set_rng_state(new_state, device_index)

    def get_rng_state(self, device_index=None):
        if device_index is None:
            return torch.cuda.get_rng_state()

        return torch.cuda.get_rng_state(device_index)

    def manual_seed(self, seed):
        return torch.cuda.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch.cuda.manual_seed_all(seed)

    def initial_seed(self, seed):
        return torch.cuda.initial_seed(seed)

    def default_generator(self, device_index):
        return torch.cuda.default_generators[device_index]

    # Streams/Events
    def Stream(self, device=None, priority=0, **kwargs):
        return torch.cuda.Stream(device, priority, **kwargs)

    def StreamContext(self, stream):
        return torch.cuda.StreamContext(stream)

    def stream(self, stream):
        return torch.cuda.stream(stream)

    def current_stream(self, device_index=None):
        return torch.cuda.current_stream(device_index)

    def default_stream(self, device_index=None):
        return torch.cuda.default_stream(device_index)

    def Event(self, **kwargs):
        return torch.cuda.Event(**kwargs)

    # Memory management
    def empty_cache(self):
        return torch.cuda.empty_cache()

    def memory_allocated(self, device_index=None):
        return torch.cuda.memory_allocated(device_index)

    def max_memory_allocated(self, device_index=None):
        return torch.cuda.max_memory_allocated(device_index)

    def reset_max_memory_allocated(self, device_index=None):
        return torch.cuda.reset_max_memory_allocated(device_index)

    def memory_cached(self, device_index=None):
        return torch.cuda.memory_cached(device_index)

    def max_memory_cached(self, device_index=None):
        return torch.cuda.max_memory_cached(device_index)

    def reset_max_memory_cached(self, device_index=None):
        return torch.cuda.reset_max_memory_cached(device_index)

    def memory_stats(self, device_index=None):
        if hasattr(torch.cuda, 'memory_stats'):
            return torch.cuda.memory_stats(device_index)

    def reset_peak_memory_stats(self, device_index=None):
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            return torch.cuda.reset_peak_memory_stats(device_index)

    def memory_reserved(self, device_index=None):
        if hasattr(torch.cuda, 'memory_reserved'):
            return torch.cuda.memory_reserved(device_index)

    def max_memory_reserved(self, device_index=None):
        if hasattr(torch.cuda, 'max_memory_reserved'):
            return torch.cuda.max_memory_reserved(device_index)

    def total_memory(self, device_index=None):
        return torch.cuda.get_device_properties(device_index).total_memory

    # Data types
    def is_bf16_supported(self):
        return torch.cuda.is_bf16_supported()

    def is_fp16_supported(self):
        major, _ = torch.cuda.get_device_capability()
        if major >= 7:
            return True
        else:
            return False

    # Misc
    def amp(self):
        if hasattr(torch.cuda, 'amp'):
            return torch.cuda.amp
        return None

    def is_available(self):
        return torch.cuda.is_available()

    def range_push(self, msg):
        if hasattr(torch.cuda.nvtx, 'range_push'):
            return torch.cuda.nvtx.range_push(msg)

    def range_pop(self):
        if hasattr(torch.cuda.nvtx, 'range_pop'):
            return torch.cuda.nvtx.range_pop()

    def lazy_call(self, callback):
        return torch.cuda._lazy_call(callback)

    def communication_backend_name(self):
        return self._communication_backend_name

    # Tensor operations

    @property
    def BFloat16Tensor(self):
        return torch.cuda.BFloat16Tensor

    @property
    def ByteTensor(self):
        return torch.cuda.ByteTensor

    @property
    def DoubleTensor(self):
        return torch.cuda.DoubleTensor

    @property
    def FloatTensor(self):
        return torch.cuda.FloatTensor

    @property
    def HalfTensor(self):
        return torch.cuda.HalfTensor

    @property
    def IntTensor(self):
        return torch.cuda.IntTensor

    @property
    def LongTensor(self):
        return torch.cuda.LongTensor

    def pin_memory(self, tensor):
        return tensor.pin_memory()

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith('cuda:'):
            return True
        else:
            return False

    def op_builder_dir(self):
        try:
            # during installation time op_builder is visible, otherwise return deepspeed.ops.op_builder
            import op_builder  # noqa: F401
            return "op_builder"
        except ImportError:
            return "deepspeed.ops.op_builder"

    def create_op_builder(self, class_name):
        try:
            from op_builder import AsyncIOBuilder, CPUAdagradBuilder, CPUAdamBuilder, FusedAdamBuilder, FusedLambBuilder, QuantizerBuilder, SparseAttnBuilder, StochasticTransformerBuilder, TransformerBuilder, InferenceBuilder, UtilsBuilder, SpatialInferenceBuilder
        except ImportError:
            from deepspeed.ops.op_builder import AsyncIOBuilder, CPUAdagradBuilder, CPUAdamBuilder, FusedAdamBuilder, FusedLambBuilder, QuantizerBuilder, SparseAttnBuilder, StochasticTransformerBuilder, TransformerBuilder, InferenceBuilder, UtilsBuilder, SpatialInferenceBuilder

        if class_name == "AsyncIOBuilder":
            return AsyncIOBuilder()
        elif class_name == "CPUAdagradBuilder":
            return CPUAdagradBuilder()
        elif class_name == "CPUAdamBuilder":
            return CPUAdamBuilder()
        elif class_name == "FusedAdamBuilder":
            return FusedAdamBuilder()
        elif class_name == "FusedLambBuilder":
            return FusedLambBuilder()
        elif class_name == "QuantizerBuilder":
            return QuantizerBuilder()
        elif class_name == "SparseAttnBuilder":
            return SparseAttnBuilder()
        elif class_name == "StochasticTransformerBuilder":
            return StochasticTransformerBuilder()
        elif class_name == "TransformerBuilder":
            return TransformerBuilder()
        elif class_name == "InferenceBuilder":
            return InferenceBuilder()
        elif class_name == "UtilsBuilder":
            return UtilsBuilder()
        elif class_name == "SpatialInferenceBuilder":
            return SpatialInferenceBuilder()
        else:
            return None

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension
