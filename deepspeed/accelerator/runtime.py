import torch
from deepspeed.accelerator import literal_device


def simple_accel_runtime_api(method):
    def wrapper(*args, **kwargs):
        device = literal_device()
        # check device type against white list
        assert device == 'cuda' or device == 'xpu'
        # check method.__name__ against white list
        assert (method.__name__ == 'set_device' or method.__name__ == 'device'
                or method.__name__ == 'device_count'
                or method.__name__ == 'current_device'
                or method.__name__ == 'synchronize' or method.__name__ == 'get_rng_state'
                or method.__name__ == 'current_stream'
                or method.__name__ == 'current_stream'
                or method.__name__ == 'memory_allocated'
                or method.__name__ == 'max_memory_allocated'
                or method.__name__ == 'max_memory_cached'
                or method.__name__ == 'reset_max_memory_allocated'
                or method.__name__ == 'reset_max_memory_cached'
                or method.__name__ == 'empty_cache' or method.__name__ == 'Event'
                or method.__name__ == 'Stream' or method.__name__ == 'stream'
                or method.__name__ == 'DoubleTensor' or method.__name__ == 'FloatTensor'
                or method.__name__ == 'HalfTensor' or method.__name__ == 'BFloat16Tensor'
                or method.__name__ == 'IntTensor' or method.__name__ == 'ByteTensor'
                or method.__name__ == 'is_available' or method.__name__ == 'manual_seed'
                or method.__name__ == 'manual_seed_all'
                or method.__name__ == '_lazy_call' or method.__name__ == 'initial_seed')
        return eval("torch.{}.{}".format(device, method.__name__))(*args, **kwargs)

    return wrapper


@simple_accel_runtime_api
def set_device(rank):
    pass


@simple_accel_runtime_api
def device(device):
    pass


@simple_accel_runtime_api
def device_count():
    pass


@simple_accel_runtime_api
def current_device():
    pass


@simple_accel_runtime_api
def synchronize():
    pass


@simple_accel_runtime_api
def get_rng_state():
    pass


@simple_accel_runtime_api
def current_stream():
    pass


@simple_accel_runtime_api
def memory_allocated():
    pass


@simple_accel_runtime_api
def max_memory_allocated():
    pass


@simple_accel_runtime_api
def max_memory_cached():
    pass


@simple_accel_runtime_api
def reset_max_memory_allocated():
    pass


@simple_accel_runtime_api
def reset_max_memory_cached():
    pass


@simple_accel_runtime_api
def empty_cache():
    pass


@simple_accel_runtime_api
def Event():
    pass


@simple_accel_runtime_api
def Stream():
    pass


@simple_accel_runtime_api
def stream():
    pass


@simple_accel_runtime_api
def DoubleTensor(data):
    pass


@simple_accel_runtime_api
def FloatTensor(data):
    pass


@simple_accel_runtime_api
def HalfTensor(data):
    pass


@simple_accel_runtime_api
def BFloat16Tensor(data):
    pass


@simple_accel_runtime_api
def IntTensor(data):
    pass


@simple_accel_runtime_api
def ByteTensor(data):
    pass


@simple_accel_runtime_api
def is_available(data):
    pass


@simple_accel_runtime_api
def manual_seed(seed):
    pass


@simple_accel_runtime_api
def manual_seed_all(seed):
    pass


@simple_accel_runtime_api
def initial_seed():
    pass


@simple_accel_runtime_api
def _lazy_call(cb):
    pass


# APIs below does not have simple implementation


def memory_stats():
    device = literal_device()
    if device == 'cuda':
        if hasattr(torch.cuda, "memory_stats"):
            return torch.cuda.memory_stats()
        else:
            return None
    else:
        assert device == 'xpu'
        return torch.xpu.memory_stats()


def reset_peak_memory_stats():
    device = literal_device()
    if device == 'cuda':
        if hasattr(torch.cuda, "reset_peak_memory_stats"):  # pytorch 1.4+
            torch.cuda.reset_peak_memory_stats()
    else:
        assert device == 'xpu'
        torch.xpu.reset_peak_memory_stats()


def default_generator(idx):
    device = literal_device()
    if device == 'cuda':
        return torch.cuda.default_generators[idx]
    else:
        assert device == 'xpu'
        torch.xpu.default_generators[idx]


def is_fp16_supported():
    device = literal_device()
    if device == 'cuda':
        major, _ = torch.cuda.get_device_capability()
        if major >= 7:
            return True
        else:
            return False
    elif device == 'xpu':
        return True
    else:
        return False


def is_bf16_supported():
    device = literal_device()
    if device == 'cuda':
        return torch.cuda.is_bf16_supported()
    elif device == 'xpu':
        return True
    else:
        return False


def memory_reserved():
    device = literal_device()
    if device == 'cuda':
        if hasattr(torch.cuda, "memory_reserved"):
            return torch.cuda.memory_reserved
        else:
            return torch.cuda.memory_allocated
    else:
        assert device == 'xpu'
        return torch.xpu.memory_reserved()


def max_memory_reserved():
    device = literal_device()
    if device == 'cuda':
        if hasattr(torch.cuda, "max_memory_reserved"):
            return torch.cuda.max_memory_reserved
        else:
            return torch.cuda.memory_cached()
    else:
        assert device == 'xpu'
        return torch.xpu.max_memory_reserved()


def memory_cached():
    device == literal_device()
    if device == 'cuda':
        return torch.cuda.memory_cached()
    else:
        assert device == 'xpu'
        return torch.xpu.memory_reserved()


def max_memory_cached():
    device == literal_device()
    if device == 'cuda':
        return torch.cuda.max_memory_cached()
    else:
        assert device == 'xpu'
        return torch.xpu.max_memory_reserved()


def total_memory():
    device = literal_device()
    # Assume all device has same amount of memory
    if device == 'cuda':
        return torch.cuda.get_device_properties(0).total_memory
    else:
        return torch.xpu.get_device_properties(0).total_memory


def default_stream():
    device = literal_device()
    if device == 'cuda':
        return torch.cuda.default_stream()
    else:
        assert device == 'xpu'
        return torch.xpu.current_stream()


def range_push(msg):
    device = literal_device()
    if device == 'cuda':
        torch.cuda.nvtx.range_push(msg)
    else:
        assert device == 'xpu'
        torch.xpu.itt.range_push(msg)


def range_pop():
    device = literal_device()
    if device == 'cuda':
        torch.cuda.nvtx.range_pop()
    else:
        assert device == 'xpu'
        torch.xpu.itt.range_pop()
