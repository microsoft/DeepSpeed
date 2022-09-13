class DeepSpeedAccelerator(object):
    def __init__(self):
        self.name = None
        self.communication_backend = None
        self.BFloat16Tensor = None
        self.ByteTensor = None
        self.DoubleTensor = None
        self.FloatTensor = None
        self.HalfTensor = None
        self.IntTensor = None
        self.LongTensor = None

    # Device APIs
    def device(self, device_index):
        pass

    def set_device(self):
        pass

    def current_device(self):
        pass

    def device_count(self):
        pass

    def synchronize(self, device_index=None):
        pass

    # RNG APIs
    def set_rng_state(self, new_state, device_index=None):
        pass

    def get_rng_state(self, device_index=None):
        pass

    def manual_seed(self, seed):
        pass

    def manual_seed_all(self, seed):
        pass

    def initial_seed(self):
        pass

    def default_generator(self, device_index):
        pass

    # Streams/Events
    def Stream(self, device_index=None, priority=0, **kwargs):
        pass

    def StreamContext(self, stream):
        pass

    def current_stream(self, device_index=None):
        pass

    def default_stream(self, device_index=None):
        pass

    def Event(self, **kwargs):
        pass

    # Memory management
    def empty_cache(self):
        pass

    def memory_allocated(self, device_index=None):
        pass

    def max_memory_allocated(self, device_index=None):
        pass

    def reset_max_memory_allocated(self, device_index=None):
        pass

    def reset_max_memory_cached(self, device_index=None):
        pass

    def memory_stats(self, device_index=None):
        pass

    def reset_peak_memory_stats(self, device_index=None):
        pass

    def memory_reserved(self, device_index=None):
        pass

    def max_memory_reserved(self, device_index=None):
        pass

    def total_memory(self, device_index=None):
        pass

    # Misc
    def is_available(self):
        pass

    def range_push(self, msg):
        pass

    def range_pop(self, msg):
        pass

    def lazy_call(self, callback):
        pass

    # Data types
    def is_bf16_supported(self):
        pass

    def is_fp_supported(self):
        pass

    # Communication APIs
    def communication_backend(self):
        pass
