import torch
from torch.profiler import profile, record_function, ProfilerActivity
class GPUProfiler(object):
    def __init__(self, model):
        self.model = Model()

    def start_gpu_profiler(inputs):
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True,profile_memory=True) as prof:
                with record_function("model_inference"): //need to implement
                        self.model(inputs)
        return prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)                    

