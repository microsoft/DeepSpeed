import torch
from torch.profiler import profile, record_function, ProfilerActivity


class CPUProfiler(object):
    def __init__(self, model):
        self.model = Model()

    def start_gpu_profiler(inputs):
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True,profile_memory=True) as prof:
                with record_function("model_inference"): 
                        self.model(inputs)
        return prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)  
