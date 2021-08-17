import torch
from torch.profiler import profile, record_function, ProfilerActivity
class GPUProfiler(object):
    def __init__(self, model):
        self.model = model

    def start_gpu_profiler():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function("model_inference"): //need to implement
                        model(inputs)
        return prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)                    
'''
with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)
'''

