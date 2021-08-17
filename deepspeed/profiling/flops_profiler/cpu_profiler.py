import torch
from torch.profiler import profile, record_function, ProfilerActivity
def cpu_profiler(inputs):
  with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)
  return (prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
