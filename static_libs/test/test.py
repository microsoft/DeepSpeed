import torch
import time
from torch.utils.cpp_extension import load
import os

CUDA_LIB64 = os.path.join(torch.utils.cpp_extension.CUDA_HOME, "lib64")
CUDA_INCLUDE = os.path.join(torch.utils.cpp_extension.CUDA_HOME, "include")
op_module = load(name='bg',
                 sources=['test/pt_binding_test.cpp'],
                 extra_include_paths=['.', CUDA_INCLUDE],
                 extra_cflags=[
                      '-O3',
                      '-std=c++14',
                      f'-L{CUDA_LIB64}',
                      '-lcudart',
                      '-lcublas',
                      '-g',
                      '-Wno-reorder',
                 ],
                 extra_ldflags=[
                     './sample.so'
                 ],
                 verbose=True)
import math
def bias_gelu1(bias, y):
    x = bias + y
    x = x * 0.5 * (1.0 + torch.erf(x / 1.41421))
    return x

a = torch.zeros(8, 1024, device='cuda').half()
bias = torch.ones(1024, device='cuda').half()

b = a.clone()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(1000):
    b= bias_gelu1(bias, b)
torch.cuda.synchronize()
print(f' PT time: {time.time()-t0}')

torch.cuda.synchronize()
t0 = time.time()
for _ in range(1000):
    op_module.bias_gelu(a, bias)
torch.cuda.synchronize()
print(f' DS-Inference time: {time.time()-t0}')
print(f" PT Results: {b}")
print(f" DS Results: {a}")
