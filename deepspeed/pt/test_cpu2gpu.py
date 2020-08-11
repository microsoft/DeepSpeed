import time
import cupy
import numpy as np
import torch
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

tensor_size = 2**(28)
# tensor_size = 5
device = torch.device('cuda:0')
a = torch.randn(tensor_size).to(device)
cupy_a = cupy.fromDlpack(to_dlpack(a))


warmup = 50
rounds = 50

for i in range(warmup):
    np_a = a.cpu().numpy()
    torch.cuda.synchronize()

start = time.time()
for i in range(rounds):
    np_a = a.cpu().numpy()
    torch.cuda.synchronize()
total = time.time() - start
print('Transfering pytorch tensor to Numpy with {}M parameters uses {:.3f}ms'.format(tensor_size//2**20, total*1000/rounds))


for i in range(warmup):
    np_a = cupy.asnumpy(cupy_a)
    cupy.cuda.get_current_stream().synchronize()

start = time.time()
for i in range(rounds):
    np_a = cupy.asnumpy(cupy_a)
    cupy.cuda.get_current_stream().synchronize()
total = time.time() - start
print('Transfering cupy tensor to Numpy with {}M parameters uses {:.3f}ms'.format(tensor_size//2**20, total*1000/rounds))