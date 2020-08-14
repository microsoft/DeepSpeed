import time
import cupy
import numpy as np
import torch
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
from mpi4py import MPI

tensor_size = 2**(20)
# tensor_size = 5
device = torch.device('cuda:0')
a = torch.randn(tensor_size).to(device)
cupy_a = cupy.fromDlpack(to_dlpack(a))


print(a.dtype)

warmup = 5
rounds = 5

for i in range(warmup):
    np_a = a.cpu().numpy()
    torch.cuda.synchronize()

for i in range(warmup):
    np_a = cupy.asnumpy(cupy_a)
    cupy.cuda.get_current_stream().synchronize()


message_sizes = [0] + [2**i for i in range(28)]
print(message_sizes)
for size in message_sizes:
    a = torch.randn(size).to(device)
    
    start = time.time()
    for i in range(rounds):
        np_a = a.cpu().numpy()
        torch.cuda.current_stream(device).synchronize()
    total = time.time() - start

    #print('Transfering pytorch tensor to Numpy with {}M parameters uses {:.3f}us'.format(size//2**20, total*1e6/rounds))
    print('Transfering pytorch tensor to Numpy with {}M parameters uses {:.3f}ms'.format(size, (total*1000)/rounds))
    
    cupy_a = cupy.fromDlpack(to_dlpack(a))

    start = time.time()
    for i in range(rounds):
        np_a = cupy.asnumpy(cupy_a)
        cupy.cuda.get_current_stream().synchronize()
    total = time.time() - start
    print('Transfering cupy tensor to Numpy with {}M parameters uses {:.3f}ms'.format(size, total*1000/rounds))
