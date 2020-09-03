import torch
from deepspeed import DeepSpeedCPUAdam
import time

device = 'cpu'
model_size = 1 * 1024 ** 3
param = torch.nn.Parameter(torch.ones(model_size, device=device))
optimizer = DeepSpeedCPUAdam([param])
#torch.set_num_threads(128)
param.grad=torch.ones(model_size, device=device)
avg = 0
for i in range(100):
    start = time.time()
    optimizer.step()
    stop = time.time()
    avg += (stop-start)
    param.grad=torch.ones(model_size, device=device)*2
print("Elapsed Time is ", avg / 100)



import torch
from deepspeed import DeepSpeedCPUAdam
import time
import numpy as np


def check_equal(first, second, atol=1e-2, verbose=False):
    if verbose:
        print()
    for i, (x, y) in enumerate(zip(first, second)):
        x = x[0].cpu().detach().numpy()
        y = y[0].cpu().detach().numpy()
        if verbose:
            print("x = {}".format(x.flatten()))
            print("y = {}".format(y.flatten()))
            print('-' * 80)
        np.testing.assert_allclose(x, y, err_msg="Index: {}".format(i), atol=atol)

device = 'cpu'
model_size = 1 * 1024 ** 2
param = torch.nn.Parameter(torch.ones(model_size, device=device))
optimizer1 = torch.optim.Adam([param])
optimizer = DeepSpeedCPUAdam([param])
#torch.set_num_threads(128)
param.grad=torch.ones(model_size, device=device)
avg = 0
for i in range(100):
    start = time.time()
    optimizer.step()
    stop = time.time()
    avg += (stop-start)
    param.grad=torch.ones(model_size, device=device)*2
print("Elapsed Time is ", avg / 100)