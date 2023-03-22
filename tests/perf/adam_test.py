'''Copyright The Microsoft DeepSpeed Team'''

import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam
import numpy
import torch
from torch.optim import AdamW, Adam, Adagrad
import time
from typing import Callable, Tuple

num_iters = 20
num_params = 1024
values = numpy.linspace(0.01, 1, num_params)
x = torch.from_numpy(values).type(torch.FloatTensor)

def reseed(seed: int) -> None:
    torch.manual_seed(seed)
    numpy.random.seed(seed)

def get_model(seed: int) -> torch.nn.Module:
    reseed(seed)
    model = torch.nn.Linear(num_params, num_params, num_params)
    return model

def build_adam(modelA, modelB):
    optimizerA = Adam(modelA.parameters(), betas=[0.9, 0.999], weight_decay=0.0, lr=1e-3)
    optimizerB = DeepSpeedCPUAdam(modelB.parameters(), betas=[0.9, 0.999], weight_decay=0.0, lr=1e-3, adamw_mode=False)
    return optimizerA, optimizerB

def test_optims(model_factory: Callable, optim_factory, seed:int = 1234, num_steps: int = num_iters):
    modelA = model_factory(seed)
    modelB = model_factory(seed)

    optimizerA, optimizerB = optim_factory(modelA, modelB)
    avgA = 0
    avgB = 0

    for i in range(num_steps):
        xA = x.clone().detach().requires_grad_(True)
        xB = x.clone().detach().requires_grad_(True)

        optimizerA.zero_grad()
        optimizerB.zero_grad()

        y = modelA(xA)
        lossA = y.sum()
        lossA.backward()
        # time the optimization step
        startA = time.time()
        optimizerA.step()
        stopA = time.time()
        avgA += stopA - startA

        y = modelB(xB)
        lossB = y.sum()
        lossB.backward()

        #time the optimization step
        startB = time.time()
        optimizerB.step()
        stopB = time.time()
        avgB += stopB - startB
        
        print(f"step: {i} {type(optimizerA).__name__}: {lossA.item():.4f} {type(optimizerB).__name__}: {lossB.item():.4f}")
    print()
    print(f"Step time: {type(optimizerA).__name__}: {avgA/num_iters:.4f} secs {type(optimizerB).__name__} {avgB/num_iters:.4f} secs")

def main():
    seed = 1234
    #then compare ds optimizer to the baselines 
    test_optims(get_model, build_adam, seed=seed)

if __name__ == "__main__":
    main()
