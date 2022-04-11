import deepspeed
from deepspeed.ops import softmax_dropout
import torch
import time

mse = torch.nn.MSELoss()
d = torch.randn(1, 1, 8, 512, requires_grad=True).bfloat16().cuda()
y = torch.randn(1, 1, 8, 512).bfloat16().cuda()

soft = torch.nn.Softmax(dim=-1)
drop = torch.nn.Dropout(0.1)

out = drop(soft(d))
out.retain_grad()
loss=mse(out, y)
loss.backward()
print(out.grad)

sd = softmax_dropout()
out = sd(d, ratio=0.1)
out.retain_grad()

loss=mse(out, y)
loss.backward()
print(out.grad)
