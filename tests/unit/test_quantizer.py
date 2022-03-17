import torch
import deepspeed

from deepspeed.ops import op_builder

quantizer_cuda_module = op_builder.QuantizerBuilder().load()


def quantize(inputs, bit, num_groups=1):
    q_range = 2**bit
    input_flat = inputs.float().reshape(num_groups, -1).contiguous()
    input_flat = torch.nan_to_num(input_flat, nan=0.0)
    input_min = input_flat.amin(-1, keepdim=True)
    input_max = input_flat.amax(-1, keepdim=True)

    scale = q_range /  (2 * torch.max(input_min.abs(), input_max.abs()))
    input_flat = (input_flat * scale).round().clamp(-q_range // 2, q_range // 2 - 1)

    return input_flat.reshape(inputs.shape).to(torch.int8).contiguous(), 1/scale.view(-1).contiguous()
        
a = torch.randn(4096,1024).half().cuda()
import time
for _ in range(1000):
    q,s  = quantize(a, 8, num_groups=1)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(1000):
    q,s  = quantize(a, 8, num_groups=1)
torch.cuda.synchronize()
print("time taken by Pytorch is", time.time()-t0)
print(q,s)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(1000):
    a_q, scale = quantizer_cuda_module.ds_quantizer(a,8,1)
torch.cuda.synchronize()
print("time taken with deepspeed is ", time.time()-t0)
print(a_q, scale)
