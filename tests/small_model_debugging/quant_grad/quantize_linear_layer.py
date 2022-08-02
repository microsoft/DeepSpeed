import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import numbers



# do quantization
def quantize(inputs, bit, num_groups=1, sysmetric=False, sparse_outlier=0, stochastic_round=True):
    # print(inputs)
    flatten_input = inputs.reshape(num_groups, -1) # flatten the input as number of groups

    if stochastic_round:
        noise = flatten_input.new(flatten_input.shape).uniform_(-0.5, 0.5)
    else:
        noise = 0. 
    
    if sparse_outlier > 0:
        sparse_part = torch.zeros_like(flatten_input)
    else:
        sparse_part = 0.

    q_range = 2**bit
    
    if sparse_outlier > 0:
        mean = flatten_input.mean(dim=1, keepdim=True) # mean of each group
        std = flatten_input.std(dim=1, keepdim=True) # std of each group
        
        # gaussian ranges, here we use 3
        range_factor = sparse_outlier
        min_values = mean - range_factor * std  
        max_values = mean + range_factor * std 

        # 
        if sysmetric:
            zero_point = 0.
            max_range = torch.max(max_values, min_values)
            scale = q_range / 2 / max_range
            _index = flatten_input.abs() > max_range 
            sparse_part[_index] += flatten_input[_index]  
            flatten_input[_index] *= 0  
        else:
            zero_point = min_values 
            max_range = max_values - min_values
            scale = q_range / max_range

            # max outlier
            _index = flatten_input > max_values 
            sparse_part[_index] += flatten_input[_index]  
            flatten_input[_index] *= 0

            # min outlier
            _index = flatten_input < min_values 
            sparse_part[_index] += flatten_input[_index]  
            flatten_input[_index] *= 0 
    else:
        min_values = flatten_input.amin(dim=1, keepdim=True) # mean of each group
        max_values = flatten_input.amax(dim=1, keepdim=True) # std of each group
        if sysmetric:
            zero_point = 0.
            max_range = torch.max(max_values, min_values)
            scale = q_range / 2 / max_range
        else:
            zero_point = min_values 
            max_range = max_values - min_values
            scale = q_range / max_range

    if sysmetric:
        inputs_quantize = (flatten_input * scale + noise).round().clamp(-(q_range >> 1), (q_range >> 1) - 1) / scale + sparse_part # quantize and dequantize step
    else:
        inputs_quantize = ( (flatten_input-zero_point) * scale + noise).round().clamp(0, q_range - 1) / scale + zero_point + sparse_part

    return inputs_quantize.reshape(inputs.shape)

########################################################################################################################################################


class qlinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, weight, bias=None):
        #print(input.size(), weight.size(), '*'*100)
        with torch.no_grad():

            output = torch.matmul(inputs, weight.t())
            
            if bias is not None:
                if len(output.size()) == 2:
                    output += bias.unsqueeze(0).expand_as(output)
                elif len(output.size()) == 3:
                    output += bias.unsqueeze(0).unsqueeze(0).expand_as(output)
                else:
                    raise Exception("Error happens in linear bias term")         

            ctx.save_for_backward(inputs, weight, bias)
        #print(f"qlinear forward output size is {output.size()}", flush=True)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():

            inputs, weight, bias = ctx.saved_tensors

            grad_input = grad_weight = grad_bias = None
            
            if bias is not None:
                grad_bias = grad_output.sum(0)

            grad_input = torch.matmul(grad_output, weight)
            last_dim_grad = grad_output.size(-1)
            last_dim_input = inputs.size(-1)
            grad_output = grad_output.reshape(-1, last_dim_grad).contiguous()
            inputs = inputs.reshape(-1, last_dim_input).contiguous()
            grad_weight = grad_output.t().mm(inputs)

            # here you can quantize the grad_weight and grad_bias
            grad_weight = quantize(grad_weight, 4, num_groups=(grad_weight.numel() // 32), sysmetric=False, sparse_outlier=6, stochastic_round=False)
            print(f"===Qlinear weight size is {grad_weight.size()}====")
            #grad_bias = quantize(grad_bias, 4, num_groups=(grad_bias.numel() // 512), sysmetric=False, sparse_outlier=6, stochastic_round=False)
        return grad_input, grad_weight, grad_bias, None, None, None

class QuantizeLinear(nn.Linear):
    def __init__(self, *kargs, bias=True, config=None):
        super(QuantizeLinear,self).__init__(*kargs, bias=True)
    
    def forward(self, input):
        output = qlinear.apply(input, self.weight, self.bias)
        return output

