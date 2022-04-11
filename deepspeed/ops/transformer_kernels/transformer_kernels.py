'''
Copyright 2020 The Microsoft DeepSpeed Team
'''
import json
import math
import importlib
import torch
from torch import nn
from torch.autograd import Function

from ..op_builder import TransformerKernelsBuilder

# Cuda modules will be imported if needed
cuda_module = None


class softmax_dropout_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attn, ratio, mask):
        out = cuda_module.softd_forward(attn, 
                                        torch.empty(1) if mask is None else mask, 
                                        ratio, 
                                        mask is None)
        
        ctx.save_for_backward(attn, mask, torch.Tensor([ratio]))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        attn, mask, ratio = ctx.saved_tensors
        ratio = ratio.item()
        return grad_output, None, None
        return cuda_module.softd_backward(grad_output, 
                                    attn, 
                                    torch.empty(1) if mask is None else mask, 
                                    ratio, 
                                    mask is None), None, None
class softmax_dropout(nn.Module):
    """
        Initialize the DeepSpeed Softmax_Dropout Layer.
    """
    def __init__(self, dim=-1):
        super(softmax_dropout, self).__init__()
        self.dim = dim
        global cuda_module
        if cuda_module is None:
            builder = TransformerKernelsBuilder()
            cuda_module = builder.load()
    def forward(self, attn, ratio, mask=None):
        return softmax_dropout_func.apply(attn, ratio, mask)