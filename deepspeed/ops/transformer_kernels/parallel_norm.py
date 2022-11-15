'''
Copyright 2020 The Microsoft DeepSpeed Team
'''
import torch
from torch import nn

from ..op_builder import TransformerKernelsBuilder

# Cuda modules will be imported if needed
cuda_module = None


class parallel_norm_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, gamma, beta, epsilon, mp_group):
        #import pdb;pdb.set_trace()
        mean, var = cuda_module.partialNorm(inp)
        if mp_group is not None:
            mp_group.all_reduce(mean)
            mp_group.all_reduce(var)
        out, var = cuda_module.partialNorm1(inp, mean, var, gamma, beta, epsilon)

        ctx.save_for_backward(out, var, gamma, beta)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, var, gamma, beta = ctx.saved_tensors
        
        inp_grad, gamma_grad, beta_grad = cuda_module.partialNorm_bwd(out, grad_output, var, gamma, beta)
        return inp_grad, gamma_grad, beta_grad, None, None


class parallel_norm(nn.Module):
    """
        Initialize the DeepSpeed Softmax_Dropout Layer.
    """
    def __init__(self, dim=-1, mp_group=None):
        super(parallel_norm, self).__init__()
        self.dim = dim
        self.mp_group = mp_group
        global cuda_module
        if cuda_module is None:
            builder = TransformerKernelsBuilder()
            cuda_module = builder.load()

    def forward(self, inp, gamma, beta, epsilon):
        return parallel_norm_func.apply(inp, gamma, beta, epsilon, self.mp_group)
