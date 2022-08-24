'''
Copyright 2020 The Microsoft DeepSpeed Team
'''
import torch
from torch import nn

from ..op_builder import TransformerKernelsBuilder

# Cuda modules will be imported if needed
cuda_module = None


class softmax_dropout_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attn, rel_pos, mask, heads, ratio, generator):
        out = cuda_module.softd_forward(attn,
                                        torch.empty(1) if mask is None else mask,
                                        ratio,
                                        mask is None,
                                        generator,
                                        rel_pos,
                                        heads)

        ctx.save_for_backward(attn, torch.Tensor([ratio]))
        ctx.rel_pos_shape = rel_pos.shape
        return out

    @staticmethod
    def backward(ctx, grad_output):
        attn, ratio = ctx.saved_tensors
        ratio = ratio.item()
        inp_grad = cuda_module.softd_backward(grad_output, attn, ratio)
        return inp_grad, inp_grad.reshape(ctx.rel_pos_shape), None, None, None, None


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

    def forward(self, attn, rel_pos, mask, heads, ratio, generator):
        return softmax_dropout_func.apply(attn, rel_pos, mask, heads, ratio, generator)
