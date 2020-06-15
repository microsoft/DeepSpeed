from torch import nn
from torch.autograd import Function
import torch
import json
import math
import deepspeed_lsh


class DeepSpeedLSHFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = deepspeed_lsh.forward(input)
        ctx.save_for_backward(input)

        return output[0]

    @staticmethod
    def backward(ctx, grad_output):
        (input) = ctx.saved_tensors

        (grad_input) = deepspeed_lsh.backward(grad_output, input)

        return (grad_input)


class DeepSpeedLSHLayer(nn.Module):
    """Initialize the DeepSpeed LSH Layer.

        Arguments:
            weight: The weight data processed in LSH.
    """
    def __init__(self, weight):
        super(DeepSpeedLSHLayer, self).__init__()

        # create the layer in cuda kernels.
        deepspeed_lsh.create_lsh_layer(weight)

    def forward(self, input):
        return DeepSpeedLSHFunction.apply(input)
