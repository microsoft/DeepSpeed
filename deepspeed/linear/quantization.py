# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from deepspeed.accelerator import get_accelerator
from deepspeed.ops.fp_quantizer import FP_Quantize
from .config import QuantizationConfig


class QuantizedParameter(nn.Parameter):
    """
    Quantized parameter class that implements weight quantization. Weights
    are stored in quantized form on GPUs, and can be dequantized on-the-fly when
    needed by the model. The weights are actually quantized during any `.to(device)`.
    """

    def __new__(
        cls,
        data: Optional[torch.Tensor] = None,
        requires_grad: bool = False,  # quantized weights should be frozen by default
        quantization: QuantizationConfig = None,
        quantizer=None,
    ):
        if data is None:
            data = torch.empty(0)
        if quantization is None:
            quantization = QuantizationConfig()
        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        if quantizer is not None:
            self.quantizer = quantizer
        else:
            self.quantizer = FP_Quantize(group_size=quantization.group_size)
        self._ensure_quantized(self)
        return self

    def _ensure_quantized(self, tensor: torch.Tensor):
        # If the tensor is on a cuda device and is not quantized, then quantize it in-place.
        if tensor.device.type == "cuda" and tensor.dtype != torch.int8:
            with get_accelerator().stream(get_accelerator().current_stream(tensor.device)):
                tensor.data = self.quantizer.quantize(tensor.data)
            assert tensor.dtype == torch.int8

    def dequantized(self) -> torch.Tensor:
        """
        Return a tensor containing the dequantized weights of this parameter.
        """
        if self.data.device.type == "cuda" and self.data.dtype == torch.int8:
            with get_accelerator().stream(get_accelerator().current_stream(self.data.device)):
                return self.quantizer.dequantize(self.data)
        return self.data

    def __getstate__(self):
        state = self.__dict__
        state["data"] = self.data
        state["requires_grad"] = self.requires_grad
        return state

    def __setstate__(self, state):
        self.quantizer = state["quantizer"]
        self.data = state["data"]
        self.requires_grad = state["requires_grad"]

    def __deepcopy__(self, memo):
        new_instance = type(self).__new__(type(self))
        state = self.__getstate__()
        new_instance.__setstate__(state)
        new_instance.quantizer = copy.deepcopy(state["quantizer"])
        new_instance.data = copy.deepcopy(state["data"])
        return new_instance

    def __copy__(self):
        new_instance = type(self).__new__(type(self))
        state = self.__getstate__()
        new_instance.__setstate__(state)
        return new_instance

    def cuda(self, device=None, non_blocking=False):
        return self.to(device="cuda" if device is None else device, non_blocking=non_blocking)

    def to(self, *args, **kwargs):
        """
        Move the parameter to the given device. Then, if the device is a cuda device,
        quantize it.
        """
        tensor = super().to(*args, **kwargs)
        self._ensure_quantized(tensor)
        return tensor


class QuantizedLinear(nn.Linear):
    """
    Linear layer that implements weight quantization. Parameters
    are stored via `QuantizedParameter` and are dequantized on-the-fly during any
    forward pass.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 bias: bool = False,
                 quantization_config: QuantizationConfig = None,
                 dtype=torch.bfloat16):
        super().__init__(input_dim, output_dim, bias=bias, dtype=dtype)
        self.weight = QuantizedParameter(self.weight.data, quantization=quantization_config)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.dequantized(), self.bias)
