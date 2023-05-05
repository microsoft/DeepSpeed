# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from torch import Tensor
from typing import Tuple
import torch.nn as nn
from typing import Dict

# quantizer_cuda_module = deepspeed.ops.op_builder.QuantizerBuilder().load()


def tensor_clamp(tensor: Tensor, min, max) -> Tensor:
    if tensor.device.type == 'cpu' and tensor.dtype == torch.float16:
        # CPU does not support FP16 clamp
        return tensor.to(dtype=torch.float32).clamp_(min, max).to(dtype=torch.float16)
    else:
        return tensor.clamp_(min, max)


def tensor_round(tensor: Tensor) -> Tensor:
    if tensor.device.type == 'cpu' and tensor.dtype == torch.float16:
        # CPU does not support FP16 round
        return tensor.to(dtype=torch.float32).round_().to(dtype=torch.float16)
    else:
        return tensor.round_()


class Quantizer:

    def __init__(self, config: Dict) -> None:
        self.config = config
        assert self.config['num_bits'] == 4 or self.config[
            'num_bits'] == 8, 'Only INT4 and INT8 quantization is supported.'
        assert self.config['symmetric'] == False, 'Only asymmetric quantization is supported at this moment.'

    def quantize(self, tensor: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        assert tensor.shape[self.config['group_dim']] % self.config['group_size'] == 0 \
            , f'Tensor shape: {tensor.shape} quantization config {self.config}'

        tensor = torch.clone(tensor)
        # Use customized CUDA quantization kernel if possible.
        # TODO: Current ZERO++ INT4 asymmetric kernel has numeric issue.
        # Uncomment this code block when Heyang's fix is merged.
        # if self.config.group_size % 8 == 0 and \
        #         self.config.group_dim == len(tensor.shape) - 1 and \
        #         self.config.num_bits == 4:
        #     shape = list(tensor.shape)
        #     shape[-1] = shape[-1] // 2
        #     quantized_tensor, scale_and_min = quantizer_cuda_module.quantize(
        #         tensor.reshape(-1, self.config.group_size),
        #         tensor.numel() // self.config.group_size,
        #         self.config.num_bits,
        #         quantizer_cuda_module.Asymmetric)
        #     return quantized_tensor.reshape(shape), \
        #         torch.narrow(scale_and_min, -1, 0, 1), \
        #             torch.narrow(scale_and_min, -1, 1, 1)

        shape = tensor.shape
        num_groups = shape[self.config['group_dim']] // self.config['group_size']
        new_shape = (shape[:self.config['group_dim']] + (num_groups, self.config['group_size']) +
                     shape[self.config['group_dim'] + 1:])
        tensor = tensor.view(new_shape)

        quantized_tensor, scale, min_value = self._quantize_int8(tensor)
        quantized_tensor = quantized_tensor.view(shape)

        if self.config['num_bits'] == 4:
            return self._compress_uint8_to_uint4(quantized_tensor), scale, min_value
        if self.config['num_bits'] == 8:
            return quantized_tensor, scale, min_value

        assert False, 'Unsupported quantization bits {}'.format(self.config['num_bits'])

    def _quantize_int8(self, tensor: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        q_range = 2**self.config['num_bits'] - 1
        min_value = tensor.amin(dim=self.config['group_dim'] + 1, keepdim=True)
        max_value = tensor.amax(dim=self.config['group_dim'] + 1, keepdim=True)

        scale = q_range / (max_value - min_value)

        tensor = tensor.sub_(min_value).mul_(scale)
        tensor = tensor_round(tensor_clamp(tensor, 0, q_range)).to(torch.uint8)
        return tensor, scale, min_value

    def _compress_uint8_to_uint4(self, tensor: Tensor) -> Tensor:
        assert tensor.shape[-1] % 2 == 0

        new_data_shape = list(tensor.shape)
        new_data_shape[-1] = new_data_shape[-1] // 2

        data = torch.empty(new_data_shape, dtype=torch.uint8, device=tensor.device)
        data = torch.bitwise_or(tensor[..., 0::2].bitwise_left_shift(4), tensor[..., 1::2])

        return data


class DeQuantizer:

    def __init__(self, config: Dict, dtype: torch.dtype) -> None:
        self.config = config
        self.dtype = dtype
        assert self.config['num_bits'] == 4 or self.config[
            'num_bits'] == 8, 'Only INT4 and INT8 quantization is supported.'
        assert self.config['symmetric'] == False, 'Only asymmetric quantization is supported at this moment.'

    def dequantize(self, tensor: Tensor, quant_scale: Tensor, quant_min: Tensor) -> Tensor:
        # Use customized CUDA quantization kernel if possible.
        # TODO: Current ZERO++ INT4 asymmetric kernel has numeric issue.
        # Uncomment this code block when Heyang's fix is merged.
        # if self.config.group_size % 8 == 0 and \
        #         self.config.group_dim == len(tensor.shape) - 1 and \
        #         self.config.num_bits == 4:
        #     shape = list(tensor.shape)
        #     shape[-1] = shape[-1] * 2

        #     quantized_tensor = quantizer_cuda_module.dequantize(
        #         tensor.reshape(-1, self.config.group_size // 2),
        #         torch.concat([quant_scale, quant_min], dim=-1),
        #         tensor.numel() // (self.config.group_size // 2),
        #         self.config.num_bits,
        #         quantizer_cuda_module.Asymmetric
        #     )
        #     return quantized_tensor.reshape(shape)

        if self.config['num_bits'] == 4:
            tensor = self._decompress_uint4_to_uint8(tensor)
        elif self.config['num_bits'] != 8:
            assert False, 'Unsupported quantization bits {}'.format(self.config['num_bits'])

        shape = tensor.shape
        num_groups = shape[self.config['group_dim']] // self.config['group_size']
        new_shape = (shape[:self.config['group_dim']] + (num_groups, self.config['group_size']) +
                     shape[self.config['group_dim'] + 1:])
        tensor = tensor.view(new_shape)

        dequantized_tensor = self._dequantize_int8(tensor, quant_scale, quant_min).view(shape)
        return dequantized_tensor

    def _dequantize_int8(self, tensor: Tensor, quant_scale: Tensor, quant_min: Tensor) -> Tensor:
        assert tensor.dtype == torch.uint8
        data = torch.zeros_like(tensor, dtype=self.dtype, device=tensor.device)
        data = data.copy_(tensor)
        data = data.div_(quant_scale).add_(quant_min)

        return data

    def _decompress_uint4_to_uint8(self, tensor: Tensor) -> Tensor:
        new_data_shape = list(tensor.shape)
        new_data_shape[-1] = new_data_shape[-1] * 2
        data = torch.empty(new_data_shape, dtype=torch.uint8, device=tensor.device)
        data[..., 0::2] = tensor.bitwise_right_shift(4)
        data[..., 1::2] = tensor.bitwise_and(0xF)

        return data


def get_AsyncPartitionedParameterSwapper(model: nn.Module):
    for param_name, param in model.named_parameters():
        if hasattr(param, 'nvme_swapper') and param.nvme_swapper is not None:
            return param.nvme_swapper
    return None
