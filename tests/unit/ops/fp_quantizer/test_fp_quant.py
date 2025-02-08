# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
from deepspeed.linear import QuantizationConfig

import deepspeed

from deepspeed.ops.fp_quantizer import FP_Quantize
from deepspeed.ops.op_builder import FPQuantizerBuilder
from deepspeed.accelerator import get_accelerator

if not deepspeed.ops.__compatible_ops__[FPQuantizerBuilder.NAME]:
    pytest.skip("FPQuantizer op is not available on this system", allow_module_level=True)

# warning: this import silently JIT builds a set of kernels and may take a minute
from qtorch.quant import float_quantize


def qtorch_quantize(input, exp_bits=4, man_bits=3, rounding="nearest", group_size=1024):
    ori_dt = input.dtype
    ori_shape = input.shape
    last_dim = group_size
    input = input.view(-1, last_dim)

    q_bits = exp_bits + man_bits + 1
    q_range = FPQuantizerBuilder.get_quant_range(q_bits)
    input_to_float = input.float()
    input_max = input_to_float.abs().amax(dim=-1, keepdim=True)

    return ((float_quantize(input_to_float / input_max * q_range, exp_bits, man_bits, rounding=rounding) * \
            input_max / q_range).to(ori_dt)).reshape(ori_shape)


@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
def test_fp_quant_meta(dtype):
    device_name = get_accelerator().device_name()
    group_size = 128
    q_bits = 8
    exp_bits = 4
    man_bits = 3

    quant_config = QuantizationConfig()
    quant_config.q_dtype = FPQuantizerBuilder.get_default_quant_dtype()
    quant_config.group_size = group_size
    fpq = FP_Quantize(quantization_config=quant_config)

    for i in range(10):
        x = torch.rand(4, 1024, dtype=dtype)

        ds_x = x.clone().to(device_name)
        x_quantized, meta_tensor = fpq.quantize(ds_x, q_bits=q_bits, return_meta_tensor=True)
        x_dequantized = fpq.dequantize(x_quantized, q_bits=q_bits, scale=meta_tensor)

        qtorch_out = qtorch_quantize(x, exp_bits=exp_bits, man_bits=man_bits, group_size=group_size)
        qtorch_error = (qtorch_out - x).abs().sum() / x.numel()
        ds_error = (x_dequantized - x).abs().sum() / x.numel()

        assert 0.0004 > abs(qtorch_error.item() - ds_error.item()), f"failed on iteration {i}"


@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
def test_fp_quant_selective(dtype):
    group_size = 128
    q_bits = 8
    exp_bits = 4
    man_bits = 3

    device_name = get_accelerator().device_name()

    quant_config = QuantizationConfig()
    quant_config.q_dtype = FPQuantizerBuilder.get_default_quant_dtype()
    quant_config.group_size = group_size
    fpq = FP_Quantize(quantization_config=quant_config)

    indexes = torch.zeros(2, dtype=torch.int32, device=device_name)
    indexes[0] = 1
    indexes[1] = 3
    for i in range(10):
        x = torch.rand(4, 1024, dtype=dtype, device=device_name)

        x = x.reshape(4, 1, x.shape[-1])
        ds_x = x.clone()
        x_quantized = fpq.quantize(ds_x, q_bits=q_bits)
        x_dequantized = fpq.selective_dequantize(x_quantized, indexes, q_bits=q_bits)

        qtorch_out = qtorch_quantize(x.index_select(0, indexes),
                                     exp_bits=exp_bits,
                                     man_bits=man_bits,
                                     group_size=group_size)
        qtorch_error = (qtorch_out - x.index_select(0, indexes)).abs().sum() / x.numel()
        ds_error = (x_dequantized - x.index_select(0, indexes)).abs().sum() / x.numel()

        assert 0.0004 > abs(qtorch_error.item() - ds_error.item()), f"failed on iteration {i}"


@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("q_bits", [8, 6, 12], ids=["qbits8", "qbits6", "qbits12"])
def test_fp_quant(dtype, q_bits):
    device_name = get_accelerator().device_name()

    quant_config = QuantizationConfig()
    quant_config.q_dtype = FPQuantizerBuilder.get_default_quant_dtype()
    quant_config.group_size = 128
    fpq = FP_Quantize(quantization_config=quant_config)

    for i in range(10):
        x = torch.rand(4, 1024, dtype=dtype)

        ds_x = x.clone().to(device_name)
        x_quantized = fpq.quantize(ds_x, q_bits=q_bits)
        x_dequantized = fpq.dequantize(x_quantized, q_bits=q_bits)

        if q_bits == 8:
            exp_bits = 4
            man_bits = 3
        elif q_bits == 6:
            exp_bits = 3
            man_bits = 2
        elif q_bits == 12:
            exp_bits = 4
            man_bits = 7
        else:
            raise ValueError(f"unknown {q_bits=}")

        qtorch_out = qtorch_quantize(x, exp_bits=exp_bits, man_bits=man_bits, group_size=quant_config.group_size)

        qtorch_error = (qtorch_out - x).abs().sum() / x.numel()
        ds_error = (x_dequantized - x).abs().sum() / x.numel()

        assert 0.0004 > abs(qtorch_error.item() - ds_error.item()), f"failed on iteration {i}"
