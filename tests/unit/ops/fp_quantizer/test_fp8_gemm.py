# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed

from deepspeed.ops.fp_quantizer import FP_Quantize
from deepspeed.ops.op_builder import FPQuantizerBuilder

if not deepspeed.ops.__compatible_ops__[FPQuantizerBuilder.NAME]:
    pytest.skip("FPQuantizer op is not available on this system", allow_module_level=True)

@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("q_bits", [8], ids=["qbits8",])
def test_fp_quant(dtype, q_bits):
    group_size = 128
    fpq = FP_Quantize(group_size=group_size)

    N = 8192
    H = 4096
    for M in [256, 512, 1024, 1536, 2048, 3072, 4096, 8192]:
        x = torch.randn(M, H, dtype=dtype, device='cuda')
        weight_bf16 = torch.randn(H, N, dtype=dtype, device='cuda')
        weight, _ = fpq.quantize(
            weight_bf16.data,
            q_bits=8,
            return_meta_tensor=True
        )
        scale = fpq.get_scales()
        out = matmul_fp8(
            x, 
            weight,
            scale,
            quantization_group_size,
        )
        out_q = matmul(x, fpq.dequantize(wq, scale=scale))
        error = ((out - out_q).abs() / (out.abs() + 1e-5)).sum() / out.numel()

        assert 0.004 > error, f"failed on iteration {i}"
