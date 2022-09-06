"""
Copyright 2022 The Microsoft DeepSpeed Team
"""

import pytest
import torch
import deepspeed
from deepspeed.ops.op_builder import InferenceBuilder

if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
    pytest.skip("Inference ops are not available on this system",
                allow_module_level=True)

inference_module = None


def allclose(x, y):
    assert x.dtype == y.dtype
    rtol, atol = {torch.float32: (5e-4, 5e-5), torch.float16: (3e-2, 2e-3)}[x.dtype]
    return torch.allclose(x, y, rtol=rtol, atol=atol)


def run_moe_res_matmult_reference(residual, coef,out):
    return out * coef[..., 0:1] + residual * coef[..., 1:]

#at::Tensor moe_res_matmul(at::Tensor& moe_res, at::Tensor& coef, at::Tensor& output)
def run_moe_res_matmult_ds(residual, coef, output):
    global inference_module
    if inference_module is None:
        inference_module = InferenceBuilder().load()
    return inference_module.moe_res_matmult(residual, coef, output)

@pytest.mark.inference
@pytest.mark.parametrize("sequence", [1, 128, 255])
@pytest.mark.parametrize("hidden_dim", [512, 1232, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_moe_residual_matmult(sequence, channels, dtype):

    residual_ds = torch.randn((sequence, sequence, hidden_dim), dtype=dtype, device='cuda')
    coeff_ds  = torch.randn(hidden_dim, 2, dtype=dtype, device='cuda')

    residual_ref = residual_ds.clone().detach()
    coeff_ref = coeff_ds.clone().detach()

    ds_out = run_moe_res_matmult_ds(residual_ds, coeff_ds, ds_out)
    ref_out = run_moe_res_maltmult_reference(residual_ref, coeff_ref)
    assert (allclose(ds_out, ref_out))
