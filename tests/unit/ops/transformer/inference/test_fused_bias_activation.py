# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import InferenceBuilder
from .inference_test_utils import allclose, get_dtypes

try:
    import triton  # noqa: F401 # type: ignore
    from deepspeed.ops.transformer.inference.triton import (
        bias_act,
    )
except ImportError:
    print("triton import failed")
if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
    pytest.skip("Inference ops are not available on this system", allow_module_level=True)

inference_module = None
torch_minor_version = None


def run_bias_act_reference(activations, bias, act):
    # Expected behavior is that of casting to float32 internally
    if act == 'gelu':
        act_fn = torch.nn.GELU()
    elif act == 'relu':
        act_fn = torch.nn.ReLU()
    elif act == 'silu':
        act_fn = torch.nn.SiLU()
    return act_fn(activations + bias)


def run_bias_act_ds(activations, bias, act):
    return bias_act(activations, bias, act)

@pytest.mark.inference_ops
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("sequence", [128])
@pytest.mark.parametrize("channels", [4096])
@pytest.mark.parametrize("act", ["gelu", "relu", "silu"])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_bias_act(batch, sequence, channels, act, dtype):
    activations_ds = torch.randn((batch, sequence, channels), dtype=dtype, device=get_accelerator().device_name())
    bias_ds = torch.randn((channels), dtype=dtype, device=get_accelerator().device_name())

    activations_ref = activations_ds.clone().detach()
    bias_ref = bias_ds.clone().detach()

    ds_out = run_bias_act_ds(activations_ds, bias_ds, act)
    ref_out = run_bias_act_reference(activations_ref, bias_ref, act)
    assert (allclose(ds_out, ref_out))
