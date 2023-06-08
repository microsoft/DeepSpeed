# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import InferenceBuilder
from .inference_test_utils import allclose, get_dtypes

if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
    pytest.skip("Inference ops are not available on this system", allow_module_level=True)

inference_module = None
torch_minor_version = None


def run_vector_matmul_reference(input_tensor, weight):
    return torch.matmul(input_tensor, weight.transpose(0, 1))


def run_linear_reference(input_tensor, weight, bias, dtype):
    linear_module = torch.nn.Linear(weight.size()[1], weight.size()[0], bias = True, dtype=dtype, device=get_accelerator().device_name())
    linear_module.weight.data.copy_(weight)
    linear_module.bias.data.copy_(bias)
    linear_output = linear_module(input_tensor)
    return linear_output


def run_vector_matmul_ds(input_tensor, weight):
    global inference_module
    if inference_module is None:
        inference_module = InferenceBuilder().load()

    q_scale = weight.scale if hasattr(weight, 'scale') else torch.empty(1)
    q_int8 = False
    async_op = False
    transposed_mode = True

    print(input_tensor.size())
    print(weight.size())

    inference_module.allocate_workspace_fp32(
                                    input_tensor.size()[2], #hidden_dim
                                    1,                      #num_heads
                                    input_tensor.size()[2], #prompt length
                                    input_tensor.size()[1], #batch_size
                                    1,                      #num_layers
                                    1,                      #mp_size
                                    False,                  #external cache
                                    0,                      #rank
                                    50,                     #max out tokens
                                    10)                     #min out tokens

    if input_tensor.dtype == torch.float16:
        return inference_module.vector_matmul_fp16(input_tensor, weight, async_op, q_scale, q_int8, transposed_mode)
    elif input_tensor.dtype == torch.int8:
        return inference_module.vector_matmul_int8(input_tensor, weight, async_op, q_scale, q_int8, transposed_mode)
    else:
        return inference_module.vector_matmul_fp32(input_tensor, weight, async_op, q_scale, q_int8, transposed_mode)


@pytest.mark.inference_ops
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("in_features", [2048])
@pytest.mark.parametrize("out_features", [2048])
@pytest.mark.parametrize("dtype", get_dtypes())
def test_vector_matmul(batch_size, in_features, out_features, dtype):

    input_tensor_ds = torch.randn(1, batch_size, in_features, dtype=dtype, device=get_accelerator().device_name())
    weight_tensor_ds = torch.randn(out_features, in_features, dtype=dtype, device=get_accelerator().device_name())

    input_tensor_ref = input_tensor_ds.clone().detach()
    weight_tensor_ref = weight_tensor_ds.clone().detach()

    ds_out = run_vector_matmul_ds(input_tensor_ds, weight_tensor_ds)
    ref_out = run_vector_matmul_reference(input_tensor_ref, weight_tensor_ref)
    if not allclose(ds_out, ref_out):
        print(ds_out)
        print(ref_out)
        print((ds_out - ref_out).abs().max())
        assert (allclose(ds_out, ref_out))


@pytest.mark.inference_ops
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("in_features", [2048])
@pytest.mark.parametrize("out_features", [2048])
@pytest.mark.parametrize("dtype", get_dtypes())
def test_vector_matmul_linear(batch_size, in_features, out_features, dtype):
    input_tensor_ds = torch.randn(1, batch_size, in_features, dtype=dtype, device=get_accelerator().device_name())
    weight_tensor_ds = torch.randn(out_features, in_features, dtype=dtype, device=get_accelerator().device_name())
    bias_tensor_ds = torch.randn(out_features, dtype=dtype, device=get_accelerator().device_name())

    input_tensor_ref = input_tensor_ds.clone().detach()
    weight_tensor_ref = weight_tensor_ds.clone().detach()
    bias_tensor_ref = bias_tensor_ds.clone().detach()

    ds_out = run_vector_matmul_ds(input_tensor_ds, weight_tensor_ds).add(bias_tensor_ds)
    ref_out = run_linear_reference(input_tensor_ref, weight_tensor_ref, bias_tensor_ref, dtype)
    if not allclose(ds_out, ref_out):
        print(ds_out)
        print(ref_out)
        print((ds_out - ref_out).abs().max())
        assert (allclose(ds_out, ref_out))