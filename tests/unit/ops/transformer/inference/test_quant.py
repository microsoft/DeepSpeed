import torch
import pytest
from deepspeed.ops import op_builder

quantizer_cuda_module = op_builder.QuantizerBuilder().load()


def allclose(x, y):
    assert x.dtype == y.dtype
    rtol, atol = {torch.float32: (1e-2, 1e-3), torch.float16: (1e-2, 1e-3)}[x.dtype]
    return torch.allclose(x, y, rtol=rtol, atol=atol)


def quantize_ref(inputs, bit, num_groups=1):
    q_range = 2**bit
    input_flat = inputs.float().reshape(num_groups, -1).contiguous()
    input_flat = torch.nan_to_num(input_flat, nan=0.0)
    input_min = input_flat.amin(-1, keepdim=True)
    input_max = input_flat.amax(-1, keepdim=True)

    scale = q_range / (2 * torch.max(input_min.abs(), input_max.abs()))
    input_flat = (input_flat * scale).round().clamp(-q_range // 2, q_range // 2 - 1)

    return input_flat.reshape(inputs.shape).to(torch.int8) / scale.view(-1).to(
        torch.float16)


@pytest.mark.inference
@pytest.mark.parametrize("input_tensor", torch.rand(8, 8, dtype=torch.float16).cuda())
def test_quant_dequant(input_tensor):
    ref_input = input_tensor.clone().detach()
    ref_out = quantize_ref(ref_input, 8)

    ds_out = quantizer_cuda_module.ds_quantize_fp16(input_tensor, 1, 8)

    assert (allclose(ds_out, ref_out))
