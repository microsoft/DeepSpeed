import torch
import pytest
from deepspeed.ops import op_builder

quantizer_cuda_module = op_builder.QuantizerBuilder().load()


def allclose(x, y):
    assert x.dtype == y.dtype
    rtol, atol = {torch.float32: (1e-1, 1e-2), torch.float16: (1e-1, 1e-2)}[x.dtype]
    return torch.allclose(x, y, rtol=rtol, atol=atol)


def run_quantization_dequantization(inputs):
    quant_input, zero_point, qscale = quantizer_cuda_module.ds_quantizer(inputs, 8)
    return quantizer_cuda_module.ds_dequantizer(quant_input,
                                                zero_point,
                                                qscale,
                                                "torch.float16")


@pytest.mark.inference
@pytest.mark.parametrize("input_tensor", torch.rand(8, 8, dtype=torch.float16).cuda())
def test_quant_dequant(input_tensor):
    ref_out = input_tensor.clone().detach()
    ds_out = run_quantization_dequantization(input_tensor)
    assert (allclose(ds_out, ref_out))
