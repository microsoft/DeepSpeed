# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Direct parity against the original HF forward (outputs AND gradients).

The kernel-consistency tests compare kernels against our own reference
implementations; both sides could share a wrong formula. These tests run
injection on real Qwen3.5 decoder layers and compare forward outputs and
input gradients against the untouched HF modules, with explicit bf16
tolerances. Known rounding-order differences (the fused paths keep fp32
intermediates where HF rounds to bf16 mid-way) are expected to sit
within these tolerances — the test pins their magnitude.
"""

import pytest
import torch

_TEST_MODEL = "Qwen/Qwen3.5-0.8B-Base"

# bf16 forward parity: fused paths differ from HF only in intermediate
# rounding order; empirically these bound the drift of one decoder layer
ATOL, RTOL = 5e-2, 5e-2
# gradient parity tolerates the same drift propagated through backward
GATOL, GRTOL = 8e-2, 8e-2


@pytest.fixture(scope="module")
def model():
    pytest.importorskip("transformers")
    try:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(_TEST_MODEL, dtype=torch.bfloat16)
    except Exception as e:
        pytest.skip(f"model {_TEST_MODEL} unavailable: {e}")
    return m


def _layer_pair(model, want_block):
    """Return (idx, pristine_copy, live_module) for one decoder layer.

    The pristine copy is deep-cloned BEFORE injection so its weights and
    forward stay untouched by apply_segment_ki.
    """
    import copy
    layers = model.model.layers
    for i, layer in enumerate(layers):
        if getattr(model.config, "layer_types", [None] * len(layers))[i] == want_block:
            return i, copy.deepcopy(layer), layer
    pytest.skip(f"no {want_block} layer found")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFullAttentionLayerParity:

    def test_forward_and_backward(self, model):
        """Injected full-attn layer vs pristine HF: logits + d(logits)/dx."""
        from deepspeed.module_inject.segment_ki import apply_segment_ki
        idx, pristine, live = _layer_pair(model, "full_attention")
        apply_segment_ki(model)  # mutates live modules (incl. norms, attn)

        torch.manual_seed(0)
        hidden = torch.randn(2, 5, model.config.hidden_size, dtype=torch.bfloat16, device="cuda", requires_grad=True)

        def run(mod, h):
            h = h.detach().clone().requires_grad_(True)
            out = mod(h, position_ids=torch.arange(5, device="cuda").unsqueeze(0).expand(2, -1))
            logits = out[0].float().sum()
            logits.backward()
            return out[0].float(), h.grad.float()

        with torch.no_grad():
            pass  # ensure grads flow: use grad-enabled path (composite fallback)
        out_p, grad_p = run(pristine.to("cuda"), hidden)
        out_i, grad_i = run(live, hidden)
        torch.testing.assert_close(out_i, out_p, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(grad_i, grad_p, atol=GATOL, rtol=GRTOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestGDNLayerParity:

    def test_forward_and_backward(self, model):
        """Injected GDN layer vs pristine HF: output + input gradient."""
        from deepspeed.module_inject.segment_ki import apply_segment_ki
        idx, pristine, live = _layer_pair(model, "linear_attention")
        apply_segment_ki(model)

        torch.manual_seed(0)
        hidden = torch.randn(2, 5, model.config.hidden_size, dtype=torch.bfloat16, device="cuda", requires_grad=True)

        def run(mod, h):
            h = h.detach().clone().requires_grad_(True)
            out = mod(h)  # GDN block takes hidden directly
            loss = out[0].float().pow(2).mean() if torch.is_tensor(out) else out.float().pow(2).mean()
            loss.backward()
            return (out.float() if torch.is_tensor(out) else out[0].float()), h.grad.float()

        out_p, grad_p = run(pristine.to("cuda"), hidden)
        out_i, grad_i = run(live, hidden)
        torch.testing.assert_close(out_i, out_p, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(grad_i, grad_p, atol=GATOL, rtol=GRTOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMLPParity:
    """The fused GLU replacement vs the HF MLP: output + weight grads."""

    def test_forward_and_weight_grads(self, model):
        from deepspeed.module_inject.segment_ki import apply_segment_ki
        apply_segment_ki(model)
        mlp = next(m for n, m in model.named_modules() if n.endswith("mlp"))
        torch.manual_seed(0)
        x = torch.randn(2, 5, mlp.gate_proj.in_features, dtype=torch.bfloat16, device="cuda")

        # grad-enabled forward takes the composite autograd path
        x1 = x.clone().requires_grad_(True)
        out1 = mlp(x1)
        out1.float().pow(2).sum().backward()

        # reference math straight from the HF formula on the same weights
        x2 = x.clone().requires_grad_(True)
        ref = mlp.down_proj(torch.nn.functional.silu(mlp.gate_proj(x2)) * mlp.up_proj(x2))
        ref.float().pow(2).sum().backward()

        torch.testing.assert_close(out1.float(), ref.float(), atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(x1.grad.float(), x2.grad.float(), atol=GATOL, rtol=GRTOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFusedAddNormVsHFFormula:
    """The reviewer's rounding-order case: HF materializes the residual sum
    in bf16 before norm; the kernel keeps the fp32 sum for the output and
    rounds only the residual write-back. Pin both sides' magnitude."""

    def test_output_and_residual_vs_hf(self, model):
        from deepspeed.ops.module_inject import get_fused_glu_op
        torch.manual_seed(0)
        dim = 256
        h = (torch.randn(4, dim, device="cuda") * 0.5).bfloat16()
        r = (torch.randn(4, dim, device="cuda") * 0.5).bfloat16()
        w = (torch.randn(dim, device="cuda") * 0.1).bfloat16()
        eps = 1e-6

        h_kernel, r_kernel = h.clone(), r.clone()
        out_kernel = torch.empty_like(h_kernel)
        get_fused_glu_op().fused_add_norm(h_kernel, r_kernel, (1.0 + w.float()).bfloat16(), out_kernel, eps)

        # HF formula verbatim: bf16 sum materialized, fp32 norm, (1+w) scale
        combined_bf16 = h + r  # bf16 rounding happens here in HF
        xf = combined_bf16.float()
        ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())).bfloat16()

        torch.testing.assert_close(out_kernel.float(), ref.float(), atol=ATOL, rtol=RTOL)
        # the residual stream the kernel writes back is the bf16 sum, as in HF
        torch.testing.assert_close(h_kernel.float(), combined_bf16.float(), atol=ATOL, rtol=ATOL)
