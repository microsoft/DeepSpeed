# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Expert precision selection must preserve the base model and gradient contract."""

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.moe.ep_experts import GroupedExperts


@pytest.mark.parametrize("counts", [[2, 1], [0, 3], [3, 0], [0, 0]])
def test_reference_experts_match_independent_linear_modules(counts):
    # Catches wrong gate/up/down orientation and lost empty-expert gradients.
    torch.manual_seed(42)
    experts = GroupedExperts(8, 16, 2, use_grouped_mm=False)
    references = []
    with torch.no_grad():
        for weight in experts.parameters():
            weight.normal_(std=0.1)
        for i in range(2):
            projections = [
                torch.nn.Linear(8, 16, bias=False),
                torch.nn.Linear(8, 16, bias=False),
                torch.nn.Linear(16, 8, bias=False)
            ]
            for linear, weight in zip(projections, (experts.w1, experts.w3, experts.w2)):
                linear.weight.copy_(weight[i])
            references.append(projections)
    x = torch.randn(sum(counts), 8, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    actual = experts(x, torch.tensor(counts))
    ref_outputs = []
    for rows, (gate, up, down) in zip(ref_x.split(counts), references):
        ref_outputs.append(down(torch.nn.functional.silu(gate(rows)) * up(rows)))
    expected = torch.cat(ref_outputs)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad)
    for i, projections in enumerate(references):
        for weight, linear in zip((experts.w1, experts.w3, experts.w2), projections):
            torch.testing.assert_close(weight.grad[i], linear.weight.grad)
            if counts[i] == 0:
                assert torch.count_nonzero(weight.grad[i]) == 0


def test_npu_expert_helper_preserves_upstream_activation(monkeypatch):
    """The NPU GEMM dispatch must use AutoEP's resolved expert activation."""
    # Pin the merge regression without NPU hardware: the stub implements the
    # native grouped GEMM contract while the helper supplies activation policy.
    import deepspeed.moe.ep_experts as ep_experts

    def fake_grouped_mm(lhs, weight, offsets, *, trans_b=False):
        weight = weight.transpose(-1, -2) if trans_b else weight
        starts = [0] + offsets.tolist()[:-1]
        return torch.cat([lhs[start:end] @ weight[i] for i, (start, end) in enumerate(zip(starts, offsets.tolist()))])

    torch.manual_seed(7)
    counts = torch.tensor([2, 1])
    x = torch.randn(4, 8, requires_grad=True)
    weights = [torch.randn(shape, requires_grad=True) for shape in ((2, 16, 8), (2, 8, 16), (2, 16, 8))]
    w1, w2, w3 = weights
    actual = ep_experts._run_experts_with_grouped_mm(w1,
                                                     w2,
                                                     w3,
                                                     x,
                                                     counts,
                                                     "swiglu_clamped",
                                                     limit=0.5,
                                                     grouped_mm=fake_grouped_mm)

    x_ref = x.detach().clone().requires_grad_()
    w1_ref, w2_ref, w3_ref = [w.detach().clone().requires_grad_() for w in weights]
    pieces = []
    start = 0
    for i, count in enumerate(counts.tolist()):
        rows = x_ref[start:start + count]
        gate = rows @ w1_ref[i].T
        up = rows @ w3_ref[i].T
        hidden = torch.nn.functional.silu(gate.clamp(max=0.5)) * up.clamp(min=-0.5, max=0.5)
        pieces.append(hidden @ w2_ref[i].T)
        start += count
    expected = torch.cat((torch.cat(pieces), x_ref.new_zeros((1, 8))))
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    for value, reference in ((x, x_ref), (w1, w1_ref), (w2, w2_ref), (w3, w3_ref)):
        torch.testing.assert_close(value.grad, reference.grad)


@pytest.mark.skipif(get_accelerator().device_name() != "npu", reason="requires an available NPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("trans_b", [False, True])
@pytest.mark.parametrize("input_grad,weight_grad", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("counts", [[0, 8, 0, 4], [0, 0, 0, 0]])
def test_npu_native_grouped_mm_matches_matmul_with_partial_gradients(dtype, trans_b, input_grad, weight_grad, counts):
    # Catches saved-tensor ordering errors, packed dW orientation and empty-group leakage.
    from deepspeed.ops.npu.grouped_mm import grouped_mm

    torch.npu.set_device(0)
    torch.manual_seed(71)
    inner, columns = 64, 32
    shape = (len(counts), columns, inner) if trans_b else (len(counts), inner, columns)
    x = (torch.randn(sum(counts), inner, device="npu", dtype=dtype) * 0.1).requires_grad_(input_grad)
    weight = (torch.randn(shape, device="npu", dtype=dtype) * 0.1).requires_grad_(weight_grad)
    ref_x = x.detach().clone().requires_grad_(input_grad)
    ref_weight = weight.detach().clone().requires_grad_(weight_grad)
    offsets = torch.tensor(counts, device="npu", dtype=torch.int64).cumsum(0)
    actual = grouped_mm(x, weight, offsets, trans_b=trans_b)

    pieces = []
    for index, rows in enumerate(ref_x.split(counts)):
        right = ref_weight[index].T if trans_b else ref_weight[index]
        pieces.append(rows @ right)
    expected = torch.cat(pieces)
    grad = torch.randn_like(expected) * 0.1
    actual.backward(grad)
    expected.backward(grad)
    torch.npu.synchronize()

    pairs = [(actual, expected)]
    for value, reference, needs_grad in ((x, ref_x, input_grad), (weight, ref_weight, weight_grad)):
        if needs_grad:
            assert value.grad is not None and reference.grad is not None
            pairs.append((value.grad, reference.grad))
        else:
            assert value.grad is None and reference.grad is None
    rtol, atol = (0.02, 0.0005) if dtype == torch.bfloat16 else (0.003, 0.00005)
    for value, reference in pairs:
        assert value.dtype == reference.dtype == dtype
        assert bool(torch.isfinite(value).all())
        torch.testing.assert_close(value, reference, rtol=rtol, atol=atol)
    if weight_grad:
        for index, count in enumerate(counts):
            if count == 0:
                assert torch.count_nonzero(weight.grad[index]) == 0


def test_grouped_override_rejects_explicit_fused_gate_up():
    # Upstream can request a fused gate/up path; a three-GEMM override cannot
    # silently ignore that public choice after this integration is merged.
    experts = GroupedExperts(8, 16, 2, use_grouped_mm=False)
    experts.gate_up_impl = "fused"

    def unexpected_kernel(*args, **kwargs):
        raise AssertionError("Unsupported execution must fail before a GEMM")

    experts.grouped_mm = unexpected_kernel
    with pytest.raises(RuntimeError, match="separate gate/up"):
        experts(torch.randn(2, 8), torch.tensor([1, 1]))
