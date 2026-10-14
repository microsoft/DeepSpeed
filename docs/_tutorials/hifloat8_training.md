---
# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
title: "TorchAO HiFloat8 training"
---

DeepSpeed selects ordinary Linear and AutoEP GroupedExperts modules, probes the
active accelerator's optional HiFloat8 APIs, and converts both kinds through one
public `torchao.quantization.quantize_` call. DeepSpeed owns selection, activation,
routing/padding, communication, optimizer and checkpoint lifecycle. TorchAO owns
numerical policy and the quantization/GEMM implementations.

```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {"stage": 2},
  "hifloat8": {
    "enabled": true,
    "backend": "torchao",
    "module_name_patterns": ["*.experts", "*.shared_experts.*_proj"],
    "min_numel": 65536
  }
}
```

AutoEP topology and trainable-parameter selection are configured separately.
Every pattern must match an eligible module. Optional `expected_module_count`
checks the complete selection before backend loading. The existing schema is
retained; legacy `backend="torchao_npu"` normalizes to `"torchao"`. The former
`torch_npu` training-helper backend and the temporary `low_precision` section are
rejected with a migration instruction.

## Conversion and device ownership

`DeepSpeedAccelerator.get_hifloat8_provider()` defaults to `None`. The NPU
implementation lazily returns the public `torchao.hifloat8` APIs after checking
optional dependency/version availability. Unsupported accelerators fail before
conversion. No NPU import or device-specific kernel selection occurs in Engine.

`ModuleFqnToConfig` maps selected Linear modules to the shared `HiFloat8Config`
and selected experts to a DeepSpeed `GroupedExpertsConfig` registry adapter.
The Linear handler converts in place; the grouped handler binds a device-provided
GEMM once. Both preserve Parameter identity, names, metadata, `requires_grad`,
mode and state-dict keys. There is no second model-conversion API or registry.
The expert container remains owned by DeepSpeed and retains its activation and
routing/padding behavior.

The separate `get_grouped_mm()` accelerator hook supplies ordinary high-precision
GEMM. Its NPU implementation calls TorchNPU with explicit list-input autograd,
packed `[E,N,K]` weights and empty-group gradients. BF16/FP16 grouped training does
not require TorchAO. CPU/CUDA dispatch retains its existing implementation.

## Optional TorchAO policy

Install `torchao>=0.18,<0.19` and the paired Ascend extension with public HiFloat8
policy/probe/grouped APIs. An upstream Linear-only extension is insufficient.
No backend import or user preloading step is required. Missing APIs, versions,
invalid policy or unavailable requested forward/dX/dW kernels fail before
conversion; policy parsing uses `HiFloat8Config.from_dict`.

Missing, null or empty `config` uses the established input/weight/gradient scale
denominators `15/15/224`. Explicit overrides remain supported. `compute_dtype`
selects high-precision staging and GEMM output precision (`null`, `"bfloat16"`
or `"float16"`); operands remain HiFloat8 payloads and accumulation is
kernel-owned. Converted Linear and expert modules return their caller's dtype,
so FP16 working GEMMs compose with BF16 residuals. Raw tensor helpers retain
their working/output dtype contract. Scaling currently supports per-tensor only.

## Validation boundary

The current NPU HiFloat8 path requires BF16 model parameters and accepts ZeRO
stages 0–2. ZeRO-3, PipelineModule, AutoTP, custom MPU and explicit DeepSpeed MoE
layers are rejected. Quantized checkpoint ingestion into ZeRO-3 is unsupported;
checkpoints retain high-precision weights and their original state names.
Reconstruct the same policy for resume. The opt-in DP2/GAS2 lifecycle entry point
is `tests/unit/runtime/hifloat8_resume_validation.py`, whose experts are replicated.
It does not establish AutoEP checkpoint recovery.

The integration includes Qwen3.5 text-wrapper routing compatibility and NPU
permutation index generation. These reuse AutoEP's existing preset and kernel
extension points without adding HiFloat8 configuration. Full-parameter convergence,
LoRA, offload, compile, EP8, multi-node and CUDA low-precision hardware validation
remain separate gates.
