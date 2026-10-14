---
# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
title: "Accelerator-owned low-precision training"
---

DeepSpeed selects ordinary Linear and AutoEP GroupedExperts modules, then asks the
active accelerator for an optional low-precision provider. The provider owns policy
parsing, capability probing, selected Linear conversion and grouped GEMM. DeepSpeed
retains model initialization, module selection, expert activation, token routing and
padding, communication, optimizer and checkpoint lifecycle.

```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {"stage": 2},
  "low_precision": {
    "enabled": true,
    "backend": "torchao",
    "format": "hifloat8",
    "module_name_patterns": ["*.experts", "*.shared_experts.*_proj"],
    "min_numel": 65536
  }
}
```

AutoEP topology is configured separately. The patterns select computation, not
trainable parameters. Every pattern must match an eligible module; optional
`expected_module_count` checks the complete selection before backend loading.

The legacy `hifloat8` section remains accepted and selects `format="hifloat8"`.
Its old `backend="torchao_npu"` value is normalized to `"torchao"`. Supply only
one section. The former `backend="torch_npu"` training helper is unsupported.

## Provider contract

`DeepSpeedAccelerator.get_low_precision_provider(backend, format)` returns a
`LowPrecisionProvider` or `None`. The default is `None`, so existing accelerator
implementations remain compatible. Providers implement:

- `config_from_dict(values)`: deserialize an opaque backend policy.
- `assert_training_available(device, config, model_dtype, linear, grouped)`:
  check model precision and the requested native forward/backward kernels.
- `convert_to_training(model, config, filter_fn)`: convert selected Linear modules
  while preserving Parameter identity, names, metadata, `requires_grad`, mode and
  state-dict keys.
- `grouped_mm(x, weight, offsets, config, trans_b)`: differentiable expert GEMM
  with cumulative offsets and the caller's high-precision output dtype.

The current provider is TorchAO HiFloat8 on Ascend NPU. An unsupported
accelerator/backend/format combination fails before conversion. Adding another
format or device requires its own provider and validation; this change does not
claim CUDA low-precision support. Disabled CPU/CUDA dispatch retains its existing
implementation and does not request a low-precision provider.

## TorchAO HiFloat8 policy

The NPU accelerator lazily imports the public `torchao.hifloat8` API. TorchAO is an
optional dependency: install `torchao>=0.18,<0.19` and a compatible Ascend extension
providing the shared policy, conversion/probe and grouped-training APIs. The
paired `torchao_npu:deepspeed_ao` implementation supplies these APIs; the presence
of an upstream Linear-only HiFloat8 package is insufficient. No backend package
import or preloading step is required in the user training entry point.

Missing APIs, unsupported versions, invalid policies and unavailable native kernels
produce an installation/capability error before quantized module conversion.
Policy parsing is delegated to `HiFloat8Config.from_dict`; DeepSpeed does not
reimplement numerical defaults or validation.

Missing, null or empty `config` selects the established input/weight/gradient scale
denominators `15/15/224`. These values influence quantized payloads and are distinct
from the format limit and the native quantizer's default range. Optional overrides
remain supported, for example `"config": {"grad_dst_type_max": 224.0}`.

`compute_dtype` may be null, `"bfloat16"` or `"float16"`; it controls high-precision
staging and GEMM result precision while actual operands remain HiFloat8-tagged
payloads. Native accumulation is kernel-owned. Converted Linear modules and
provider-backed experts return the input/model dtype, allowing FP16 working GEMMs
to compose with BF16 residuals. Raw AO tensor helpers retain their configured
working/output dtype contract. One immutable policy is shared by converted Linear
modules and selected experts. Scaling currently supports per-tensor only.

## Native high-precision grouped GEMM

`DeepSpeedAccelerator.get_grouped_mm()` supplies an optional native differentiable
GEMM override. The NPU adapter uses `torch_npu.npu_grouped_matmul` and explicit
list-input autograd, including empty-rank zero gradients. Its BF16/FP16 path does
not depend on TorchAO. Providers and native overrides both accept `trans_b=True`
for `[E,N,K]` parameter storage and produce dW in that layout.

DeepSpeed supplies activation and routing/padding behavior around these operators.
CUDA/Triton and CPU expert paths retain their existing implementation.

## Training and checkpoint boundary

The current HiFloat8 provider requires BF16 model parameters. The integration
accepts ZeRO stages 0–2 and AutoEP experts; it rejects ZeRO-3, AutoTP, PipelineModule,
custom model-parallel units and explicit DeepSpeed MoE layers. Offload, compilation,
LoRA, multi-node and full-parameter convergence require separate validation.

Checkpoints retain high-precision parameters and unchanged state-dict names.
Reconstruct the engine with the same provider policy to resume. This is not a
quantized-weight checkpoint format and does not implement TorchAO quantized
checkpoint ingestion into ZeRO-3. The DP2/ZeRO-2 lifecycle validation entry point
is `tests/unit/runtime/hifloat8_resume_validation.py`; its experts are replicated,
so it does not establish AutoEP checkpoint recovery.
