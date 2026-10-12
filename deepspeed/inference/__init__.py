# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

_V2_NAMES = ("RaggedInferenceEngineConfig", "DeepSpeedTPConfig", "InferenceEngineV2", "build_hf_engine",
             "build_engine_from_ds_checkpoint")


def __getattr__(name):
    # These names come from FastGen (inference v2), which loads ~175 modules that import deepspeed does not need, so
    # v2 is imported on first access. from deepspeed.inference import build_hf_engine still works.
    if name in _V2_NAMES:
        from . import v2
        globals()[name] = getattr(v2, name)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
