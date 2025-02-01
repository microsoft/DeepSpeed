# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Iterable, Set

import torch


LOWER_PRECISION_SAFE_MODULES = [
    "torch.nn.Linear",
    "torch.nn.Conv1d",
    "torch.nn.Conv2d",
    "torch.nn.Conv3d",
]


SUPPORTED_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32
}


def validate_auto_cast_settings(engine):
    # Verify autocast setting
    if engine.torch_autocast_enabled():
        assert not engine.fp16_enabled(), "Cannot enable both torch autocast and fp16"
        assert not engine.bfloat16_enabled(), "Cannot enable both torch autocast and bfloat16"

    assert all(p.dtype == torch.float32 for p in engine.parameters()), "All parameters must be float32 for torch autocast"


def init_autocast_params(model: torch.nn.Module, dtype: torch.dtype) -> None:
    for module in model.modules():
        if module.__class__.__name__ in LOWER_PRECISION_SAFE_MODULES:
            for p in module.parameters(recurse=False):
                p.autocast_dtype = dtype


def get_autocast_dtypes(params: Iterable) -> Set[torch.dtype]:
    return {p.autocast_dtype if hasattr(p, "autocast_dtype") else p.dtype for p in params}
