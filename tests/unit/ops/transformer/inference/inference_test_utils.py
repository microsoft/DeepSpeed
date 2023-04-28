# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.accelerator import get_accelerator

TOLERANCES = {torch.float32: (5e-4, 5e-5), torch.float16: (3e-2, 2e-3)}
if get_accelerator().is_bf16_supported():
    # Note: BF16 tolerance is higher than FP16 because of the lower precision (7 (+1) bits vs
    # 10 (+1) bits)
    TOLERANCES[torch.bfloat16] = (4.8e-1, 3.2e-2)

DTYPES = [torch.float16, torch.float32]
if get_accelerator().is_bf16_supported():
    DTYPES.append(torch.bfloat16)


def allclose(x, y):
    assert x.dtype == y.dtype
    rtol, atol = TOLERANCES[x.dtype]
    return torch.allclose(x, y, rtol=rtol, atol=atol)
