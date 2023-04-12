# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from enum import IntEnum


class ActivationFuncType(IntEnum):
    UNKNOWN = 0
    GELU = 1
    ReLU = 2
    GEGLU = 3


class NormType(IntEnum):
    UNKNOWN = 0
    LayerNorm = 1
    GroupNorm = 2
    RMSNorm = 3
