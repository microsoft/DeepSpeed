# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# Contributed by The DeepSpeed Team

from enum import IntEnum


class ActivationFuncType(IntEnum):
    UNKNOWN = 0
    GELU = 1
    ReLU = 2
