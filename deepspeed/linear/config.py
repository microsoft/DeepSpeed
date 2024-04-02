# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from dataclasses import dataclass


@dataclass
class LoRAConfig:
    lora_r: int = 64
    lora_alpha: float = 16
    base_weight_sharding: int = 1


@dataclass
class QuantizationConfig:
    q_bits: int = 8
    rounding: str = "nearest"
    mantissa_bits: int = 3
    group_size: int = 512
