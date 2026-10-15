# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Lazy TorchAO registry integration for DeepSpeed expert containers."""

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

from torchao.core.config import AOBaseConfig
from torchao.quantization.transform_module import register_quantize_module_handler

from deepspeed.moe.ep_experts import GroupedExperts


@dataclass
class GroupedExpertsConfig(AOBaseConfig):
    config: Any
    grouped_mm: Callable


@register_quantize_module_handler(GroupedExpertsConfig)
def _configure_grouped_experts(module, config):
    if not isinstance(module, GroupedExperts):
        raise TypeError("GroupedExpertsConfig requires a DeepSpeed GroupedExperts module")
    module.grouped_mm = partial(config.grouped_mm, config=config.config)
    return module
