# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Optional training operations supplied by an accelerator backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import torch


class LowPrecisionProvider(ABC):
    """Own policy interpretation and kernels without owning the training loop.

    Probing must finish before conversion. Conversion preserves parameter
    identity and state-dict keys; grouped operations accept packed expert
    weights and return the caller's high-precision output dtype.
    """

    @abstractmethod
    def config_from_dict(self, values: dict | None) -> Any:
        ...

    @abstractmethod
    def assert_training_available(self, *, device: torch.device, config: Any, model_dtype: torch.dtype, linear: bool,
                                  grouped: bool) -> None:
        ...

    @abstractmethod
    def convert_to_training(self, model: torch.nn.Module, config: Any, *,
                            filter_fn: Callable[[torch.nn.Module, str], bool]) -> torch.nn.Module:
        ...

    @abstractmethod
    def grouped_mm(self,
                   x: torch.Tensor,
                   weight: torch.Tensor,
                   offsets: torch.Tensor,
                   *,
                   config: Any,
                   trans_b: bool = False) -> torch.Tensor:
        ...
