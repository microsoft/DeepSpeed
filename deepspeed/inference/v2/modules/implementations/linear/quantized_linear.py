# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Any, Dict, Optional

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import InferenceCoreBuilder
from ....allocator import empty_from
from ....inference_utils import is_gated
from ....kernels.core_ops import (
    CUDAWf6Af16Linear,
    CUDABiasActivation,
    CUDAGatedActivation,
)

from ...interfaces import DSLinearBase, DSLinearRegistry
from ...configs import DSLinearConfig
from ....inference_parameter import InferenceParameter


@DSLinearRegistry.register_module
class QuantizedWf6Af16Linear(DSLinearBase):
    """
    Linear DSModule for FP6 weight-only quantization kernel, where weight is FP6 and activation is FP16.
    """

    @staticmethod
    def name():
        return 'quantized_wf6af16_linear'

    @staticmethod
    def supports_config(config: DSLinearConfig) -> bool:
        if config.input_dtype != config.output_dtype:
            return False

        # As for fp6 data items, they are packed and stored in a set of fp16 tensors. E.g., 8 fp6 data items are stored
        # in 3 fp16 tensor.
        if config.input_dtype != torch.float16:
            return False

        if is_gated(config.activation):
            try:
                _ = CUDAGatedActivation(
                    config.out_channels, config.output_dtype, config.activation)
            except ValueError:
                return False
        else:
            try:
                _ = CUDABiasActivation(
                    config.out_channels, config.output_dtype, config.activation)
            except ValueError:
                return False

        return True

    def __init__(self, config: DSLinearConfig, implementation_config: Dict[str, Any]) -> None:
        super().__init__(config, implementation_config)

        self._linear_impl = CUDAWf6Af16Linear()
        self.M = self._config.out_channels
        self.K = self._config.in_channels

        if is_gated(config.activation):
            self._is_gated = True
            self._act_fn = CUDAGatedActivation(
                config.out_channels, config.output_dtype, config.activation)
            self._double_buffer = torch.empty((config.max_tokens, config.out_channels * 2),
                                              dtype=config.output_dtype,
                                              device=get_accelerator().current_device())
        else:
            self._is_gated = False
            self._act_fn = CUDABiasActivation(
                config.out_channels, config.output_dtype, config.activation)

        self._output = torch.empty((config.max_tokens, config.out_channels),
                                   dtype=config.output_dtype,
                                   device=get_accelerator().current_device())

        self.inf_module = InferenceCoreBuilder().load()
        self.inf_module.create_handle()
        self.preprocess_weight = self.inf_module.preprocess_weight
        self.preprocess_scales = self.inf_module.preprocess_scales

    def transform_param(self, param: torch.Tensor) -> InferenceParameter:
        """
        Converts param to same data type as input and output.

        Parameters:
            param (torch.Tensor): Weight or bias tensor.
        """
        # It expects that the quantization scales are store in the attribute `scales`.

        if param.ndim == 1:  # bias, do nothing
            return InferenceParameter.initialize(param)

        device = get_accelerator().current_device()

        if not hasattr(param, "scales"):
            raise ValueError(
                f"param {param.name} does not have attribute `scales`")
        if param.scales is None:
            raise ValueError(f"scales is None")
        scales = param.scales.cpu()
        # if self._is_gated:
        #     # dummy scales for early stage testing
        #     scales = torch.ones(
        #         (self.M * 2, self.K // self.K), dtype=torch.float16)
        # else:
        #     # dummy scales for early stage testing
        #     scales = torch.ones((self.M, self.K // self.K),
        #                         dtype=torch.float16)
        weight = param.cpu()
        # Split the fake quantized fp6 weight into the 4-bit part and 2-bit part.
        weights_2bit, weights_4bit = self.preprocess_weight(weight)

        self.group_size = scales.size(1) // self.K
        if self._is_gated:
            assert weight.shape[0] == self.M * 2
            scales = self.preprocess_scales(scales, self.M * 2, self.K)
        else:
            scales = self.preprocess_scales(scales, self.M, self.K)
        assert self.group_size % 64 == 0, f"group size {self.group_size} is not supported"

        param = weights_4bit
        return InferenceParameter.initialize(param, weights_2bit=weights_2bit, scales=scales)

    def forward(self, hidden_states: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor] = None) -> torch.Tensor:
        weights_4bit = w
        weights_2bit = w.weights_2bit
        scales = w.scales
        output = empty_from(
            self._output, (hidden_states.shape[0], self._config.out_channels))
        N = hidden_states.shape[0]
        if self._is_gated:
            staging_output = empty_from(
                self._double_buffer, (hidden_states.shape[0], self._config.out_channels * 2))
            self._linear_impl(staging_output, hidden_states, weights_4bit,
                              weights_2bit, scales, self.M * 2, hidden_states.shape[0], self.K)
            self._act_fn(output, staging_output, b)
        else:
            self._linear_impl(output, hidden_states, weights_4bit,
                              weights_2bit, scales, self.M, hidden_states.shape[0], self.K)
            self._act_fn(output, b)

        return output

    @property
    def output(self) -> torch.Tensor:
        """
        Return the padded, pre-allocated output Tensor.
        """
        return self._output
