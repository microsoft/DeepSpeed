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
                _ = CUDAGatedActivation(config.out_channels, config.output_dtype, config.activation)
            except ValueError:
                return False
        else:
            try:
                _ = CUDABiasActivation(config.out_channels, config.output_dtype, config.activation)
            except ValueError:
                return False

        return True


    def __init__(self, config: DSLinearConfig, implementation_config: Dict[str, Any]) -> None:
        super().__init__(config, implementation_config)

        self._linear_impl = CUDAWf6Af16Linear()
        self.M = self._config.out_channels
        self.K= self._config.max_tokens
<<<<<<< HEAD
        self.group_size = 128
=======
        self.group_size = 128 
        self.scale = torch.ones((self.M, self.K//self.group_size), dtype=torch.float16, device=get_accelerator().current_device())
        self.weights_2bit = torch.empty((self.M * self.K * 2//8), dtype=torch.uint8, device=get_accelerator().current_device())
        self.weights_4bit = torch.empty((self.M * self.K * 4//8), dtype=torch.uint8, device=get_accelerator().current_device())
>>>>>>> kernel debug

        if is_gated(config.activation):
            self._is_gated = True
            self._act_fn = CUDAGatedActivation(config.out_channels, config.output_dtype, config.activation)
            self._double_buffer = torch.empty((config.max_tokens, config.out_channels * 2),
                                              dtype=config.output_dtype,
                                              device=get_accelerator().current_device())
        else:
            self._is_gated = False
            self._act_fn = CUDABiasActivation(config.out_channels, config.output_dtype, config.activation)

        self._output = torch.empty((config.max_tokens, config.out_channels),
                                   dtype=config.output_dtype,
                                   device=get_accelerator().current_device())

        self.inf_module = InferenceCoreBuilder().load()
        self.inf_module.create_handle()
        self.get_4and2bit_weights = self.inf_module.get_4and2bit_weights


    def transform_param(self, param: torch.Tensor) -> InferenceParameter:
        """
        Converts param to same data type as input and output.

        Parameters:
            param (torch.Tensor): Weight or bias tensor.
        """
        # It expects that the quantization scales are store in the attribute `fp6_quant_scales`.
        # TODO: use builtin attribute/function `q_per_channel_scales` instead in the future, which cannot be
        # used directly as it is not supported by the "CPU" or "GPU" backend currently.
        
        # assert(param.fp6_quant_scales is not None) # Comments out for early stage testing
        
        # Split the fake quantized fp6 weight into the 4-bit part and 2-bit part.
        device = get_accelerator().current_device()
        # TODO: get the correct shape of the weight tensor.
        dummy = 128
        weights_4bit = torch.zeros([dummy, dummy], dtype=torch.uint8, device=device)
        weights_2bit = torch.zeros([dummy, dummy], dtype=torch.uint8, device=device)
        self.get_4and2bit_weights(weights_4bit, weights_2bit, param)
        # The following is the dummy one for early stage testing. It will be replaced by:
        # scales = param.fp6_quant_scales
        scales = torch.ones([dummy], dtype=torch.uint8, device=device) # dummy scales for early stage testing
        del param
        
        return InferenceParameter.initialize(weights_4bit, weights_2bit = weights_2bit, scales = scales)


    def forward(self, hidden_states: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor] = None) -> torch.Tensor:

        output = empty_from(self._output, (hidden_states.shape[0], self._config.out_channels))
        N = hidden_states.shape[0]
        assert (N in [2**i for i in range(3,7)]) or N % 128 == 0, f"accumulated seq-len {N} is not supported"
        assert self.group_size % 64 == 0, f"group size {self.group_size} is not supported"

        if self._is_gated:
            staging_output = empty_from(self._double_buffer, (hidden_states.shape[0], self._config.out_channels * 2))
            self._linear_impl(staging_output, hidden_states, w, self.M, N, self.K)
            self._act_fn(output, staging_output, b)
        else:
            self._linear_impl(output, hidden_states, w, self.M, N, self.K)
            self._act_fn(output, b)

        return output


    @property
    def output(self) -> torch.Tensor:
        """
        Return the padded, pre-allocated output Tensor.
        """
        return self._output
