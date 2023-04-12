# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp


class GELUGemmOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig):
        super(GELUGemmOp, self).__init__(config)
        if self.config.fp16:
            self.fused_gemm_gelu = self.inference_cuda_module.fused_gemm_gelu_fp16  # type: ignore
        else:
            self.fused_gemm_gelu = self.inference_cuda_module.fused_gemm_gelu_fp32  # type: ignore

    def forward(self, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, weight_out: torch.Tensor):

        output = self.fused_gemm_gelu(
            input,
            weight,
            weight.scale if hasattr(weight, 'scale') else torch.empty(1),  # type: ignore
            bias,
            weight_out,
            weight_out.scale if hasattr(weight_out, 'scale') else torch.empty(1),  # type: ignore
            self.config.q_int8,
            self.config.transposed_mode)

        return output
