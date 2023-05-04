# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import torch.nn.functional as F
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp
from deepspeed.utils.types import NormType


class QKVGemmOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig):
        super(QKVGemmOp, self).__init__(config)
        try:
            if self.config.norm_type == NormType.LayerNorm:
                if self.config.dtype in [torch.float16, torch.int8]:
                    self.qkv_gemm_func = self.inference_module.qkv_gemm_fp16  # type: ignore
                elif self.config.dtype == torch.bfloat16:
                    self.qkv_gemm_func = self.inference_module.qkv_gemm_bf16
                else:
                    self.qkv_gemm_func = self.inference_module.qkv_gemm_fp32  # type: ignore
            elif self.config.norm_type == NormType.RMSNorm:
                if self.config.dtype in [torch.float16, torch.int8]:
                    self.qkv_gemm_func = self.inference_module.rms_qkv_gemm_fp16  # type: ignore
                elif self.config.dtype == torch.bfloat16:
                    self.qkv_gemm_func = self.inference_module.rms_qkv_gemm_bf16
                else:
                    self.qkv_gemm_func = self.inference_module.rms_qkv_gemm_fp32  # type: ignore
        except AttributeError:
            self.qkv_gemm_func = None

    def forward(self, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, gamma: torch.Tensor,
                beta: torch.Tensor):

        add_bias = bias is not None
        bias = bias if add_bias else torch.empty(1)  # type: ignore
        q_scale = weight.scale if hasattr(weight, 'scale') else torch.empty(1)  # type: ignore
        q_int8 = self.config.dtype == torch.int8

        if self.qkv_gemm_func != None:
            if self.config.norm_type == NormType.LayerNorm:
                output, norm = self.qkv_gemm_func(input, weight, q_scale, bias, gamma, beta, self.config.epsilon,
                                                  add_bias, q_int8, self.config.transposed_mode)
            else:
                output, norm = self.qkv_gemm_func(input, weight, q_scale, gamma, self.config.epsilon, q_int8,
                                                  self.config.transposed_mode)
        elif not self.config.transposed_mode:
            # fallback
            inp_norm = F.layer_norm(input, (input.shape[2], ), gamma, beta, self.config.epsilon)
            tmp = torch.matmul(inp_norm, weight)
            if add_bias:
                tmp += bias
            output = [tmp, inp_norm]
        else:
            # fallback
            raise NotImplementedError

        return output, norm
