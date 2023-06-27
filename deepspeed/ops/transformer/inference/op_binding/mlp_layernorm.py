# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Optional

import os
import torch
import torch.nn.functional as F
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp
from deepspeed.utils.types import NormType


class MLPLayerNormOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig):
        super(MLPLayerNormOp, self).__init__(config)
        try:
            if self.config.norm_type == NormType.LayerNorm:
                if self.config.dtype in [torch.float16, torch.int8]:
                    self.mlp_layer_norm_func = self.inference_module.mlp_layer_norm_fp16  # type: ignore
            elif self.config.norm_type == NormType.RMSNorm:
                # TODO (lekurile): implement the RMS norm as a standalone function
                #if self.config.dtype in [torch.float16, torch.int8]:
                #    self.mlp_gemm_func = self.inference_module.rms_mlp_gemm_fp16  # type: ignore
                #elif self.config.dtype == torch.bfloat16:
                #    self.mlp_gemm_func = self.inference_module.rms_mlp_gemm_bf16
                #else:
                #    self.mlp_gemm_func = self.inference_module.rms_mlp_gemm_fp32  # type: ignore
        except AttributeError:
            # TODO (lekurile): implement a fallback layernorm for both 'LayerNorm' and 'RMSNorm'
            #if self.config.norm_type == NormType.LayerNorm:
            #    self.mlp_gemm_func = self.mlp_gemm_fallback
            #elif self.config.norm_type == NormType.RMSNorm:
            #    self.mlp_gemm_func = self.rms_mlp_gemm_fallback

    def mlp_gemm_fallback(self):
        # TODO (lekurile): implement this

    def rms_mlp_gemm_fallback(self):

    def forward(self,
                input: torch.Tensor,
                residual: torch.Tensor,
                weight_interm: torch.Tensor,
                weight_out: torch.Tensor,
                input_bias: Optional[torch.Tensor] = None,
                bias: Optional[torch.Tensor] = None,
                gamma: Optional[torch.Tensor] = None,
                beta: Optional[torch.Tensor] = None):

        #at::Tensor& input,
        #at::Tensor& residual,
        #at::Tensor& input_bias,
        #at::Tensor& gamma,
        #at::Tensor& beta,
        #const float epsilon,
        #bool mlp_after_attn,
        #int layer_id)
        layer_norm = self.mlp_layer_norm_func(input,
                                              residual,
                                              bias,
                                              self.attn_nw,
                                              self.attn_nb,
                                              self.config.epsilon,
                                              self.config.mlp_after_attn,
                                              self.config.layer_id)
        return layer_norm
