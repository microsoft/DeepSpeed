# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math
import torch
import torch.nn as nn
from deepspeed import comm as dist
from deepspeed.utils.types import GATED_ACTIVATION_TYPES
from deepspeed.accelerator import get_accelerator
from .op_binding import MLPGemmOp, VectorMatMulOp, GELUGemmOp, ResidualAddOp, MLPFunctions


class DeepSpeedMLP(nn.Module):
    _inter_w_buffers = []

    def __init__(self, config, mp_group=None, q_scales=None, q_groups=1, merge_count=1, mlp_extra_grouping=False):
        super(DeepSpeedMLP, self).__init__()

        self.config = config

        data_type = torch.half if self.config.dtype == torch.int8 else self.config.dtype
        data_type_fp = data_type
        device = get_accelerator().current_device_name()

        proj_factor = 2 if self.config.mlp_act_func_type in GATED_ACTIVATION_TYPES else 1
        self.config.intermediate_size = self.config.intermediate_size if self.config.intermediate_size > 0 else 4 * self.config.hidden_size
        self.intm_w_sz_per_partition = self.config.intermediate_size * proj_factor // self.config.mp_size
        self.intm_o_sz_per_partition = self.config.intermediate_size // self.config.mp_size

        if self.config.set_empty_params:
            self.attn_nw = None
            self.attn_nb = None
            self.inter_w = None
            self.inter_b = None
            self.inter_up_w = None
            self.inter_up_b = None
            self.inter_gate_w = None
            self.inter_gate_b = None
            self.output_w = None
            self.output_b = None
        else:
            self.attn_nw = nn.Parameter(torch.empty(self.config.hidden_size, dtype=data_type_fp, device=device),
                                        requires_grad=False)
            self.attn_nb = nn.Parameter(torch.empty(self.config.hidden_size, dtype=data_type_fp, device=device),
                                        requires_grad=False)

            self.inter_w = nn.Parameter(torch.empty(self.config.hidden_size,
                                                    self.intm_w_sz_per_partition,
                                                    dtype=data_type,
                                                    device=device),
                                        requires_grad=False)
            self.inter_b = nn.Parameter(torch.empty(self.intm_w_sz_per_partition, dtype=data_type_fp, device=device),
                                        requires_grad=False)
            self.output_w = nn.Parameter(torch.empty(self.intm_o_sz_per_partition,
                                                     self.config.hidden_size,
                                                     dtype=data_type,
                                                     device=device),
                                         requires_grad=False)
            self.output_b = nn.Parameter(torch.empty(self.config.hidden_size, dtype=data_type_fp, device=device),
                                         requires_grad=False)

        # used for quantization
        self.q_scales = q_scales
        self.q_groups = q_groups * 2 if mlp_extra_grouping else q_groups
        self.merge_count = int(math.log2(merge_count))
        self.mp_group = mp_group

        self.mlp_gemm_func = MLPGemmOp(config)
        self.vector_matmul_func = VectorMatMulOp(config)
        self.fused_gemm_gelu = GELUGemmOp(config)
        self.residual_add_func = ResidualAddOp(config)
        self.mlp_functions = MLPFunctions(config)

        if len(DeepSpeedMLP._inter_w_buffers) == 0:
            DeepSpeedMLP._inter_w_buffers = [
                torch.empty(self.config.hidden_size, self.intm_w_sz_per_partition, dtype=data_type, device=device),
                torch.empty(self.intm_w_sz_per_partition, dtype=data_type_fp, device=device)
            ]

    def _merge_inter_w(self):
        inter_w = DeepSpeedMLP._inter_w_buffers[0]
        inter_w[:self.intm_w_sz_per_partition, :] = self.inter_up_w  # type: ignore
        inter_w[self.intm_w_sz_per_partition:, :] = self.inter_gate_w  # type: ignore
        if self.inter_up_b is not None:
            inter_b = DeepSpeedMLP._inter_w_buffers[1]
            inter_b[:self.intm_w_sz_per_partition] = self.inter_up_b  # type: ignore
            inter_b[self.intm_w_sz_per_partition:] = self.inter_gate_b  # type: ignore
        return DeepSpeedMLP._inter_w_buffers

    def forward(self, input, residual, residual_norm, bias, weight):
        if self.inter_w is None:
            self._inter_w, self._inter_b = self._merge_inter_w()
        else:
            self._inter_w = self.inter_w
            self._inter_b = self.inter_b

        residual_add = None
        if self.attn_nw is None:
            output = self.fused_gemm_gelu(input=residual_norm,
                                            weight=self.inter_w,
                                            bias=self.inter_b,
                                            weight_out=self.output_w)
        else:
                                                                #at::Tensor& input,
                                                                #at::Tensor& residual,
                                                                #at::Tensor& input_bias,
                                                                #at::Tensor& gamma,
                                                                #at::Tensor& beta,
                                                                #const float epsilon,
                                                                #bool mlp_after_attn,
                                                                #int layer_id)
            mlp_post_ln = self.mlp_functions.inference_module.mlp_layer_norm_fp16(input,
                                                                    residual,
                                                                    bias,
                                                                    self.attn_nw,
                                                                    self.attn_nb,
                                                                    self.config.epsilon,
                                                                    self.config.mlp_after_attn,
                                                                    self.config.layer_id)

                                            #at::Tensor mlp_gemm_fc(at::Tensor& inp_norm,
                                            #                       at::Tensor& input,
                                            #                       at::Tensor& weight,
                                            #                       at::Tensor& q_scale,
                                            #                       bool q_int8,
                                            #                       bool transposed_mode,
                                            #                       bool fc1,
                                            #                       int layer_id)
            mlp_post_fc1 = self.mlp_functions.inference_module.mlp_gemm_fc_fp16(mlp_post_ln,
                                                                                input,
                                                                                self.inter_w,
                                                                                self.inter_w.scale if hasattr(self.inter_w, 'scale') else torch.empty(1),  # type: ignore
                                                                                self.config.dtype == torch.int8,
                                                                                self.config.transposed_mode,
                                                                                True,
                                                                                self.config.layer_id)

                                        #at::Tensor mlp_activation(at::Tensor& input,
                                        #                          at::Tensor& input_mlp,
                                        #                          at::Tensor& weight,
                                        #                          at::Tensor& bias,
                                        #                          bool q_int8,
                                        #                          int activation_type,
                                        #                          bool transposed_mode,
                                        #                          int layer_id)
            #(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: torch.Tensor, arg4: bool, arg5: int, arg6: bool, arg7: int)
            mlp_post_act = self.mlp_functions.inference_module.mlp_activation_fp16(mlp_post_fc1,
                                                                                input,
                                                                                self.inter_w,
                                                                                self.inter_b,
                                                                                self.config.dtype == torch.int8,
                                                                                self.config.mlp_act_func_type,
                                                                                self.config.transposed_mode,
                                                                                self.config.layer_id)

                                            #at::Tensor mlp_gemm_fc(at::Tensor& inp_norm,
                                            #                       at::Tensor& input,
                                            #                       at::Tensor& weight,
                                            #                       at::Tensor& q_scale,
                                            #                       bool q_int8,
                                            #                       bool transposed_mode,
                                            #                       bool fc1,
                                            #                       int layer_id)
            # TODO (lekurile): Check the size difference in fc2 inside mlp_gemm_fc_fp16
            mlp_post_fc2 = self.mlp_functions.inference_module.mlp_gemm_fc_fp16(mlp_post_act,
                                                                                input,
                                                                                self.output_w,
                                                                                self.output_w.scale if hasattr(self.output_w, 'scale') else torch.empty(1),  # type: ignore
                                                                                self.config.dtype == torch.int8,
                                                                                self.config.transposed_mode,
                                                                                False,
                                                                                self.config.layer_id)

            output = mlp_post_fc2
            residual_add = mlp_post_ln

            #else:
            #output, residual_add = self.mlp_gemm_func(input=input,
            #                                            residual=residual,
            #                                            weight_interm=self.inter_w,
            #                                            weight_out=self.output_w,
            #                                            input_bias=bias,
            #                                            bias=self.inter_b,
            #                                            gamma=self.attn_nw,
            #                                            beta=self.attn_nb)

        residual = self.residual_add_func(hidden_state=output,
                                            residual=residual,
                                            add_bias=bias is not None,
                                            attention_output=input,
                                            attention_bias=bias if bias is not None else self.output_b,
                                            final_bias=self.output_b,
                                            residual_add=residual_add)

        if self.mp_group is not None and dist.get_world_size(group=self.mp_group) > 1:
            dist.all_reduce(residual, group=self.mp_group)

        return residual
