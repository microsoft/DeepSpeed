# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math
import torch
import torch.nn as nn
from deepspeed import comm as dist
from deepspeed.utils.types import GATED_ACTIVATION_TYPES
from deepspeed.accelerator import get_accelerator
from .op_binding import MLPGemmOp, VectorMatMulOp, GELUGemmOp, ResidualAddOp

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, dtype, device, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states

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
        self.post_attention_layernorm = LlamaRMSNorm(self.config.hidden_size, dtype=data_type, device=device)

        self.gate_proj = nn.Linear(self.config.hidden_size, self.config.intermediate_size, bias=False, device=device, dtype=data_type)
        self.down_proj = nn.Linear(self.config.intermediate_size, self.config.hidden_size, bias=False, device=device, dtype=data_type)
        self.up_proj = nn.Linear(self.config.hidden_size, self.config.intermediate_size, bias=False, device=device, dtype=data_type)

        self.fc1 = nn.Linear(self.config.hidden_size,
                             self.config.intermediate_size,
                             bias=True,
                             device=device,
                             dtype=data_type)
        self.fc2 = nn.Linear(self.config.intermediate_size,
                             self.config.hidden_size,
                             bias=True,
                             device=device,
                             dtype=data_type)
        self.activation_fn = nn.ReLU()
        self.final_layer_norm = nn.LayerNorm(self.config.hidden_size,
                                             elementwise_affine=True,
                                             dtype=data_type,
                                             device=device)

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


    def mlp_baseline(self, input, residual, bias):
        hidden_states = input + residual

        residual = hidden_states
        self.post_attention_layernorm.weight.data.copy_(self.attn_nw)
        hidden_states = self.post_attention_layernorm(hidden_states)
 
        inter_weights = self.inter_w.split(self.config.intermediate_size, 1)

        if self.config.transposed_mode:
            self.up_proj.weight.data.copy_(inter_weights[0])
            self.gate_proj.weight.data.copy_(inter_weights[1])
            self.down_proj.weight.data.copy_(self.output_w)
        else:
            self.up_proj.weight.data.copy_(inter_weights[0].transpose(0, 1))
            self.gate_proj.weight.data.copy_(inter_weights[1].transpose(0, 1))
            self.down_proj.weight.data.copy_(self.output_w.transpose(0, 1))

        output = self.down_proj(nn.functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        output = output + residual

        return output

    # Copy of mlp_baseline for debugging
    def debug_mlp_baseline(self, input, residual, bias):
        hidden_states = input + residual
        debug = True
        if debug: print(f'ds a4 residual add: hidden_states ={hidden_states}, {torch.norm(hidden_states)}')

        residual = hidden_states
        self.post_attention_layernorm.weight.data.copy_(self.attn_nw)
        hidden_states = self.post_attention_layernorm(hidden_states)
        if debug: print(f'ds post attn ln weights ={self.post_attention_layernorm.weight.data}, {torch.norm(self.post_attention_layernorm.weight.data)}')
        if debug: print(f'ds a4 post attn ln: hidden_states ={hidden_states}, {torch.norm(hidden_states)}')

        inter_weights = self.inter_w.split(self.config.intermediate_size, 1)

        if self.config.transposed_mode:
            self.up_proj.weight.data.copy_(inter_weights[0])
            self.gate_proj.weight.data.copy_(inter_weights[1])
            self.down_proj.weight.data.copy_(self.output_w)
        else:
            self.up_proj.weight.data.copy_(inter_weights[0].transpose(0, 1))
            self.gate_proj.weight.data.copy_(inter_weights[1].transpose(0, 1))
            self.down_proj.weight.data.copy_(self.output_w.transpose(0, 1))

        if debug: print(f'ds gate_proj weights ={self.gate_proj.weight.data}, {torch.norm(self.gate_proj.weight.data)}')
        if debug: print(f'ds up_proj weights ={self.up_proj.weight.data}, {torch.norm(self.up_proj.weight.data)}')
        if debug: print(f'ds down_proj weights ={self.down_proj.weight.data}, {torch.norm(self.down_proj.weight.data)}')

        output = self.down_proj(nn.functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        if debug: print(f'ds a4 mlp: output ={output}, {torch.norm(output)}')

        output = output + residual
        if debug: print(f'ds a4 residual add 2: output ={output}, {torch.norm(output)}')

        return output


    def forward(self, input, residual, residual_norm, bias, weight, mlp_base=False, mlp_debug=False):
        if self.inter_w is None:
            self._inter_w, self._inter_b = self._merge_inter_w()
        else:
            self._inter_w = self.inter_w
            self._inter_b = self.inter_b

        residual_add = None

        # only set mlp_base=True from the caller. This ensures that only the model that is supported (like opt) calls this
        if mlp_base and not mlp_debug:
            residual = self.mlp_baseline(input, residual, bias)
        elif mlp_base and mlp_debug:
            residual = self.debug_mlp_baseline(input, residual, bias)
        else:
            if self.attn_nw is None:
                output = self.fused_gemm_gelu(input=residual_norm,
                                              weight=self.inter_w,
                                              bias=self.inter_b,
                                              weight_out=self.output_w)
            else:
                # mlp_gemm_func ~= gemm(relu(layernorm(input) + bias))
                output, residual_add = self.mlp_gemm_func(input=input,
                                                          residual=residual,
                                                          weight_interm=self.inter_w,
                                                          weight_out=self.output_w,
                                                          input_bias=bias,
                                                          bias=self.inter_b,
                                                          gamma=self.attn_nw,
                                                          beta=self.attn_nb)

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
