# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math
import torch
import torch.nn as nn
from deepspeed import comm as dist
#from deepspeed.utils.types import GATED_ACTIVATION_TYPES
from deepspeed.accelerator import get_accelerator
from .op_binding import MLPGemmOp, VectorMatMulOp, GELUGemmOp, ResidualAddOp


class DeepSpeedMLP(nn.Module):
    #_inter_w_buffers = []

    def __init__(self, config, mp_group=None, q_scales=None, q_groups=1, merge_count=1, mlp_extra_grouping=False):
        super(DeepSpeedMLP, self).__init__()

        self.config = config

        data_type = torch.int8 if config.q_int8 else torch.half if config.fp16 else torch.float
        data_type_fp = torch.half if config.fp16 else torch.float

        device = get_accelerator().current_device_name()

        #proj_factor = 2 if self.config.mlp_act_func_type in GATED_ACTIVATION_TYPES else 1
        #self.config.intermediate_size = self.config.intermediate_size if self.config.intermediate_size > 0 else 4 * self.config.hidden_size
        #self.intm_w_sz_per_partition = self.config.intermediate_size * proj_factor // self.config.mp_size
        #self.intm_o_sz_per_partition = self.config.intermediate_size // self.config.mp_size

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
            intm_size_per_partition = self.config.intermediate_size // self.config.mp_size

            self.inter_w = nn.Parameter(torch.empty(self.config.hidden_size,
                                                    intm_size_per_partition,
                                                    dtype=data_type,
                                                    device=device),
                                        requires_grad=False)
            self.inter_b = nn.Parameter(torch.empty(intm_size_per_partition, dtype=data_type_fp, device=device),
                                        requires_grad=False)
            self.output_w = nn.Parameter(torch.empty(intm_size_per_partition,
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

        #if len(DeepSpeedMLP._inter_w_buffers) == 0:
        #    DeepSpeedMLP._inter_w_buffers = [
        #        torch.empty(self.config.hidden_size, self.intm_w_sz_per_partition, dtype=data_type, device=device),
        #        torch.empty(self.intm_w_sz_per_partition, dtype=data_type_fp, device=device)
        #    ]

    #def _merge_inter_w(self):
    #    inter_w = DeepSpeedMLP._inter_w_buffers[0]
    #    inter_w[:self.intm_w_sz_per_partition, :] = self.inter_up_w  # type: ignore
    #    inter_w[self.intm_w_sz_per_partition:, :] = self.inter_gate_w  # type: ignore
    #    if self.inter_up_b is not None:
    #        inter_b = DeepSpeedMLP._inter_w_buffers[1]
    #        inter_b[:self.intm_w_sz_per_partition] = self.inter_up_b  # type: ignore
    #        inter_b[self.intm_w_sz_per_partition:] = self.inter_gate_b  # type: ignore
    #    return DeepSpeedMLP._inter_w_buffers


    def mlp_baseline(self, input, residual, bias):

        # pytorch baseline to add bias. Note: We should not add bias here when attn_base=True.
        #input = input + bias

        # pytorch baseline to do add residual (residual=input)
        input = input + residual

        # copy the weight and bias to fc1
        if self.config.transposed_mode:
            self.fc1.weight.data.copy_(self.inter_w)
        else:
            self.fc1.weight.data.copy_(self.inter_w.transpose(0, 1))
        self.fc1.bias.data.copy_(self.inter_b)

        # copy the weight and bias to fc2
        if self.config.transposed_mode:
            self.fc2.weight.data.copy_(self.output_w)
        else:
            self.fc2.weight.data.copy_(self.output_w.transpose(0, 1))
        self.fc2.bias.data.copy_(self.output_b)

        self.final_layer_norm.bias.data.copy_(self.attn_nb)
        self.final_layer_norm.weight.data.copy_(self.attn_nw)

        residual = input
        input = self.final_layer_norm(input)
        input = self.fc1(input)
        output = self.activation_fn(input)
        output = self.fc2(output)

        # pytorch baseline residual add
        residual = output + residual

        return residual

    # Copy of mlp_baseline for debugging
    def debug_mlp_baseline(self, input, residual, bias):
        debug = False
        print_tensors = True

        # pytorch baseline to add bias. Note: We should not add bias here when attn_base=True.
        #input = input + bias
        if debug: print(f'ds a4 attn + ln + bias-add: norm = {torch.norm(input)}')
        if print_tensors: print(f'tensor = {input}')

        # pytorch baseline to do add residual (residual=input)
        input = input + residual
        if debug: print(f'ds a4 attn + ln + bias-add + residual-add: norm = {torch.norm(input)}')
        if print_tensors: print(f'tensor = {input}')

        # copy the weight and bias to fc1
        if self.config.transposed_mode:
            self.fc1.weight.data.copy_(self.inter_w)
        else:
            self.fc1.weight.data.copy_(self.inter_w.transpose(0, 1))
        self.fc1.bias.data.copy_(self.inter_b)

        # copy the weight and bias to fc2
        if self.config.transposed_mode:
            self.fc2.weight.data.copy_(self.output_w)
        else:
            self.fc2.weight.data.copy_(self.output_w.transpose(0, 1))
        self.fc2.bias.data.copy_(self.output_b)

        if debug: print(f"inside ds mlp: b4 ln weight (shape, norm) = {self.fc1.weight.shape}, {self.fc1.weight.norm()}")
        if debug: print(f"inside ds mlp: b4 ln bias  (shape, norm)  = {self.fc1.bias.shape}, {self.fc1.bias.norm()}")
        if debug: print(f"inside ds mlp: b4 ln input (shape, norm)  = {input.shape}, {input.norm()}")
        if print_tensors: print(f"inside ds mlp: b4 ln input tensor = {input}")

        # do the layernorm
        #print(f"self.final_layer_norm.weight = {self.final_layer_norm.weight}")
        #print(f"self.final_layer_norm.weight norm = {self.final_layer_norm.weight.norm()}")

        self.final_layer_norm.bias.data.copy_(self.attn_nb)
        self.final_layer_norm.weight.data.copy_(self.attn_nw)

        #print(f"self.final_layer_norm.weight = {self.final_layer_norm.weight}")
        #print(f"self.final_layer_norm.weight norm = {self.final_layer_norm.weight.norm()}")

        # probably need a cuda sync - because it was giving wrong output without the next prints
        if debug: print(f"self.final_layer_norm w norm = {self.final_layer_norm.weight.norm()}")
        if debug: print(f"self.final_layer_norm b norm = {self.final_layer_norm.bias.norm()}")

        #print(f"self.final_layer_norm b norm = {self.output_b.norm()}")
        #print(f"self.final_layer_norm b norm = {self.attn_nb.norm()}")
        # bias here is 0 but HF has a really bias.
        residual = input

        input = self.final_layer_norm(input)

        if debug:
            print(
                f"inside ds mlp: a4 ln weight (shape, norm) = {self.fc1.weight.shape}, {self.fc1.weight.norm()}")
        if print_tensors: print(f"inside ds mlp: a4 ln weight = {self.fc1.weight}")
        if debug:
            print(f"inside ds mlp: a4 ln bias (shape, norm)  = {self.fc1.bias.shape}, {self.fc1.bias.norm()}")
        if print_tensors: print(f"inside ds mlp: a4 ln bias = {self.fc1.bias}")
        if debug: print(f"inside ds mlp: a4 ln input  = {input.shape}, {input.norm()}")
        if print_tensors: print(f"inside ds mlp: a4 ln input tensor = {input}")

        input = self.fc1(input)

        if debug: print(f"inside ds mlp: a4 fc1 norm: {input.norm()}")
        if print_tensors: print(f"inside ds mlp: a4 fc1: {input}")

        output = self.activation_fn(input)

        if debug: print(f"inside ds mlp: a4 relu norm: {output.norm()}")
        if print_tensors: print(f"inside ds mlp: a4 relu: {output}")

        if debug: print(f"inside ds mlp: fc2 weight (shape, norm) = {self.fc2.weight.shape}, {self.fc2.weight.norm()}")
        if debug: print(f"inside ds mlp: fc2 bias  (shape, norm)  = {self.fc2.bias.shape}, {self.fc2.bias.norm()}")

        output = self.fc2(output)

        if debug: print(f"inside ds mlp: a4 fc2 norm: {output.norm()}")
        if print_tensors: print(f"inside ds mlp: a4 fc2: {output}")

        # pytorch baseline residual add
        residual = output + residual
        if debug: print(f"residual norm = {residual.norm()}")
        if print_tensors: print(f"residual = {residual}")

        return residual

    def forward(self, input, residual, residual_norm, bias, weight, mlp_base=False, mlp_debug=False):
        #if self.inter_w is None:
        #    self._inter_w, self._inter_b = self._merge_inter_w()
        #else:
        #    self._inter_w = self.inter_w
        #    self._inter_b = self.inter_b

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
