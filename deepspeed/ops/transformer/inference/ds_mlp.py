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

        self.fc1 = nn.Linear(self.config.hidden_size, self.config.intermediate_size, bias=True, dtype=data_type)
        self.fc2 = nn.Linear(self.config.intermediate_size, self.config.hidden_size, bias=True, dtype=data_type)
        self.activation_fn = nn.ReLU()
        self.final_layer_norm = nn.LayerNorm(self.config.hidden_size, elementwise_affine=True, dtype=data_type, device=device)

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
            debug = True

            if debug: print(f'input norm before mlp: norm = {torch.norm(input)}')

            # pytorch baseline to do add bias.
            # TODO (lekurile): If attn removed, remove this bias addtiiona as well
            input = input + bias
            if debug: print(f'ds a4 attn + ln + bias-add: norm = {torch.norm(input)}')

            # pytorch baseline to do add residual (residual=input)
            input = input + residual
            if debug: print(f'ds a4 attn + ln + bias-add + residual-add: norm = {torch.norm(input)}')

            # copy the weight and bias to fc1
            self.fc1.weight.data.copy_(self.inter_w.transpose(0, 1))
            self.fc1.bias.data.copy_(self.inter_b)

            # copy the weight and bias to fc2
            self.fc2.weight.data.copy_(self.output_w.transpose(0, 1))
            self.fc2.bias.data.copy_(self.output_b)
            torch.cuda.synchronize()

            if debug: print(f"inside ds mlp: b4 ln weight = {self.fc1.weight.shape}, {self.fc1.weight.norm()}")
            if debug: print(f"inside ds mlp: b4 ln bias   = {self.fc1.bias.shape}, {self.fc1.bias.norm()}")
            if debug: print(f"inside ds mlp: b4 ln input  = {input.shape}, {input.norm()}")
            #if debug: print(f"inside ds mlp: b4 ln input tensor = {input}")

            # do the layernorm
            if debug: print(f"self.final_layer_norm w norm = {self.final_layer_norm.weight.norm()}")
            if debug: print(f"self.final_layer_norm b norm = {self.final_layer_norm.bias.norm()}")
            if debug: print(f"self.attn_nb = {self.attn_nb}")

            self.final_layer_norm.bias.data.copy_(self.attn_nb)
            torch.cuda.synchronize()

            # probably need a cuda sync - because it was giving wrong output without the next prints
            if debug: print(f"self.final_layer_norm b norm = {self.final_layer_norm.bias.norm()}")

            #print(f"self.final_layer_norm b norm = {self.output_b.norm()}")
            #print(f"self.final_layer_norm b norm = {self.attn_nb.norm()}")
            # bias here is 0 but HF has a really bias.
            residual = input

            input = self.final_layer_norm(input)

            if debug: print(f"inside ds mlp: a4 ln weight = {self.fc1.weight.shape}, {self.fc1.weight.norm()}")
            if debug: print(f"inside ds mlp: a4 ln bias   = {self.fc1.bias.shape}, {self.fc1.bias.norm()}")
            if debug: print(f"inside ds mlp: a4 ln input (shape, norm) = {input.shape}, {input.norm()}")
            #if debug: print(f"inside ds mlp: a4 ln input tensor = {input}")

            input = self.fc1(input)

            torch.save(input, f'logs/torch_mlp_fc1_tensor_layer_{self.config.layer_id}.pt')

            if debug: print(f"inside ds mlp: a4 fc1: {input.norm()}")

            output = self.activation_fn(input)

            if debug: print(f"inside ds mlp: a4 ac: {output.norm()}")

            output = self.fc2(output)

            torch.save(output, f'logs/torch_mlp_fc2_tensor_layer_{self.config.layer_id}.pt')

            if debug: print(f"inside ds mlp: a4 fc2: {output.norm()}")

            # pytorch baseline residual add
            residual = output + residual
            if debug: print(f"residual = {residual.norm()}")

            torch.save(residual, f'logs/torch_mlp_out_tensor_layer_{self.config.layer_id}.pt')

            return residual

    def forward(self, input, residual, residual_norm, bias, weight):
        if self.inter_w is None:
            self._inter_w, self._inter_b = self._merge_inter_w()
        else:
            self._inter_w = self.inter_w
            self._inter_b = self.inter_b

        residual_add = None

        # mlp_base = True  => calls a pytorch baseline mlp
        # mlp_base = False => calls the DS mlp
        mlp_base = True

        if mlp_base:
            residual = self.mlp_baseline(input, residual, bias)
        else:
            if self.attn_nw is None:
                output = self.fused_gemm_gelu(input=residual_norm,
                                                weight=self.inter_w,
                                                bias=self.inter_b,
                                                weight_out=self.output_w)
            else:
                # mlp_gemm_func ~= gemm(relu(layernorm(input) + bias))
                print(f"input.norm before mlp_gemm_func = {input.norm()}")
                output, residual_add = self.mlp_gemm_func(input=input,
                                                            residual=residual,
                                                            weight_interm=self.inter_w,
                                                            weight_out=self.output_w,
                                                            input_bias=bias,
                                                            bias=self.inter_b,
                                                            gamma=self.attn_nw,
                                                            beta=self.attn_nb)
                print(f"output Norm Python: {output.norm()}")
                print(f"residual_add Norm Python: {residual_add.norm()}")

                output_w_bias = output + self.output_b #TODO: use this for fc2 comparison
                torch.save(output_w_bias, f'logs/ds_mlp_fc2_tensor_layer_{self.config.layer_id}.pt')

                #torch.save(output, f'logs/ds_mlp_fc2_tensor_layer_{self.config.layer_id}.pt')
                #exit(0)


            residual = self.residual_add_func(hidden_state=output,
                                                residual=residual,
                                                add_bias=bias is not None,
                                                attention_output=input,
                                                attention_bias=bias if bias is not None else self.output_b,
                                                final_bias=self.output_b,
                                                residual_add=residual_add)

            torch.save(residual, f'logs/ds_mlp_out_tensor_layer_{self.config.layer_id}.pt')

            if self.mp_group is not None and dist.get_world_size(group=self.mp_group) > 1:
                dist.all_reduce(residual, group=self.mp_group)

        return residual
