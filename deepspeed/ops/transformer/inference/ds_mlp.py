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

    def mlp_baseline(self, input, residual, bias, save_tensors, attn_base):
            debug = False

            if debug: print(f'input norm before mlp: norm = {torch.norm(input)}')

            # pytorch baseline to do add bias.
            # TODO (lekurile): If attn removed, remove this bias addtiiona as well
            if not attn_base: input = input + bias
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

            if save_tensors: torch.save(input, f'logs/torch_mlp_ln_tensor_layer_{self.config.layer_id}.pt')

            if debug: print(f"inside ds mlp: a4 ln weight = {self.fc1.weight.shape}, {self.fc1.weight.norm()}")
            if debug: print(f"inside ds mlp: a4 ln bias   = {self.fc1.bias.shape}, {self.fc1.bias.norm()}")
            if debug: print(f"inside ds mlp: a4 ln input (shape, norm) = {input.shape}, {input.norm()}")
            #if debug: print(f"inside ds mlp: a4 ln input tensor = {input}")

            input = self.fc1(input)

            if save_tensors: torch.save(input, f'logs/torch_mlp_fc1_tensor_layer_{self.config.layer_id}.pt')

            if debug: print(f"inside ds mlp: a4 fc1: {input.norm()}")

            output = self.activation_fn(input)

            if save_tensors: torch.save(output, f'logs/torch_mlp_act_tensor_layer_{self.config.layer_id}.pt')
            #import pdb; pdb.set_trace()

            if debug: print(f"inside ds mlp: a4 ac: {output.norm()}")

            output = self.fc2(output)

            if save_tensors: torch.save(output, f'logs/torch_mlp_fc2_tensor_layer_{self.config.layer_id}.pt')

            if debug: print(f"inside ds mlp: a4 fc2: {output.norm()}")

            # pytorch baseline residual add
            residual = output + residual
            if debug: print(f"residual = {residual}, {residual.norm()}")

            if save_tensors: torch.save(residual, f'logs/torch_mlp_out_tensor_layer_{self.config.layer_id}.pt')

            return residual

    def forward(self, input, residual, residual_norm, bias, weight):
        if self.inter_w is None:
            self._inter_w, self._inter_b = self._merge_inter_w()
        else:
            self._inter_w = self.inter_w
            self._inter_b = self.inter_b

        #print(f"input = {input}")
        residual_add = None

        # mlp_base = True  => calls a pytorch baseline mlp
        # mlp_base = False => calls the DS mlp
        mlp_base = True
        attn_base = True
        mlp_functions = True

        debug = False
        save_tensors = True
        print(f"mlp_base = {mlp_base}, attn_base = {attn_base}")

        if mlp_base:
            residual = self.mlp_baseline(input, residual, bias, save_tensors, attn_base)
        elif self.attn_nw is None:
            output = self.fused_gemm_gelu(input=residual_norm,
                                            weight=self.inter_w,
                                            bias=self.inter_b,
                                            weight_out=self.output_w)
        else:
            if attn_base:
                #import pdb; pdb.set_trace()
                #bias = torch.zeros(bias.size(), dtype=bias.dtype)
                bias = bias.zero_()
                #import pdb; pdb.set_trace()

            if mlp_functions:
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

                #print(f"post_ln input = {input}")
                if save_tensors: torch.save(mlp_post_ln, f'logs/ds_mlp_ln_new_tensor_layer_{self.config.layer_id}.pt')
                #import pdb; pdb.set_trace()
                # TODO (lekurile): call layernorm separately followed by add

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
                #print(f"post_fc1 input = {input}")
                if save_tensors: torch.save(mlp_post_fc1, f'logs/ds_mlp_fc1_new_tensor_layer_{self.config.layer_id}.pt')
                #import pdb; pdb.set_trace()


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
                #print(f"post_act input = {input}")
                if save_tensors: torch.save(mlp_post_act, f'logs/ds_mlp_act_new_tensor_layer_{self.config.layer_id}.pt')
                #import pdb; pdb.set_trace()

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
                #print(f"post_fc2 input = {input}")
                #if save_tensors: torch.save(mlp_post_fc2, f'logs/ds_mlp_fc2_new_tensor_layer_{self.config.layer_id}.pt')

                output = mlp_post_fc2
                residual_add = mlp_post_ln
                #import pdb; pdb.set_trace()

                output_w_bias = output + self.output_b #TODO: use this for fc2 comparison
                if save_tensors: torch.save(output_w_bias, f'logs/ds_mlp_fc2_new_tensor_layer_{self.config.layer_id}.pt')
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

                #output_w_bias = output + self.output_b #TODO: use this for fc2 comparison
                #if save_tensors: torch.save(output_w_bias, f'logs/ds_mlp_fc2_tensor_layer_{self.config.layer_id}.pt')

                if save_tensors: torch.save(output, f'logs/ds_mlp_fc2_tensor_layer_{self.config.layer_id}.pt')

            # RESIDUAL_ADD_FUNC FALLBACK IMPLEMENTATION
            #    tmp = (residual.float() + attention_output.float() + attention_bias.float() +
            #            final_bias.float()) / self.config.mp_size + hidden_state.float()

            #residual_torch = output + self.output_b + residual + bias + input

            #print(f"\n==================== layer_{self.config.layer_id} ====================")
            #print(f"output = {output}")
            #print(f"self.output_b = {self.output_b}")
            #print(f"residual = {residual}")
            #print(f"bias = {bias}")
            #print(f"input = {input}")
            #print(f"==================== layer_{self.config.layer_id} ====================\n")

            residual = self.residual_add_func(hidden_state=output,
                                                residual=residual,
                                                add_bias=bias is not None,
                                                attention_output=input,
                                                attention_bias=bias if bias is not None else self.output_b,
                                                final_bias=self.output_b,
                                                residual_add=residual_add)
            if debug: print(f"residual_torch norm = {residual_torch.norm()}")
            if debug: print(f"residual norm = {residual.norm()}")

            if mlp_functions:
                if save_tensors: torch.save(residual, f'logs/ds_mlp_out_new_tensor_layer_{self.config.layer_id}.pt')
            else:
                if save_tensors: torch.save(residual, f'logs/ds_mlp_out_tensor_layer_{self.config.layer_id}.pt')

            if self.mp_group is not None and dist.get_world_size(group=self.mp_group) > 1:
                dist.all_reduce(residual, group=self.mp_group)

        #return residual_torch
        return residual
