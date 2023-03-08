'''Copyright The Microsoft DeepSpeed Team'''

import torch
import torch.nn.functional as F
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp


class MLPGemmOp(BaseOp):
    def __init__(self, config: DeepSpeedInferenceConfig):
        super(MLPGemmOp, self).__init__(config)
        if not torch.cuda.is_available():
            self.mlp_gemm_func = None
        elif self.config.fp16:
            self.mlp_gemm_func = self.inference_cuda_module.mlp_gemm_fp16
        elif self.config.bf16:
            self.mlp_gemm_func = self.inference_cuda_module.mlp_gemm_bf16
        else:
            self.mlp_gemm_func = self.inference_cuda_module.mlp_gemm_fp32

    def forward(self,
                input: torch.Tensor,
                residual: torch.Tensor,
                input_bias: torch.Tensor,
                weight_interm: torch.Tensor,
                weight_out: torch.Tensor,
                bias: torch.Tensor,
                gamma: torch.Tensor,
                beta: torch.Tensor):
        if self.mlp_gemm_func is None:
            assert self.config.mlp_after_attn
            inp_norm = F.layer_norm(input+residual+input_bias, (input.shape[2],), gamma, beta, self.config.epsilon)
            tmp = torch.matmul(inp_norm, weight_interm)
            tmp = F.gelu(tmp + bias)
            tmp = torch.matmul(tmp, weight_out)
            return tmp, inp_norm
        output, residual_add = self.mlp_gemm_func(
                                    input,
                                    residual,
                                    input_bias,
                                    weight_interm,
                                    weight_out,
                                    bias,
                                    gamma,
                                    beta,
                                    self.config.epsilon,
                                    self.config.pre_layer_norm,
                                    self.config.mlp_after_attn,
                                    weight_interm.scale,
                                    weight_out.scale,
                                    self.config.q_int8,
                                    self.config.mlp_act_func_type)
        return output, residual_add
