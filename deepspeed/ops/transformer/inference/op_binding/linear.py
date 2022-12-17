import torch
from deepspeed.ops.transformer import DeepSpeedInferenceConfig
from .base import BaseOp
from typing import Optional


class LinearOp(BaseOp):
    def __init__(self, config: DeepSpeedInferenceConfig):
        super(LinearOp, self).__init__()
        self.config = config
        if self.config.fp16:
            self.linear_func = self.inference_cuda_module.linear_layer_fp16
        else:
            self.linear_func = self.inference_cuda_module.linear_layer_fp32

    def forward(self,
                input: torch.Tensor,
                weight: torch.Tensor,
                bias: torch.Tensor,
                add_bias: bool,
                do_flash_attn: bool,
                num_heads: int,
                q_scale: Optional[torch.Tensor] = None,
                external_cache: bool = None,
                num_layers: int = None,
                q_int8: bool = None,
                max_out_tokens: int = None):
        qkv_out = self.linear_func(input,
                                   weight,
                                   bias,
                                   add_bias,
                                   do_flash_attn,
                                   num_heads)
        return qkv_out
