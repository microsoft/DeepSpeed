'''Copyright The Microsoft DeepSpeed Team'''

import torch
import torch.nn.functional as F
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp


class SoftmaxOp(BaseOp):
    def __init__(self, config: DeepSpeedInferenceConfig):
        super(SoftmaxOp, self).__init__(config)
        self.num_attention_heads_per_partition = config.heads // config.mp_size
        try:
            if self.config.fp16:
                self.softmax_func = self.inference_cuda_module.softmax_fp16
            elif self.config.bf16:
                self.softmax_func = self.inference_cuda_module.softmax_bf16
            else:
                self.softmax_func = self.inference_cuda_module.softmax_fp32
        except AttributeError:
            self.softmax_func = None

    def forward(self,
                attn_scores: torch.Tensor,
                attn_mask: torch.Tensor,
                alibi: torch.Tensor,
                triangular: bool,
                recompute: bool,
                local_attention: bool,
                window_size: int,
                async_op: bool,
                layer_scale: float,
                head_offset: int):
        if self.softmax_func != None:
            output = self.softmax_func(attn_scores,
                                       attn_mask,
                                       alibi,
                                       triangular,
                                       recompute,
                                       local_attention,
                                       window_size,
                                       async_op,
                                       layer_scale,
                                       head_offset,
                                       self.config.mp_size)
        else:
            # fallback
            alibi = alibi[head_offset:head_offset +
                          self.num_attention_heads_per_partition]
            input_dtype = attn_scores.dtype
            if (triangular):
                tri = ~torch.tril(torch.ones_like(attn_scores)).to(bool)
                attn_scores = torch.masked_fill(attn_scores * layer_scale,
                                                tri,
                                                torch.finfo(input_dtype).min)
            if alibi is not None:
                attn_scores += alibi
            if attn_mask is not None:
                # expand atten_mask from two dim into 4 dim, insert two dims in the middle
                attn_mask = attn_mask[:, None, None, :]
                attn_scores += attn_mask
            output = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(input_dtype)

        return output
