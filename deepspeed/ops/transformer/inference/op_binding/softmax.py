import torch
from deepspeed.ops.transformer import DeepSpeedInferenceConfig
from .base import BaseOp


class SoftmaxOp(BaseOp):
    def __init__(self, config: DeepSpeedInferenceConfig):
        super(SoftmaxOp, self).__init__()
        self.config = config
        if self.config.fp16:
            self.softmax_func = self.inference_cuda_module.softmax_fp16
        else:
            raise NotImplementedError()

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
                head_offset: int,
                mp_size: int):
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
        return output
