# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch.nn as nn
from deepspeed.accelerator import get_accelerator

# currently there is no cuda version, so we have to get builder like this
FlashAttentionBuilder = get_accelerator().get_op_builder("FlashAttentionBuilder")

flash_attn_builder = None


class DeepSpeedFlashAttn(nn.Module):
    """Initialize the DeepSpeed Flash Attention.

        Arguments:
            causal: bool. Whether to apply causal attention mask. (default: True)

            softmax_scale: float. The temperature to use for the softmax attention. (default: 1/sqrt(head_size))

            dropout_p: float. The dropout rate to apply to the attention (default: 0.0)

            return_softmax: bool. return softmax result or not, only for testing. (default: False)
    """

    def __init__(self, causal=True, softmax_scale=None, dropout_p=0.0, return_softmax=False):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = dropout_p
        self.return_softmax = return_softmax

        # check if builder available
        assert (FlashAttentionBuilder is not None)

        global flash_attn_builder
        if flash_attn_builder is None:
            flash_attn_builder = FlashAttentionBuilder().load()

    def forward(self, q, k, v):
        return flash_attn_builder.flash_attn_func(q, k, v, self.dropout_p, self.softmax_scale, self.causal,
                                                  self.return_softmax)
