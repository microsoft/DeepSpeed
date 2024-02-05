# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from ....inference_utils import DtypeEnum
from deepspeed.ops.op_builder import InferenceCoreBuilder
from ... import DSKernelBase


class CUDAWf6Af16Linear(DSKernelBase):
    """
    Wrapper around the CUDA kernel of Wf6Af16 quantized linear.

    Performs z = x @ y
    """
    supported_dtypes = [DtypeEnum.fp16]

    def __init__(self):
        self.inf_module = InferenceCoreBuilder().load()
        self.inf_module.create_handle()
        self.kernel = self.inf_module.cuda_wf6af16_linear

    def __call__(self, output: torch.Tensor, hidden_states: torch.Tensor, weights_2bit: torch.Tensor,
                 weights_4bit: torch.Tensor, scale: torch.Tensor, out_channels, tokens, in_channels) -> torch.Tensor:
        """
        Matmul kernel of FP6 weight-only quantized linear. All inputs should be contiguous.
        It does not support batched-matmul.

        Parameters:
            output (torch.Tensor): Output tensor. Shape is of [token_number, out_features]
            hidden_states (torch.Tensor): Input tensor. Shape is of [token_number, in_features]
            weights_2bit (torch.Tensor): Input tensor of the 2-bit slice. Shape is of [out_features*2/8, in_features]
            weights_4bit (torch.Tensor): Input tensor of the 4-bit slice. Shape is of [out_features*4/8, in_features]
            scale (torch.Tensor): Input tensor. Shape is of [out_features], since the scale is per output channel
            out_channels (int): The number of output channels
            tokens (int): The number of tokens
            in_channels (int): The number of input channels
        """

        if out_channels % 256 != 0 or in_channels % 64 != 0:
            raise ValueError("The out and in channel should be multiple of 256 and 64 respectively.")

        # TODO: optimize the heuristic of split k selection.
        split_k_dict = {15360: 3, 27648: 2, 5120: 10, 10240: 5, 57344: 7, 8192: 6, 21504: 5, 7168: 7, 28672: 7}
        split_k = 1
        if not tokens > 128 and out_channels in split_k_dict:
            split_k = split_k_dict[out_channels]
        workspace = self.get_workspace(out_channels, tokens, in_channels, split_k, torch.float, hidden_states.device)
        self.kernel(output, hidden_states, weights_2bit, weights_4bit, scale, workspace, out_channels, tokens,
                    in_channels, split_k)

    def get_workspace(self, out_channels: int, tokens: int, in_channels: int, split_k: int, dtype,
                      device) -> torch.Tensor:
        """
        Allocate workspace for the kernel. The workspace is used to store the intermediate results of the matmul before
        split-K. The split-K size is determined by the size of the matmul.
        """
        workspace = torch.empty((split_k, out_channels, tokens), dtype=dtype, device=device)

        return workspace
