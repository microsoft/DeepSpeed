# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from ....inference_utils import DtypeEnum
from deepspeed.ops.op_builder import InferenceCoreBuilder
from typing import Tuple
from ... import DSKernelBase


class CUDAWf6Af16Linear(DSKernelBase):
    """
    Wrapper around the CUDA kernel of Wf6Af16 quantized linear.

    Performs z = x @ y
    """

    def __init__(self):
        self.inf_module = InferenceCoreBuilder().load()
        self.inf_module.create_handle()
        self.kernel = self.inf_module.cuda_wf6af16_linear


    def __call__(self, output: torch.Tensor, hidden_states: torch.Tensor, packed_weights: torch.Tensor, M: int, N: int, K: int) -> torch.Tensor:
        """
        Matmul kernel as implemented via CUDA directly. The input must be 2D or larger. If
        n-dimensional, the leading dimensions are folded into each other:
            2D: m = x.size(0)
            3D: m = x.size(0) * x.size(1)
            4D: m = x.size(0) * x.size(1) * x.size(2) (etc...)
        All inputs should be contiguous.

        Parameters:
            output (torch.Tensor): Output tensor. Shape is of [*, out_features]
            hidden_states (torch.Tensor): Input tensor. Shape is of [*, in_features]
            packed_weights (torch.Tensor): Input tensor. Shape is of [?], it maintains the 4-bit and 2-bit weights, 
                    and the quantization scales. The 2-bit weights and the scale are store in the attributes.

        Returns:
            z (torch.Tensor): Output tensor. Shape is of [m, n]
        """

        # TODO: deal with batched-matmul. As the current implementation only supports 2D input, we need to split the
        # batched-matmul into multiple 2D matmul.

        # Workaround packed_weights becomes a tuple. No idea why..
        weights_4bit = torch.stack(list(packed_weights.data), dim=0)
        weights_2bit = packed_weights.weights_2bit
        # TODO: determine the value of `split_k` based on the input size and GPU arch
        split_k = 1
        workspace = self.get_workspace(M, N, K, split_k, torch.float, hidden_states.device)
        self.kernel(output, hidden_states, weights_4bit,
                    weights_2bit, packed_weights.scales, workspace, M, N, K, split_k)
        return output

    def requested_workspace_size(self) -> int:
        """
        The workspace is for the split-K GEMM. The maximum num elements is: 
            max-split-key * max-accumulated-seq * max-hidden
        """
        max_split_k = 108 # This is the SM number of A100
        max_accumulated_seq = 768 # This is a valued determined by the implementation of FastGen
        max_hidden = 28672 # This is the `intermediate_size` of Llama-2-70b
        
        # Assumes fp16 data type, which occupies 2 bytes.
        return max_split_k * max_accumulated_seq * max_hidden * 2

