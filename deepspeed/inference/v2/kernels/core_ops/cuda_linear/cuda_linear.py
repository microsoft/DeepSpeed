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


    def __call__(self, output: torch.Tensor, hidden_states: torch.Tensor, weights: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
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
            weights (torch.Tensor): Input tensor. Shape is of [out_features, in_features]
            scale (torch.Tensor): Input tensor. Shape is of [1] or [out_features], since the scale is per output channel

        Returns:
            z (torch.Tensor): Output tensor. Shape is of [m, n]
        """

        # TODO: deal with batched-matmul. As the current implementation only supports 2D input, we need to split the
        # batched-matmul into multiple 2D matmul.

        weights_4bit, weights_2bit = self.split_f6_weights(weights)
        M, N, K, split_k = self.get_matmul_size(hidden_states, weights)
        # TODO: the corrent work flow is to allocate the workspace on the fly for split-K. Need to check how the 
        # cuBLAS and CUTLASS do for the workspace allocation.
        workspace = self.get_workspace(M, N, K, split_k, hidden_states.dtype, hidden_states.device)

        self.kernel(output, hidden_states, weights_4bit,
                    weights_2bit, scale, workspace, M, N, K, split_k)
        return output

    def split_f6_weights(self, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split the weights into two tensors: one holding 2-bit content and the other holding 4-bit content
        """
        pass

    def get_workspace(M: int, N: int, K: int, split_k: int, dtype, device) -> torch.Tensor:
        """
        Allocate workspace for the kernel. The workspace is used to store the intermediate results of the matmul before
        split-K. The split-K size is determined by the size of the matmul.
        """
        workspace = torch.empty((split_k, M, N), dtype=dtype, device=device)
        # TODO: allocate workspace in advance to avoid memory allocation overhead

        return workspace
