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

    # TODO: scale of quantization

    def __init__(self, fp_dtype: DtypeEnum):
        """
        Parameters:
            fp_dtype (torch.dtype): Data type for the input/output. Supported value
                is torch.float16.
        """
        fp_dtype = DtypeEnum(fp_dtype)
        if fp_dtype is not DtypeEnum.fp16:
            raise ValueError("Unsupported data type: {}, supported_dtypes is fp16".format(
                fp_dtype))

        self.inf_module = InferenceCoreBuilder().load()
        self.inf_module.create_handle()
        self.kernel = self.inf_module.cuda_wf6af16_linear

    def __call__(self, output: torch.Tensor, hidden_states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
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

        Returns:
            z (torch.Tensor): Output tensor. Shape is of [m, n]
        """

        # TODO: deal with the case of dimension number > 2

        self.kernel(output, hidden_states, weights)
        return output
