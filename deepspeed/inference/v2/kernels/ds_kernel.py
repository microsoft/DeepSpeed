# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from abc import ABC, abstractmethod
from typing import Optional

import torch

from deepspeed.inference.v2.logging import inference_logger


class DSKernelBase(ABC):

    _workspace: Optional[torch.Tensor] = None

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """
        If necessary trigger compilation and warmup
        Autotuning of the kernel would happen at this stage to
        eliminate any potential hangs that might occur mid-deployment
        Validate that the desired run configuration is compatible.

        It is not necessary to call super on this method.
        """
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        However the kernel needs to be called, it can be called here. Auto-tuning
        should never be performed here.

        All inputs/outputs should be passed as arguments to this function. No allocations
        should be performed here.
        """
        raise NotImplementedError()

    def requested_workspace_size(self) -> int:
        """
        Return the requested workspace size in bytes.

        This should be overloaded if the kernel requires a workspace.

        Returns:
            int: Number of bytes necessary.
        """
        return 0

    def get_workspace(self, bytes: int) -> torch.Tensor:
        """
        Return the data pointer to the scratchpad memory.

        Args:
            bytes (int): Number of bytes necessary.

        Raises:
            RuntimeError: If the workspace is not allocated.
            ValueError: If the workspace is not large enough.
        """
        if DSKernelBase._workspace is None:
            raise RuntimeError("Workspace not allocated")
        if DSKernelBase._workspace.numel() < bytes:
            raise ValueError("Workspace too small")
        return DSKernelBase._workspace

    @staticmethod
    def create_workspace(bytes: int) -> int:
        """
        Create a workspace of the requested size.

        Args:
            bytes (int): Number of bytes necessary.

        Raises:
            RuntimeError: If the workspace is already allocated.
        """
        if DSKernelBase._workspace is not None:
            raise RuntimeError("Workspace already allocated")

        if bytes > 0:
            inference_logger().info(f"Allocating {bytes} bytes of workspace")

        DSKernelBase._workspace = torch.empty((bytes, ), dtype=torch.uint8)
