# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Optional

import os
import torch
import torch.nn.functional as F
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp
from deepspeed.utils.types import NormType


class MLPFunctions(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig):
        super(MLPFunctions, self).__init__(config)

    #def forward(self,
    #    return
