import argparse
import torch
import pytest
import random
from torch import nn
from deepspeed import DeepSpeedLSHLayer

import sys

#device = torch.device("cuda")
device = torch.device("cpu")
kwargs_fp32 = {'dtype': torch.float, 'device': device, 'requires_grad': True}
kwargs_fp16 = {'dtype': torch.half, 'device': device, 'requires_grad': True}


# FP16 test cases can only run on the devices support FP16.
@pytest.mark.parametrize('bsz, weight_dim0, weight_dim1, use_fp16',
                         [
                             (10, 30528,1024,False),
                         ]) # yapf: disable
def test_forward(bsz, weight_dim0, weight_dim1, use_fp16):
    kwargs = kwargs_fp16 if use_fp16 else kwargs_fp32

    weights = torch.randn(weight_dim0, weight_dim1, **kwargs)
    input = torch.randn(bsz, weight_dim1, **kwargs)

    lsh = DeepSpeedLSHLayer(weights)
    voc_index = lsh(input)
