# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from unit.util import required_torch_version

def is_torch_elastic_compatible():
    return required_torch_version(min_version=1.11)
