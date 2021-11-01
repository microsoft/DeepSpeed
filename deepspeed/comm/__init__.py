import torch
from .utils import *

supported_torch_version = False

# See more details at: https://github.com/pytorch/pytorch/pull/48767
# The PG API in torch versions lesser than 1.8 are different so it is
# non-trivial to support both in the same API. We will just use the
# DS comm. backend in deepspeed/comm/comm.py if torch version if 1.8+.

if older_torch():
    supported_torch_version = False
    from torch.distributed import *
else:
    supported_torch_version = True
    from .comm import *
