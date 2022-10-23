"""
Copyright 2021 The Microsoft DeepSpeed Team
Licensed under the MIT license.

Functionality of swapping optimizer tensors to/from (NVMe) storage devices.
"""
from deepspeed.accelerator.real_accelerator import get_accelerator
assert get_accelerator().create_op_builder("AsyncIOBuilder").is_compatible()
