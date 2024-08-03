# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

<<<<<<< HEAD:inference/v2/model_implementations/qwen_v2_moe/__init__.py
from .policy import Qwen2MoePolicy
=======
from .quantize import FP_Quantize, Quantizer

try:
    import triton
    from .fp8_gemm import matmul_fp8
except ImportError:
    pass
>>>>>>> 426445c324b56621271b3e609e7e9c49dc915892:deepspeed/ops/fp_quantizer/__init__.py
