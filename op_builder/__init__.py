"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
from deepspeed.accelerator import get_accelerator
from .cpu_adam import CPUAdamBuilder
from .cpu_adagrad import CPUAdagradBuilder
from .fused_adam import FusedAdamBuilder
from .fused_lamb import FusedLambBuilder
from .sparse_attn import SparseAttnBuilder
from .transformer import TransformerBuilder
from .stochastic_transformer import StochasticTransformerBuilder
from .utils import UtilsBuilder
from .async_io import AsyncIOBuilder
from .transformer_inference import InferenceBuilder
from .quantizer import QuantizerBuilder
from .builder import get_default_compute_capabilities, OpBuilder
from .dropout import DropoutBuilder
from .feedforward import FeedForwardBuilder
from .gelu import GeluBuilder
from .layer_reorder import LayerReorderBuilder
from .normalize import NormalizeBuilder
from .softmax import SoftmaxBuilder
from .stridedbatchgemm import StridedBatchGemmBuilder

# TODO: infer this list instead of hard coded
# List of all available ops
__op_builders__ = [
    get_accelerator().create_op_builder("CPUAdamBuilder"),
    get_accelerator().create_op_builder("CPUAdagradBuilder"),
    get_accelerator().create_op_builder("DropoutBuilder"),
    get_accelerator().create_op_builder("FeedForwardBuilder"),
    get_accelerator().create_op_builder("FusedAdamBuilder"),
    get_accelerator().create_op_builder("FusedLambBuilder"),
    get_accelerator().create_op_builder("GeluBuilder"),
    get_accelerator().create_op_builder("LayerReorderBuilder"),
    get_accelerator().create_op_builder("NormalizeBuilder"),
    get_accelerator().create_op_builder("SoftmaxBuilder"),
    get_accelerator().create_op_builder("SparseAttnBuilder"),
    get_accelerator().create_op_builder("TransformerBuilder"),
    get_accelerator().create_op_builder("StochasticTransformerBuilder"),
    get_accelerator().create_op_builder("StridedBatchGemmBuilder"),
    get_accelerator().create_op_builder("AsyncIOBuilder"),
    get_accelerator().create_op_builder("UtilsBuilder"),
    get_accelerator().create_op_builder("QuantizerBuilder"),
    get_accelerator().create_op_builder("InferenceBuilder")
]
ALL_OPS = {op.name: op for op in __op_builders__ if op is not None}
