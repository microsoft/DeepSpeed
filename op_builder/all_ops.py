"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
from deepspeed.accelerator import get_accelerator

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
