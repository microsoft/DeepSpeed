import torch
from .transformer import TransformerBuilder


class StochasticTransformerBuilder(TransformerBuilder):
    BUILD_VAR = "DS_BUILD_STOCHASTIC_TRANSFORMER"
    OP_NAME = "stochastic_transformer_op"

    def __init__(self, name_prefix=''):
        super().__init__(name=self.OP_NAME, name_prefix=name_prefix)

    def nvcc_args(self):
        args = super().nvcc_args()
        args.append('-D__STOCHASTIC_MODE__')
        return args
