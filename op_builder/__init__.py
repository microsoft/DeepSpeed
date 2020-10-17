from .builder import command_exists
from .cpu_adam import CPUAdamBuilder
from .fused_adam import FusedAdamBuilder
from .fused_lamb import FusedLambBuilder
from .sparse_attn import SparseAttnBuilder
from .transformer import TransformerBuilder
from .utils import UtilsBuilder

# List of all available ops
ALL_OPS = [
    CPUAdamBuilder(),
    FusedAdamBuilder(),
    FusedLambBuilder(),
    SparseAttnBuilder(),
    TransformerBuilder(stochastic_mode=True),
    TransformerBuilder(stochastic_mode=False),
    UtilsBuilder()
]
