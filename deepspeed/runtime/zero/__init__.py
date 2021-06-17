"""
"Copyright 2020 The Microsoft DeepSpeed Team.
Licensed under the MIT license.
"""

from .partition_parameters import ZeroParamType
from .partition_parameters import ZeroParamStatus
from .partition_parameters import Init
from .partition_parameters import GatheredParameters
from .partition_parameters import register_external_parameter
from .linear import LinearFunctionForZeroStage3 as ds_linear
from .linear import LinearNoTransposeFunctionForZeroStage3 as ds_linear_no_transpose
from .tiling import TiledLinear
from .tiling import TiledLinearReturnBias
