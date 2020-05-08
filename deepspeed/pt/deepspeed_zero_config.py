"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

#from deepspeed.pt.deepspeed_constants import *
from deepspeed.pt.deepspeed_config_utils import get_scalar_param

#########################################
# ZeRO optimization
#########################################
# ZeRO optimization. By default, this optimization is not enabled.
# Users have to configure the desired optimization (0 means disabled) in params.json as below example:
ZERO_FORMAT = '''
ZeRO optimization should be enabled as:
"session_params": {
  "zero_optimization": {
    "stage": [0|1|2],
    "allgather_partitions": [true|false],
    "allgather_bucket_size": 500000000,
    "reduce_scatter": [true|false],
    "reduce_bucket_size": 500000000
    }
}
'''

ZERO_OPTIMIZATION = 'zero_optimization'
ZERO_OPTIMIZATION_DISABLED = 0
ZERO_OPTIMIZATION_OPTIMIZER_STATES = 1
ZERO_OPTIMIZATION_GRADIENTS = 2
ZERO_OPTIMIZATION_WEIGHTS = 3
MAX_STAGE_ZERO_OPTIMIZATION = ZERO_OPTIMIZATION_GRADIENTS

ZERO_OPTIMIZATION_STAGE = 'stage'
ZERO_OPTIMIZATION_STAGE_1 = 'stage_1'
ZERO_OPTIMIZATION_STAGE_2 = 'stage_2'
ZERO_OPTIMIZATION_STAGE_3 = 'stage_3'

ZERO_OPTIMIZATION_STAGE_DEFAULT = ZERO_OPTIMIZATION_DISABLED

ZERO_OPTIMIZATION_ALLGATHER_PARTITIONS = 'allgather_partitions'
ZERO_OPTIMIZATION_ALLGATHER_PARTITIONS_DEFAULT = True

ZERO_OPTIMIZATION_REDUCE_SCATTER = 'reduce_scatter'
ZERO_OPTIMIZATION_REDUCE_SCATTER_DEFAULT = True

ZERO_OPTIMIZATION_CONTIGIOUS_GRADIENTS = 'contigious_gradients'
ZERO_OPTIMIZATION_CONTIGIOUS_GRADIENTS_DEFAULT = True

ZERO_OPTIMIZATION_REDUCE_BUCKET_SIZE = 'reduce_bucket_size'
ZERO_OPTIMIZATION_REDUCE_BUCKET_SIZE_DEFAULT = 500000000

ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE = 'allgather_bucket_size'
ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE_DEFAULT = 500000000

ZERO_OPTIMIZATION_DEFAULT = {
    ZERO_OPTIMIZATION_STAGE: ZERO_OPTIMIZATION_STAGE_DEFAULT,
    ZERO_OPTIMIZATION_CONTIGIOUS_GRADIENTS:
    ZERO_OPTIMIZATION_CONTIGIOUS_GRADIENTS_DEFAULT,
    ZERO_OPTIMIZATION_REDUCE_SCATTER: ZERO_OPTIMIZATION_REDUCE_SCATTER_DEFAULT,
    ZERO_OPTIMIZATION_REDUCE_BUCKET_SIZE: ZERO_OPTIMIZATION_REDUCE_BUCKET_SIZE_DEFAULT,
    ZERO_OPTIMIZATION_ALLGATHER_PARTITIONS:
    ZERO_OPTIMIZATION_ALLGATHER_PARTITIONS_DEFAULT,
    ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE:
    ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE_DEFAULT
}


class DeepSpeedZeroConfig(object):
    def __init__(self, param_dict):
        super(DeepSpeedZeroConfig, self).__init__()

        self.stage = None
        self.contigious_gradients = None
        self.reduce_scatter = None
        self.reduce_bucket_size = None
        self.allgather_partitions = None
        self.allgather_bucket_size = None

        if ZERO_OPTIMIZATION in param_dict.keys():
            zero_config_dict = param_dict[ZERO_OPTIMIZATION]
        else:
            zero_config_dict = ZERO_OPTIMIZATION_DEFAULT

        self._initialize(zero_config_dict)

    """
    For json serialization
    """

    def repr(self):
        return self.__dict__

    def _initialize(self, zero_config_dict):
        self.stage = get_scalar_param(zero_config_dict,
                                      ZERO_OPTIMIZATION_STAGE,
                                      ZERO_OPTIMIZATION_STAGE_DEFAULT)

        self.contigious_gradients = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_CONTIGIOUS_GRADIENTS,
            ZERO_OPTIMIZATION_CONTIGIOUS_GRADIENTS_DEFAULT)

        self.reduce_bucket_size = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_REDUCE_BUCKET_SIZE,
            ZERO_OPTIMIZATION_REDUCE_BUCKET_SIZE_DEFAULT)

        self.reduce_scatter = get_scalar_param(zero_config_dict,
                                               ZERO_OPTIMIZATION_REDUCE_SCATTER,
                                               ZERO_OPTIMIZATION_REDUCE_SCATTER_DEFAULT)

        self.allgather_partitions = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_ALLGATHER_PARTITIONS,
            ZERO_OPTIMIZATION_ALLGATHER_PARTITIONS_DEFAULT)

        self.allgather_bucket_size = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE,
            ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE_DEFAULT)
