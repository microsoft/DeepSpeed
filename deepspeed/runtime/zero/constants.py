"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from .offload_constants import *

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
    "stage3_max_live_parameters" : 1000000000,
    "stage3_max_reuse_distance" : 1000000000,
    "allgather_partitions": [true|false],
    "allgather_bucket_size": 500000000,
    "reduce_scatter": [true|false],
    "contiguous_gradients" : [true|false]
    "overlap_comm": [true|false],
    "reduce_bucket_size": 500000000,
    "load_from_fp32_weights": [true|false],
    "cpu_offload": [true|false] (deprecated),
    "cpu_offload_params" : [true|false] (deprecated),
    "cpu_offload_use_pin_memory": [true|false] (deprecated),
    "sub_group_size" : 1000000000000,
    "offload_param": {...},
    "offload_optimizer": {...}
    }
}
'''

ZERO_OPTIMIZATION = 'zero_optimization'
ZERO_OPTIMIZATION_DISABLED = 0
ZERO_OPTIMIZATION_OPTIMIZER_STATES = 1
ZERO_OPTIMIZATION_GRADIENTS = 2
ZERO_OPTIMIZATION_WEIGHTS = 3
MAX_STAGE_ZERO_OPTIMIZATION = ZERO_OPTIMIZATION_WEIGHTS

ZERO_OPTIMIZATION_STAGE = 'stage'
ZERO_OPTIMIZATION_STAGE_1 = 'stage_1'
ZERO_OPTIMIZATION_STAGE_2 = 'stage_2'
ZERO_OPTIMIZATION_STAGE_3 = 'stage_3'

ZERO_OPTIMIZATION_STAGE_DEFAULT = ZERO_OPTIMIZATION_DISABLED

ZERO_OPTIMIZATION_ALLGATHER_PARTITIONS = 'allgather_partitions'
ZERO_OPTIMIZATION_ALLGATHER_PARTITIONS_DEFAULT = True

ZERO_OPTIMIZATION_REDUCE_SCATTER = 'reduce_scatter'
ZERO_OPTIMIZATION_REDUCE_SCATTER_DEFAULT = True

ZERO_OPTIMIZATION_OVERLAP_COMM = 'overlap_comm'
ZERO_OPTIMIZATION_OVERLAP_COMM_DEFAULT = False
ZERO3_OPTIMIZATION_OVERLAP_COMM_DEFAULT = True

ZERO_OPTIMIZATION_CONTIGUOUS_GRADIENTS = 'contiguous_gradients'
ZERO_OPTIMIZATION_CONTIGUOUS_GRADIENTS_DEFAULT = False
ZERO3_OPTIMIZATION_CONTIGUOUS_GRADIENTS_DEFAULT = False

ZERO_OPTIMIZATION_REDUCE_BUCKET_SIZE = 'reduce_bucket_size'
ZERO_OPTIMIZATION_REDUCE_BUCKET_SIZE_DEFAULT = 500000000

ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE = 'allgather_bucket_size'
ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE_DEFAULT = 500000000
ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE_DEPRECATED = 'allgather_size'
ZERO_OPTIMIZATION_LOAD_FROM_FP32_WEIGHTS = 'load_from_fp32_weights'
ZERO_OPTIMIZATION_LOAD_FROM_FP32_WEIGHTS_DEFAULT = True

ZERO_OPTIMIZATION_ELASTIC_CHECKPOINT = 'elastic_checkpoint'
ZERO_OPTIMIZATION_ELASTIC_CHECKPOINT_DEFAULT = True

ZERO_OPTIMIZATION_CPU_OFFLOAD = 'cpu_offload'
ZERO_OPTIMIZATION_CPU_OFFLOAD_DEFAULT = False

ZERO_OPTIMIZATION_CPU_OFFLOAD_PARAMS = 'cpu_offload_params'
ZERO_OPTIMIZATION_CPU_OFFLOAD_PARAMS_DEFAULT = False

ZERO_OPTIMIZATION_CPU_OFFLOAD_USE_PIN_MEMORY = 'cpu_offload_use_pin_memory'
ZERO_OPTIMIZATION_CPU_OFFLOAD_USE_PIN_MEMORY_DEFAULT = False

ZERO_OPTIMIZATION_OFFLOAD_PARAM = OFFLOAD_PARAM
ZERO_OPTIMIZATION_OFFLOAD_PARAM_DEFAULT = None

ZERO_OPTIMIZATION_OFFLOAD_OPTIMIZER = OFFLOAD_OPTIMIZER
ZERO_OPTIMIZATION_OFFLOAD_OPTIMIZER_DEFAULT = None

ZERO_OPTIMIZATION_SUB_GROUP_SIZE = 'sub_group_size'
ZERO_OPTIMIZATION_SUB_GROUP_SIZE_DEFAULT = 1000000000000

#maximum number of parameters per GPU before releasing them
ZERO_OPTIMIZATION_MAX_LIVE_PARAMETERS = 'stage3_max_live_parameters'
ZERO_OPTIMIZATION_MAX_LIVE_PARAMETERS_DEFAULT = 1000000000

#release a parameter only if the reuse distance is larger than specified
ZERO_OPTIMIZATION_MAX_REUSE_DISTANCE = 'stage3_max_reuse_distance'
ZERO_OPTIMIZATION_MAX_REUSE_DISTANCE_DEFAULT = 1000000000

ZERO_OPTIMIZATION_PREFETCH_BUCKET_SIZE = 'stage3_prefetch_bucket_size'
ZERO_OPTIMIZATION_PREFETCH_BUCKET_SIZE_DEFAULT = 50000000

#parameters smaller than the threshold are only communicated once after the
#parameters are updated and are persisted thoughout the trainging
#avoid tons of latency bound communication
ZERO_OPTIMIZATION_PARAM_PERSISTENCE_THRESHOLD = 'stage3_param_persistence_threshold'
ZERO_OPTIMIZATION_PARAM_PERSISTENCE_THRESHOLD_DEFAULT = 100000

# gathers params for saving a model - inefficient but is required in certain situations
ZERO_OPTIMIZATION_GATHER_FP16_WEIGHTS_ON_MODEL_SAVE = 'stage3_gather_fp16_weights_on_model_save'
ZERO_OPTIMIZATION_GATHER_FP16_WEIGHTS_ON_MODEL_SAVE_DEFAULT = False

ZERO_OPTIMIZATION_DEFAULT = {
    ZERO_OPTIMIZATION_STAGE:
    ZERO_OPTIMIZATION_STAGE_DEFAULT,
    ZERO_OPTIMIZATION_CONTIGUOUS_GRADIENTS:
    ZERO_OPTIMIZATION_CONTIGUOUS_GRADIENTS_DEFAULT,
    ZERO_OPTIMIZATION_REDUCE_SCATTER:
    ZERO_OPTIMIZATION_REDUCE_SCATTER_DEFAULT,
    ZERO_OPTIMIZATION_REDUCE_BUCKET_SIZE:
    ZERO_OPTIMIZATION_REDUCE_BUCKET_SIZE_DEFAULT,
    ZERO_OPTIMIZATION_ALLGATHER_PARTITIONS:
    ZERO_OPTIMIZATION_ALLGATHER_PARTITIONS_DEFAULT,
    ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE:
    ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE_DEFAULT,
    ZERO_OPTIMIZATION_LOAD_FROM_FP32_WEIGHTS:
    ZERO_OPTIMIZATION_LOAD_FROM_FP32_WEIGHTS_DEFAULT,
    ZERO_OPTIMIZATION_ELASTIC_CHECKPOINT:
    ZERO_OPTIMIZATION_ELASTIC_CHECKPOINT_DEFAULT,
    ZERO_OPTIMIZATION_OFFLOAD_PARAM:
    ZERO_OPTIMIZATION_OFFLOAD_PARAM_DEFAULT,
    ZERO_OPTIMIZATION_OFFLOAD_OPTIMIZER:
    ZERO_OPTIMIZATION_OFFLOAD_OPTIMIZER_DEFAULT,
    ZERO_OPTIMIZATION_SUB_GROUP_SIZE:
    ZERO_OPTIMIZATION_SUB_GROUP_SIZE_DEFAULT,
    ZERO_OPTIMIZATION_MAX_LIVE_PARAMETERS:
    ZERO_OPTIMIZATION_MAX_LIVE_PARAMETERS_DEFAULT,
    ZERO_OPTIMIZATION_MAX_REUSE_DISTANCE:
    ZERO_OPTIMIZATION_MAX_REUSE_DISTANCE_DEFAULT,
    ZERO_OPTIMIZATION_PREFETCH_BUCKET_SIZE:
    ZERO_OPTIMIZATION_PREFETCH_BUCKET_SIZE_DEFAULT,
    ZERO_OPTIMIZATION_PARAM_PERSISTENCE_THRESHOLD:
    ZERO_OPTIMIZATION_PARAM_PERSISTENCE_THRESHOLD_DEFAULT,
    ZERO_OPTIMIZATION_GATHER_FP16_WEIGHTS_ON_MODEL_SAVE:
    ZERO_OPTIMIZATION_GATHER_FP16_WEIGHTS_ON_MODEL_SAVE_DEFAULT
}
