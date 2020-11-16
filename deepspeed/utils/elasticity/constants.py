"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

#########################################
# Elasticity
#########################################
''' Elasticity Utility in DeepSpeed can be used to create highly elastic jobs compatible with 
a large number of GPUs. For elastic jobs, DeepSpeed will provide a batch size that can 
support a large number of GPUs based on the user specified parameters
'''
ELASTICITY_FORMAT = '''
Elasticity  should be enabled as:
"session_params": {
  "elasticity": {
    "enabled": [true|false],
    "max_acceptable_batch_size": 2000,
    "micro_batches": [2,4,6],
    "min_gpus": 1,
    "max_gpus" : 10000
    }
}
'''

ELASTICITY='elasticity'
ELASTICITY_ENABLED='enabled'
ELASTICITY_ENABLED_DEFAULT=False
ELASTICITY_MAX_ACCEPTABLE_BATCH_SIZE='max_acceptable_batch_size'
ELASTICITY_MAX_ACCEPTABLE_BATCH_SIZE_DEFAULT=2000
ELASTICITY_MICRO_BATCHES='micro_batches'
ELASTICITY_MICRO_BATCHES_DEFAULT=[2,4,6]
ELASTICITY_MIN_GPUS='min_gpus'
ELASTICITY_MIN_GPUS_DEFAULT=1
ELASTICITY_MAX_GPUS='max_gpus'
ELASTICITY_MAX_GPUS_DEFAULT=10000


