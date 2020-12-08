"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

#########################################
# Elasticity
#########################################
''' Elasticity Utility in DeepSpeed can be used to create highly elastic jobs compatible
with a large number of GPUs. For elastic jobs, DeepSpeed will provide a batch size that
can support a large number of GPUs based on the user specified parameters
'''
FORMAT = '''
Elasticity should be enabled as:
"elasticity": {
  "enabled": true,
  "max_train_batch_size": 2000,
  "micro_batch_sizes": [2,4,6],
  "min_gpus": 1,
  "max_gpus" : 10000
  "min_time": 20
  "version": 0.1
}
'''

ELASTICITY = 'elasticity'

ENABLED = 'enabled'
ENABLED_DEFAULT = False

# Max acceptable train_batch_size
MAX_ACCEPTABLE_BATCH_SIZE = 'max_train_batch_size'
MAX_ACCEPTABLE_BATCH_SIZE_DEFAULT = 2000

# Acceptable micro batch sizes, same as train_micro_batch_size_per_gpu
MICRO_BATCHES = 'micro_batch_sizes'
MICRO_BATCHES_DEFAULT = [2, 4, 6]

# Min/max of GPUs to search over
MIN_GPUS = 'min_gpus'
MIN_GPUS_DEFAULT = 1
MAX_GPUS = 'max_gpus'
MAX_GPUS_DEFAULT = 10000

# Minimum running time (minutes) before the scheduler will scale us
MIN_TIME = "min_time"
MIN_TIME_DEFAULT = "20"

PREFER_LARGER_BATCH = 'prefer_larger_batch'
PREFER_LARGER_BATCH_DEFAULT = True

# Version of elastic logic to use
VERSION = "version"
VERSION_DEFAULT = 0.1
