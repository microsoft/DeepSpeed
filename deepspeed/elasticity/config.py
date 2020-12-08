"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
from .constants import *


class ElasticityError(Exception):
    pass


class ElasticityConfig:
    """
    "elasticity": {
        "enabled": true,
        "max_train_batch_size": 2000,
        "micro_batch_sizes": [2,4,6],
        "min_gpus": 1,
        "max_gpus" : 10000
        "min_time": 20
        "version": 0.1
    }
    """
    def __init__(self, param_dict):
        self.max_acceptable_batch_size = param_dict.get(
            MAX_ACCEPTABLE_BATCH_SIZE,
            MAX_ACCEPTABLE_BATCH_SIZE_DEFAULT)
        self.micro_batches = param_dict.get(MICRO_BATCHES, MICRO_BATCHES_DEFAULT)
        self.min_gpus = param_dict.get(MIN_GPUS, MIN_GPUS_DEFAULT)
        self.max_gpus = param_dict.get(MAX_GPUS, MAX_GPUS_DEFAULT)
        self.min_time = param_dict.get(MIN_TIME, MIN_TIME_DEFAULT)
        self.version = param_dict.get(VERSION, VERSION_DEFAULT)
        self.prefer_larger_batch_size = param_dict.get(PREFER_LARGER_BATCH,
                                                       PREFER_LARGER_BATCH_DEFAULT)

        #batch size computed based on elasticity config
        self.computed_batch_size = None

        #micro batch size for this run computed based on
        #elasticity configs, and the world size
        self.computed_micro_batch = None

        #gradient accumulation steps for this run computed based on
        #elasticity configs, and the world size
        self.computed_gradient_accumulation_step = None

        self._initialize()

    def _initialize(self):
        #batch size computed based on elasticity config
        self.computed_batch_size = None

        #micro batch size for this run computed based on
        #elasticity configs, and the world size
        self.computed_micro_batch = None

        #gradient accumulation steps for this run computed based on
        #elasticity configs, and the world size
        self.computed_gradient_accumulation_step = None
