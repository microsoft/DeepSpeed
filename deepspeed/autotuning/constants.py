"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

#########################################
# autotunner implementation constants
#########################################

import os

DEFAULT_TEMPLATE_PATH_ZERO_0 = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            "config_templates",
                                            "template_zero0.json")
DEFAULT_TEMPLATE_PATH_ZERO_1 = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            "config_templates",
                                            "template_zero1.json")
DEFAULT_TEMPLATE_PATH_ZERO_2 = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            "config_templates",
                                            "template_zero2.json")
DEFAULT_TEMPLATE_PATH_ZERO_3 = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            "config_templates",
                                            "template_zero3.json")

DEFAULT_EXPRS_DIR = os.path.join(os.getcwd(), "autotuning_exps")
DEFAULT_RESULTS_DIR = os.path.join(os.getcwd(), "autotuning_results")

METRIC_PERCENT_DIFF_CONST = 0.05
DS_CONFIG = "ds_config"
BUFSIZE = 1  # line buffer size for writing files

#########################################
# MODEL INFO
#########################################
AUTOTUNING_MODEL_INFO_PATH = "model_info_path"
AUTOTUNING_MODEL_INFO_PATH_DEFAULT = None

MODEL_INFO_FORMAT = '''
"model_info": {
  "num_params": 1000000000,
  "hidden_size": 10,
  "num_layers": 12,
}
'''
MODEL_INFO = "model_info"
MODEL_INFO_PROFILE = "profile"
MODEL_INFO_PROFILE_DEFAULT = False
MODEL_INFO_NUM_PARAMS = "num_params"
MODEL_INFO_NUM_PARAMS_DEFAULT = None
MODEL_INFO_HIDDEN_SIZE = "hideen_size"
MODEL_INFO_HIDDEN_SIZE_DEFAULT = None
MODEL_INFO_NUM_LAYERS = "num_layers"
MODEL_INFO_NUM_LAYERS_DEFAULT = None

MODEL_INFO_KEY_DEFAULT_DICT = {
    MODEL_INFO_PROFILE: MODEL_INFO_PROFILE_DEFAULT,
    MODEL_INFO_NUM_PARAMS: MODEL_INFO_NUM_PARAMS_DEFAULT,
    MODEL_INFO_HIDDEN_SIZE: MODEL_INFO_HIDDEN_SIZE_DEFAULT,
    MODEL_INFO_NUM_LAYERS: MODEL_INFO_NUM_LAYERS_DEFAULT
}

#########################################
# autotunner search space constants
#########################################

DEFAULT_HF_CONFIG = {
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
}

DEFAULT_MIN_MEM_CONFIG = {
    "train_micro_batch_size_per_gpu": 1,
    "zero_optimization": {
        "stage": 3
    },
    "memory_break_down": False
}

DEFAULT_TUNING_SPACE_ZERO_0 = {"zero_optimization": {"stage": 0}}

DEFAULT_TUNING_SPACE_ZERO_1 = {
    "zero_optimization": {
        "stage": 1,
        "reduce_bucket_size": [5e7,
                               5e8,
                               1e9],
        "allgather_bucket_size": [5e7,
                                  5e8,
                                  1e9],
    }
}

DEFAULT_TUNING_SPACE_ZERO_2 = {
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": [True,
                         False],
        "reduce_scatter": [False,
                           True],
        "reduce_bucket_size": [5e7,
                               5e8,
                               1e9],
        "allgather_bucket_size": [5e7,
                                  5e8,
                                  1e9],
        "contiguous_gradients": [False,
                                 True]
    },
}

DEFAULT_TUNING_SPACE_ZERO_3 = {
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": [True,
                         False],
        "reduce_scatter": [False,
                           True],
        "reduce_bucket_size": [5e7,
                               5e8,
                               1e9],
        "allgather_partitions": [True,
                                 False],
        "allgather_bucket_size": [5e7,
                                  5e8,
                                  1e9],
        "contiguous_gradients": [False,
                                 True]
    },
}

GLOBAL_TUNING_SPACE = 'global'
TUNING_MICRO_BATCH_SIZE_PREFIX = "z"
