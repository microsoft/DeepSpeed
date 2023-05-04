# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import deepspeed

from unit.common import DistributedTest
from unit.simple_model import *

from unit.checkpoint.common import *


class TestMiCSCheckpoint(DistributedTest):
    world_size = 4

    def _toy_model_config(self):

        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": 'Adam',
                "params": {
                    "lr": 0.00015,
                    "betas": [0.8, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 3e-7
                }
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            },
            "wall_clock_breakdown": True,
            "zero_optimization": {
                "stage": 3,
                "mics_shard_size": 2
            }
        }

        hidden_dim = 10
        with deepspeed.zero.MiCS_Init(config_dict_or_path=config_dict):
            models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]

        return config_dict, hidden_dim, models

    def test_load_optimizer_state(self, tmpdir):
        config_dict, hidden_dim, models = self._toy_model_config()
        checkpoint_correctness_verification(config_dict, models, hidden_dim, tmpdir, load_optimizer_states=True)

    def test_not_load_optimizer_state(self, tmpdir):
        config_dict, hidden_dim, models = self._toy_model_config()
        checkpoint_correctness_verification(config_dict, models, hidden_dim, tmpdir, load_optimizer_states=False)

    def test_load_module_only(self, tmpdir):
        config_dict, hidden_dim, models = self._toy_model_config()
        checkpoint_correctness_verification(config_dict, models, hidden_dim, tmpdir, load_module_only=True)
