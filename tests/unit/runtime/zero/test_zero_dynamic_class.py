# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from unit.common import DistributedTest

import deepspeed


class TestNewClassDeclaredInsideNestedInit(DistributedTest):
    world_size = 1

    def test_new_class_declared_inside_nested_init(self):
        ds_config = dict(train_batch_size=1, zero_optimization=dict(stage=3))

        with deepspeed.zero.Init(config_dict_or_path=ds_config):

            class MyModel(torch.nn.Module):

                def __init__(self):
                    super().__init__()
                    self.fc = torch.nn.Linear(4, 4)

            with deepspeed.zero.Init(config_dict_or_path=ds_config):
                model = MyModel()

        # ensure that zero3 processed the parameter
        assert hasattr(model.fc.weight, "ds_id")


class TestNewClassDeclaredInsideInit(DistributedTest):
    world_size = 1

    def test_new_class_declared_inside_init(self):
        ds_config = dict(train_batch_size=1, zero_optimization=dict(stage=3))

        with deepspeed.zero.Init(config_dict_or_path=ds_config):

            class MyModel(torch.nn.Module):

                def __init__(self):
                    super().__init__()
                    self.fc = torch.nn.Linear(1, 1)

            model = MyModel()
        # ensure that zero3 processed the parameter
        assert hasattr(model.fc.weight, "ds_id")
