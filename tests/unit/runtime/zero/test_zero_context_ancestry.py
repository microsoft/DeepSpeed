# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.accelerator import get_accelerator

from utils import setup_serial_env
from unit.common import DistributedTest

config = {
    "train_batch_size": 1,
    "steps_per_print": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.00015
        }
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 138.
    },
    "zero_optimization": {
        "stage": 3,
        "stage3_param_persistence_threshold": 1,
    }
}


# test that sub-classes get params that aren't prematurely partitioned and thus requiring gathering
# fixed by https://github.com/microsoft/DeepSpeed/pull/1202
class GrandPa(torch.nn.Module):

    def __init__(self, *args):
        super().__init__(*args)
        self.param_grandpa = torch.nn.Parameter(torch.ones(5))
        self.param_grandpa.data = (self.param_grandpa.data + 1).data  # test param is not yet partitioned


class Pa(GrandPa):

    def __init__(self, *args):
        super().__init__(*args)
        self.param_pa = torch.nn.Parameter(torch.ones(5))
        self.param_pa.data = (self.param_pa.data + 1).data  # test param is not yet partitioned
        self.param_grandpa.data = (self.param_grandpa.data + 1).data  # test param is not yet partitioned


class Son(Pa):

    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.ones(5))
        self.param.data = (self.param.data + 1).data  # test param is not yet partitioned
        self.param_pa.data = (self.param_pa.data + 1).data  # test param is not yet partitioned
        self.param_grandpa.data = (self.param_grandpa.data + 1).data  # test param is not yet partitioned


class TestSerialParamInit(DistributedTest):
    world_size = 1
    init_distributed = False
    set_dist_env = False

    def test_subclass_param_init(self):
        setup_serial_env()
        with deepspeed.zero.Init(config=config):
            model = Son().cpu()

        # test that all params have been partitioned
        assert model.param_grandpa.ds_status == ZeroParamStatus.NOT_AVAILABLE
        assert model.param_pa.ds_status == ZeroParamStatus.NOT_AVAILABLE
        assert model.param.ds_status == ZeroParamStatus.NOT_AVAILABLE

        # test that the weights manipulation during each __init__ worked in all w/o needing gathering
        ones = torch.ones(5).half().to(get_accelerator().device_name())
        with deepspeed.zero.GatheredParameters(list(model.parameters(recurse=False))):
            assert torch.equal(model.param, ones + 1)
            assert torch.equal(model.param_pa, ones + 2)
            assert torch.equal(model.param_grandpa, ones + 3)


class TestDSInitWZinit(DistributedTest):
    world_size = 2

    def test(self):
        ds_config = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            }
        }

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.linear = torch.nn.Linear(4, 4)

            def magic(self):
                return 42

        with deepspeed.zero.Init():
            model = Model()
            engine, *_ = deepspeed.initialize(model=model, config=ds_config, model_parameters=model.parameters())
        assert engine.magic() == 42


class TestNestingInit(DistributedTest):
    world_size = 1
    init_distributed = False
    set_dist_env = False

    def test_nesting_init(self):
        ds_config = dict(train_batch_size=1, zero_optimization=dict(stage=3))

        with deepspeed.zero.Init(config_dict_or_path=ds_config):
            with deepspeed.zero.Init(config_dict_or_path=ds_config):
                model = torch.nn.Linear(4, 4)

        deepspeed_engine, *_ = deepspeed.initialize(model=model, config_params=ds_config)
        # ensure that zero3 processed the parameter
        assert hasattr(deepspeed_engine.weight, "ds_id")


class TestNewClassDeclaredInsideInit(DistributedTest):
    world_size = 1
    init_distributed = False
    set_dist_env = False

    def test_new_class_declared_inside_init(self):
        ds_config = dict(train_batch_size=1, zero_optimization=dict(stage=3))

        with deepspeed.zero.Init(config_dict_or_path=ds_config):

            class MyModel(torch.nn.Module):

                def __init__(self):
                    super().__init__()
                    self.fc = torch.nn.Linear(4, 4)

            with deepspeed.zero.Init(config_dict_or_path=ds_config):
                model = MyModel()

        deepspeed_engine, *_ = deepspeed.initialize(model=model, config_params=ds_config)
        # ensure that zero3 processed the parameter
        assert hasattr(deepspeed_engine.fc.weight, "ds_id")


class TestNewClassDeclaredInsideInitFailure(DistributedTest):
    world_size = 1
    init_distributed = False
    set_dist_env = False

    def test_new_class_declared_inside_init_failure(self):
        ds_config = dict(train_batch_size=1, zero_optimization=dict(stage=3))

        try:
            with deepspeed.zero.Init(config_dict_or_path=ds_config):

                class MyModel(torch.nn.Module):

                    def __init__(self):
                        super().__init__()
                        self.fc = torch.nn.Linear(1, 1)

                model = MyModel()

            assert False, "Should have failed. A subclass of torch.nn.Module must be defined before zero.Init() where an instance of the class is created."
        except RuntimeError as e:
            pass
        except:
            assert False, "Should have failed. Runtime error is expected."
