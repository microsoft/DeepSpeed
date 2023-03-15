import os
from types import SimpleNamespace

import torch
import pytest

import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus, partitioned_param_data_shape
import deepspeed.comm as dist

from unit.common import DistributedTest, get_master_port
from unit.simple_model import SimpleModel


def setup_serial_env():
    # Setup for a serial run
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = get_master_port()
    os.environ['LOCAL_RANK'] = '0'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'


def test_scattered_init_dist():
    setup_serial_env()
    assert not dist.is_initialized()
    with deepspeed.zero.Init():
        assert dist.is_initialized()


class TestScatterGather(DistributedTest):
    world_size = 2

    def test(self):
        with deepspeed.zero.Init():
            l = torch.nn.Linear(6, 3)
        assert l.weight.ds_status == ZeroParamStatus.NOT_AVAILABLE
        assert l.weight.shape == torch.Size(partitioned_param_data_shape)

        # Ensure there is no impact outside the context
        l2 = torch.nn.Linear(6, 3)
        assert not hasattr(l2.weight, 'ds_status')
        assert l2.weight.numel() == l2.in_features * l2.out_features

        with deepspeed.zero.GatheredParameters(l.weight):
            assert l.weight.ds_status == ZeroParamStatus.AVAILABLE
            assert l.weight.numel() == l.in_features * l.out_features


class TestGatherUpdate(DistributedTest):
    world_size = 2

    def test(self):
        with deepspeed.zero.Init():
            l = torch.nn.Linear(4, 2)
        assert l.weight.ds_status == ZeroParamStatus.NOT_AVAILABLE

        # Gather and make a change
        with deepspeed.zero.GatheredParameters(l.weight, modifier_rank=1):
            assert l.weight.ds_status == ZeroParamStatus.AVAILABLE
            if dist.get_rank() == 1:
                with torch.no_grad():
                    l.weight.zero_()

        # should now be scattered again

        # Now gather again and ensure the change is global
        with deepspeed.zero.GatheredParameters(l.weight):
            # all ranks compare
            assert torch.equal(l.weight, torch.zeros_like(l.weight))


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


class TestZeroGatheredParametersFree(DistributedTest):
    world_size = 1

    def test(self):
        config_dict = {"train_batch_size": 1, "zero_optimization": {"stage": 3}}
        hidden_dim = 10

        class MyModel(torch.nn.Module):
            def __init__(self, hidden_dim):
                super(MyModel, self).__init__()
                self.l1 = torch.nn.Linear(hidden_dim, hidden_dim)

        with deepspeed.zero.Init(config_dict_or_path=config_dict):
            model = MyModel(hidden_dim)

        with deepspeed.zero.GatheredParameters(list(model.parameters())):
            assert model.l1.weight.numel() != 0, "GatheredParameters should give a non-0-sized tensor"

        # on exit from `GatheredParameters` the gathered params should be freed and not leak memory
        assert model.l1.weight.numel() == 0, "outside of GatheredParameters the param should go back to be 0-sized"


class TestSerialContext(DistributedTest):
    world_size = 1
    init_distributed = False
    set_dist_env = False


class ModelContainer(torch.nn.Module):
    def __init__(self, dim=16, return_obj=False):
        super().__init__()
        self.dim = dim
        self.linear1 = torch.nn.Linear(dim, dim)
        self.dangler = DanglingAttention(dim, return_obj=return_obj)

    def forward(self, input):
        act1 = self.linear1(input)
        # bias is actually dangler.d_linear1.bias
        act2, bias = self.dangler(act1)
        return (act2 + bias).sum()


class DanglingExt(torch.nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.dim = dim
        self.container = ModelContainer(dim)

    def forward(self, input):
        out = self.container(input)

        # Make sure it's at the right level of the stack
        assert len(self._external_params) == 0
        assert len(self.container._external_params) == 1
        assert len(self.container.dangler._external_params) == 0
        # Ensure we have registered the original unmodified bias parameter as an ext param
        assert id(self.container.dangler.d_linear.bias
                  ) in self.container._external_params.keys()
        return out


def test_ext_param_return():
    setup_serial_env()

    net = DanglingExt()

    args = SimpleNamespace(local_rank=0)
    engine, optim, _, _ = deepspeed.initialize(args=args,
                                               model=net,
                                               model_parameters=net.parameters(),
                                               config=config)

    for _ in range(5):
        input = torch.rand(net.dim).to(engine.device).half()
        loss = engine(input)
        engine.backward(loss)
        engine.step()


@pytest.mark.skip('WIP')
def test_ext_param_returnobj():
    setup_serial_env()
    print()

    net = ModelContainer(return_obj=True)

    args = SimpleNamespace(local_rank=0)
    engine, optim, _, _ = deepspeed.initialize(args=args,
                                               model=net,
                                               model_parameters=net.parameters(),
                                               config=config)

    for _ in range(5):
        input = torch.rand(net.dim).to(engine.device).half()
        loss = engine(input)
        assert len(net._external_params) == 1
        assert len(net.dangler._external_params) == 0
        engine.backward(loss)
        engine.step()


class ModelContainerVariableOutputType(ModelContainer):
    def __init__(self, dim=16, output_type=dict):
        super().__init__()
        self.output_type = output_type
        self.dim = dim
        self.linear1 = torch.nn.Linear(dim, dim)

    def forward(self, input):
        act1 = self.linear1(input)
        if self.output_type is dict:
            return {'loss': act1.sum()}
        if self.output_type is torch.tensor:
            return act1.sum()


@pytest.mark.parametrize('output_type', [torch.tensor, dict, None])
def test_stage_3_output_type(output_type):
    setup_serial_env()
    print()

    net = ModelContainerVariableOutputType(output_type=output_type)

    args = SimpleNamespace(local_rank=0)
    engine, optim, _, _ = deepspeed.initialize(args=args,
                                               model=net,
                                               model_parameters=net.parameters(),
                                               config=config)

    for _ in range(1):
        input = torch.rand(net.dim).to(engine.device).half()
        loss = engine(input)
        if loss is not None:
            if isinstance(loss, dict):
                loss = loss['loss']
            engine.backward(loss)
            engine.step()


# Test that no sub-class or super-class is missed
class ConvX(torch.nn.Conv1d):
    def __init__(self, *args):
        super().__init__(*args)
        # This would not be partitioned before bugfix 5ca8167
        self.param_in = torch.nn.Parameter(torch.FloatTensor(5).uniform_())

    def forward(self, x):
        return x


class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvX(1, 3, 4)
        self.param = torch.nn.Parameter(torch.FloatTensor(5).uniform_())

    def forward(self, x):
        return x


def test_subclass_param():
    setup_serial_env()
    with deepspeed.zero.Init(config=config):
        model = ConvNet()

    assert model.param.ds_status == ZeroParamStatus.NOT_AVAILABLE
    assert model.conv1.param_in.ds_status == ZeroParamStatus.NOT_AVAILABLE


# test that sub-classes get params that aren't prematurely partitioned and thus requiring gathering
# fixed by https://github.com/microsoft/DeepSpeed/pull/1202
class GrandPa(torch.nn.Module):
    def __init__(self, *args):
        super().__init__(*args)
        self.param_grandpa = torch.nn.Parameter(torch.ones(5))
        self.param_grandpa.data = (self.param_grandpa.data +
                                   1).data  # test param is not yet partitioned


class Pa(GrandPa):
    def __init__(self, *args):
        super().__init__(*args)
        self.param_pa = torch.nn.Parameter(torch.ones(5))
        self.param_pa.data = (self.param_pa.data +
                              1).data  # test param is not yet partitioned
        self.param_grandpa.data = (self.param_grandpa.data +
                                   1).data  # test param is not yet partitioned


class Son(Pa):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.ones(5))
        self.param.data = (self.param.data + 1).data  # test param is not yet partitioned
        self.param_pa.data = (self.param_pa.data +
                              1).data  # test param is not yet partitioned
        self.param_grandpa.data = (self.param_grandpa.data +
                                   1).data  # test param is not yet partitioned


def test_subclass_param_init():
    setup_serial_env()
    with deepspeed.zero.Init(config=config):
        model = Son().cpu()

    # test that all params have been partitioned
    assert model.param_grandpa.ds_status == ZeroParamStatus.NOT_AVAILABLE
    assert model.param_pa.ds_status == ZeroParamStatus.NOT_AVAILABLE
    assert model.param.ds_status == ZeroParamStatus.NOT_AVAILABLE

    # test that the weights manipulation during each __init__ worked in all w/o needing gathering
    ones = torch.ones(5).half().cuda()
    with deepspeed.zero.GatheredParameters(model.parameters(recurse=False)):
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
