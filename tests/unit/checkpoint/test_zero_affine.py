# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint.affine import contiguous_split_map
from deepspeed.checkpoint.affine_transfer import PlanBudget, plan_transfer
from deepspeed.checkpoint.zero_affine import (zero_optimizer_storage_affine_maps, zero_optimizer_storage_shard,
                                              zero_param_storage_affine_map, zero_param_storage_shard,
                                              zero_storage_affine_map)
from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_optimizer_state
from unit.common import DistributedTest
from unit.simple_model import random_dataloader


def test_replicas_list_every_holder():
    full = torch.arange(6.)
    affine_map = zero_storage_affine_map(6, [4, 1, 6])

    assert set(affine_map.shard_shapes) == {1, 4, 6}
    for rank in (1, 4, 6):
        assert torch.equal(affine_map.extract(full, rank), full)

    asked = []

    def selector(request):
        asked.append(sorted(candidate.source_rank for candidate in request.candidates))
        return request.candidates[-1].source_rank

    plan_transfer(contiguous_split_map((6, ), [3, 3], 0), affine_map, PlanBudget(), replica_selector=selector)
    # Each target block may be read from any of the three replicas.
    assert asked == [[1, 4, 6], [1, 4, 6]]


def test_zero3_partitions_drop_padding():
    # 10 elements in partitions of 4: holders keep 4, 4 and 2 elements; the fourth is all padding.
    full = torch.arange(10.)
    affine_map = zero_storage_affine_map(10, [2, 0, 3, 1], partition_numel=4)

    assert affine_map.shard_shapes == {2: (4, ), 0: (4, ), 3: (2, ), 1: (0, )}
    assert torch.equal(affine_map.extract(full, 2), full[0:4])
    assert torch.equal(affine_map.extract(full, 0), full[4:8])
    assert torch.equal(affine_map.extract(full, 3), full[8:10])
    assert affine_map.extract(full, 1).numel() == 0


@pytest.mark.parametrize('numel, holders, partition', [(9, [0, 1], 4), (4, [0, 0], None), (4, [], None)])
def test_rejects_storage_that_cannot_hold_the_parameter(numel, holders, partition):
    with pytest.raises(ValueError):
        zero_storage_affine_map(numel, holders, partition)


@pytest.mark.parametrize('accessor', [zero_param_storage_affine_map, zero_param_storage_shard])
def test_rejects_quantized_zero3_parameter_storage(accessor):
    param = torch.nn.Parameter(torch.zeros(8), requires_grad=False)
    param.ds_id = 0
    param.ds_numel = 8
    param.ds_tensor = torch.zeros(4, dtype=torch.int8)
    param.ds_tensor.ds_numel = 4
    param.ds_tensor.ds_quant_scale = torch.ones(1)

    with pytest.raises(NotImplementedError, match='Quantized ZeRO-3 parameter storage'):
        accessor(param)


class StraddlingModel(torch.nn.Module):
    # 351 trainable elements split over two ranks put the boundary inside ``first.bias``, so one
    # parameter is held in part by each rank. ``frozen`` has no optimizer state.
    def __init__(self, hidden_dim):
        super().__init__()
        self.frozen = torch.nn.Linear(hidden_dim, hidden_dim)
        self.frozen.requires_grad_(False)
        self.first = torch.nn.Linear(hidden_dim, hidden_dim)
        self.second = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, y):
        return torch.nn.functional.cross_entropy(self.second(self.first(self.frozen(x))), y)


def _full_param(param):
    if hasattr(param, 'ds_id'):
        with deepspeed.zero.GatheredParameters([param]):
            return param.data.detach().clone().view(-1)
    return param.data.detach().clone().view(-1)


def _full_optimizer_value(param, key):
    if key is None:
        return safe_get_full_fp32_param(param).view(-1)
    return safe_get_full_optimizer_state(param, key).view(-1)


class TestZeroOptimizerStorageMaps(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize('zero_stage, bf16_optimizer', [(1, False), (2, False), (3, False), (1, True)])
    def test_maps_describe_live_storage(self, zero_stage, bf16_optimizer):
        if not get_accelerator().is_bf16_supported():
            pytest.skip('requires bf16')
        hidden_dim = 13
        config = {
            'train_micro_batch_size_per_gpu': 1,
            'optimizer': {
                'type': 'Adam',
                'params': {
                    'lr': 1e-2,
                    'torch_adam': True
                }
            },
            'bf16': {
                'enabled': True
            },
            'zero_optimization': {
                'stage': zero_stage
            },
        }
        if bf16_optimizer:
            # fp32 gradient accumulation under ZeRO-1 selects the bf16 optimizer.
            config['data_types'] = {'grad_accum_dtype': 'fp32'}
        if zero_stage == 3:
            with deepspeed.zero.Init(config_dict_or_path=config):
                model = StraddlingModel(hidden_dim)
        else:
            model = StraddlingModel(hidden_dim)
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
        params = list(engine.module.parameters())
        rank = dist.get_rank()

        if zero_stage != 3:
            # first.bias straddles the partition boundary, so every rank holds part of it.
            with pytest.raises(RuntimeError, match='first optimizer step'):
                zero_optimizer_storage_shard(engine.module.first.bias, 'exp_avg')

        for x, y in random_dataloader(engine, 3, hidden_dim, engine.device, torch.bfloat16):
            engine.backward(engine(x, y))
            engine.step()

        maps = zero_optimizer_storage_affine_maps(params)
        assert maps[0] is None and maps[1] is None
        assert zero_optimizer_storage_shard(engine.module.frozen.weight) is None
        for param, affine_map in zip(params[2:], maps[2:]):
            affine_map.validate_coverage()

            # The low-precision parameter is the fp32 master copy rounded, which checks the master
            # copy's map against storage that does not come from the optimizer.
            full_param = _full_param(param)
            assert torch.equal(_full_optimizer_value(param, None).to(full_param.dtype), full_param)
            param_map = zero_param_storage_affine_map(param)
            assert torch.equal(zero_param_storage_shard(param), param_map.extract(full_param, rank))

            for key in (None, 'exp_avg', 'exp_avg_sq'):
                full = _full_optimizer_value(param, key)
                assert torch.equal(zero_optimizer_storage_shard(param, key), affine_map.extract(full, rank))

            # The shard is ZeRO's storage: writing each rank's id into it must show up in the gathered
            # state exactly where the map places that rank's range.
            zero_optimizer_storage_shard(param, 'exp_avg').fill_(rank + 1)
            written = _full_optimizer_value(param, 'exp_avg')
            expected = torch.zeros_like(written)
            for holder, pieces in affine_map.pieces_by_rank.items():
                for piece in pieces:
                    expected[piece.source_offset:piece.source_offset + piece.shape[0]] = holder + 1
            assert torch.equal(written, expected)

        engine.destroy()
