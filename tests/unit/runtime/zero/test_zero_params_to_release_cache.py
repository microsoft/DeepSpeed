# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import gc
import weakref

import torch

import deepspeed
import deepspeed.runtime.zero.parameter_offload as po_mod
import deepspeed.runtime.zero.partitioned_param_coordinator as ppc_mod
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.runtime.zero.partitioned_param_coordinator import (
    InflightParamRegistry,
    PartitionedParameterCoordinator,
    ZeRoTraceMode,
)
from unit.common import DistributedTest, preferred_dtype

PREFIX = "_PartitionedParameterCoordinator__"


def _make_module(ds_id, numel=4):
    module = torch.nn.Linear(numel, 1, bias=False)
    module.ds_id = ds_id
    module.ds_external_parameters = lambda: []
    module.ds_recompute_parameters = set()
    param = module.weight
    param.ds_id = ds_id
    param.ds_persist = False
    param.ds_numel = numel
    param.ds_status = ZeroParamStatus.AVAILABLE
    param.ds_active_sub_modules = {ds_id}
    param.is_external_param = False

    def _partition(free_data=True):
        param.ds_status = ZeroParamStatus.NOT_AVAILABLE

    param.partition = _partition
    return module, param


def _make_coordinator():
    return PartitionedParameterCoordinator(
        prefetch_bucket_sz=0,
        max_reuse_distance_in_numel=100,
        max_available_parameters_in_numel=10**9,
        allgather_stream=None,
        inflight_param_registry=InflightParamRegistry(),
    )


def _set_complete_trace(coordinator, order, step_id=1):
    # Set completed trace state directly to isolate release_sub_module without distributed comm.
    setattr(coordinator, PREFIX + "trace_mode", ZeRoTraceMode.COMPLETE)
    setattr(coordinator, PREFIX + "submodule_order", tuple(order))
    setattr(coordinator, PREFIX + "step_id", step_id)


def test_release_sub_module_follows_retraced_order(monkeypatch):
    monkeypatch.setattr(ppc_mod, "print_rank_0", lambda *args, **kwargs: None)
    a, param_a = _make_module(1)
    b, _ = _make_module(2)
    c, _ = _make_module(3)
    coordinator = _make_coordinator()

    # In trace [a, b, a], module `a` is reused at step 2 within max_reuse_distance_in_numel,
    # so releasing `a` at step 1 must keep `param_a` AVAILABLE.
    _set_complete_trace(coordinator, [a, b, a], step_id=1)
    coordinator.release_sub_module(a, forward=True)
    assert param_a.ds_status == ZeroParamStatus.AVAILABLE

    # After trace invalidation and re-recording [a, b, c], `a` is no longer reused after step 1,
    # so releasing `a` at step 1 must partition `param_a`.
    coordinator._clear_trace_structures()
    param_a.ds_active_sub_modules = {a.ds_id}
    _set_complete_trace(coordinator, [a, b, c], step_id=1)
    coordinator.release_sub_module(a, forward=True)
    assert param_a.ds_status == ZeroParamStatus.NOT_AVAILABLE


def test_backward_recompute_release_does_not_mutate_cached_forward_release(monkeypatch):
    monkeypatch.setattr(ppc_mod, "print_rank_0", lambda *args, **kwargs: None)
    a, param_a = _make_module(10)
    b, _ = _make_module(20)
    coordinator = _make_coordinator()

    # At step 1 in trace [a, b, a], `param_a` is reused at step 2 so normal release keeps it AVAILABLE.
    # During backward with activation checkpointing, `ds_recompute_parameters` forces `param_a` to release.
    _set_complete_trace(coordinator, [a, b, a], step_id=1)
    setattr(coordinator, PREFIX + "active_backward_submodules", {a.ds_id: a})
    a.ds_recompute_parameters.add(param_a)
    coordinator.release_sub_module(a, forward=False)
    assert param_a.ds_status == ZeroParamStatus.NOT_AVAILABLE

    # Subsequent forward release at the same (module, step_id) without recompute parameters must not
    # inherit the backward recompute release decision from the cache.
    param_a.ds_status = ZeroParamStatus.AVAILABLE
    param_a.ds_active_sub_modules = {a.ds_id}
    _set_complete_trace(coordinator, [a, b, a], step_id=1)
    coordinator.release_sub_module(a, forward=True)
    assert param_a.ds_status == ZeroParamStatus.AVAILABLE


def test_coordinator_and_modules_are_garbage_collected(monkeypatch):
    monkeypatch.setattr(ppc_mod, "print_rank_0", lambda *args, **kwargs: None)
    a, param_a = _make_module(1)
    coordinator = _make_coordinator()
    _set_complete_trace(coordinator, [a], step_id=0)
    coordinator.release_sub_module(a, forward=True)

    coordinator_ref = weakref.ref(coordinator)
    module_ref = weakref.ref(a)
    del coordinator, a, param_a
    gc.collect()

    assert coordinator_ref() is None
    assert module_ref() is None


class _TwoLayerModel(torch.nn.Module):

    def __init__(self, hidden_dim=16):
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class TestZero3CoordinatorCacheLifecycle(DistributedTest):
    world_size = 1

    def test_engine_coordinator_released_after_complete_trace(self):
        hidden = 16
        dtype = preferred_dtype()
        config = {
            "train_batch_size": 1,
            "fp16": {
                "enabled": dtype is torch.float16
            },
            "bf16": {
                "enabled": dtype is torch.bfloat16
            },
            "zero_optimization": {
                "stage": 3
            },
        }
        model = _TwoLayerModel(hidden)
        engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=config,
            model_parameters=list(model.parameters()),
        )

        x = torch.randn(2, hidden, dtype=dtype, device=engine.device)
        # First forward enters RECORD and records the trace; second forward enters COMPLETE
        # and populates the coordinator's release cache.
        engine(x)
        engine(x)

        coordinator = engine.optimizer.get_param_coordinator()
        assert coordinator.is_complete_trace()
        coordinator_ref = weakref.ref(coordinator)
        submodule_ref = weakref.ref(engine.module.fc1)

        engine.destroy()
        po_mod.FWD_MODULE_STACK.clear()
        del coordinator, engine, optimizer, model, x
        gc.collect()

        assert coordinator_ref() is None
        assert submodule_ref() is None
