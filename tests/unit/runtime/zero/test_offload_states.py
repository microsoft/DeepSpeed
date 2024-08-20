# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
import torch

from unit.common import DistributedTest
from unit.simple_model import random_dataloader, SimpleModel

import deepspeed
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum, OffloadStateTypeEnum


def run_model(model, config_dict, hidden_dim, dtype, include, pin_memory, non_blocking):
    model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
    data_loader = random_dataloader(model=model,
                                    total_samples=10,
                                    hidden_dim=hidden_dim,
                                    device=model.device,
                                    dtype=dtype)
    dist.barrier()
    for batch in data_loader:
        loss = model(batch[0], batch[1])
        model.backward(loss)
        model.step()

        alloc_before_offload = get_accelerator().memory_allocated()
        model.offload_states(include=include,
                             device=OffloadDeviceEnum.cpu,
                             pin_memory=pin_memory,
                             non_blocking=non_blocking)
        alloc_after_offload = get_accelerator().memory_allocated()
        assert alloc_after_offload < alloc_before_offload, f"Allocated memory should decrease after offload"
        model.offload_states_back()
        assert alloc_after_offload < get_accelerator().memory_allocated(
        ), f"Allocated memory should increase after offload back"

    # Needed in ZeRO 3. Not doing so can give memory leak
    model.destroy()


@pytest.mark.parametrize("included_state", [
    OffloadStateTypeEnum.hp_params, OffloadStateTypeEnum.lp_params, OffloadStateTypeEnum.opt_states,
    OffloadStateTypeEnum.lp_grads, OffloadStateTypeEnum.contiguous_grad_buffer, None
])
@pytest.mark.parametrize("pin_memory", [False, True])
@pytest.mark.parametrize("non_blocking", [False, True])
class TestOffloadStates(DistributedTest):
    # Need multiple gpus to test possible hanging
    world_size = 2

    def test_offload_states(self, included_state, pin_memory, non_blocking):
        hidden_dim = 1024

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "zero_optimization": {
                "stage": 3,
            }
        }
        config_dict["bf16"] = {"enabled": True}

        with deepspeed.zero.Init(config_dict_or_path=config_dict):
            model = SimpleModel(hidden_dim, nlayers=4)

        include = None if included_state is None else [included_state]
        run_model(model, config_dict, hidden_dim, torch.bfloat16, include, pin_memory, non_blocking)
