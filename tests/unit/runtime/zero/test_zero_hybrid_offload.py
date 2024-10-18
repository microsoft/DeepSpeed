# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed.comm as dist
import torch

from unit.common import DistributedTest, preferred_dtype
from unit.simple_model import random_dataloader

import deepspeed
from deepspeed.accelerator import get_accelerator


class SimpleModel(torch.nn.Module):

    def __init__(self, hidden_dim):
        super(SimpleModel, self).__init__()
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim, bias=False),
             torch.nn.Linear(hidden_dim, hidden_dim, bias=False)])
        self.act = torch.nn.ReLU()
        self.cel = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        for m in self.linears:
            x = self.act(m(x))
        loss = self.cel(x, y)
        return x, loss


def run_model(model, config_dict, hidden_dim, dtype):
    model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
    data_loader = random_dataloader(model=model,
                                    total_samples=10,
                                    hidden_dim=hidden_dim,
                                    device=model.device,
                                    dtype=dtype)
    dist.barrier()

    assert all(p.numel() == 0 for p in model.parameters())

    with deepspeed.zero.ZeRO3HybridOffload(model, hidden_dim**2 + 100):
        # Has params on device?
        assert any(p.numel() > 0 for p in model.parameters()
                   if p.device == torch.device(get_accelerator().current_device())), "No params on device"
        # Has params on cpu?
        assert any(p.numel() > 0 for p in model.parameters() if p.device == torch.device('cpu')), "No params on cpu"

        for batch in data_loader:
            loss = model(batch[0], batch[1])
            loss = loss[1]

    # Needed in ZeRO 3. Not doing so can give memory leak
    model.destroy()


class TestZeRO3HybridOffload(DistributedTest):
    # Need multiple gpus to test possible hanging
    world_size = 2
    reuse_dist_env = True

    def test(self):
        hidden_dim = 128

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "zero_optimization": {
                "stage": 3,
                "stage3_prefetch_bucket_size": hidden_dim**2,
                "stage3_param_persistence_threshold": 0,
                "stage3_max_reuse_distance": 0,
            }
        }
        if get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True}
        elif get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}

        model = SimpleModel(hidden_dim)
        model.eval()
        run_model(model, config_dict, hidden_dim, preferred_dtype())
