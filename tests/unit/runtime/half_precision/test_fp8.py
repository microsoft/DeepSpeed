# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import deepspeed
import pytest
from unit.common import DistributedTest
from unit.simple_model import random_dataloader
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

@pytest.mark.parametrize("base_datatype", ["fp16"])
class TestFp8ComposabilityAcrossZero(DistributedTest):
    world_size = 1

    def test(self, base_datatype):
        hidden_dim = 4096
        if base_datatype == "fp16":
            model_dtype = torch.float16
        elif base_datatype == "bf16":
            model_dtype = torch.bfloat16
        else:
            model_dtype = torch.float32

        torch.random.manual_seed(15)
        # TransformerEngine Model
        model = te.Linear(hidden_dim, hidden_dim, bias=True, params_dtype=model_dtype).cuda()

        # Create FP8 recipe. Note: All input args are optional.
        fp8_recipe = recipe.DelayedScaling(fp8_format=recipe.Format.HYBRID,
                                           amax_history_len=16,
                                           amax_compute_algo="max")
        # config
        zero_stage = [0, 1, 2, 3]
        for stage in zero_stage:
            config = {
                "train_batch_size": 8,
                "gradient_accumulation_steps": 1,
                "optimizer": {
                    "type": "Adam",
                    "params": {
                        "lr": 0.00001
                    }
                },
                "zero_optimization": {
                    "stage": stage,
                },
                "fp16": {
                    "enabled": True,
                    "loss_scale": 0.1
                },
                "bf16": {
                    "enabled": False
                }
            }
            # Init DeepSpeed
            model, optimizer, _, _ = deepspeed.initialize(args=None, model=model,
                                                          model_parameters=model.parameters(), config=config)

            data = torch.randn(128, hidden_dim, device=model.device, dtype=model_dtype)
            for datum in data:
                # Enables autocasting for the forward pass
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    out = model(datum)
                loss = out.mean()
                model.backward(loss)
                model.step()
            print(loss)
