# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum

from unit.runtime.compile.common import DistributedCompileTest, compare_loss


class TestZeRO(DistributedCompileTest):
    world_size = 2

    @pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize('zero_stage', [1, 2, 3])
    @pytest.mark.parametrize('offload_device', [OffloadDeviceEnum.none, OffloadDeviceEnum.cpu, OffloadDeviceEnum.nvme])
    def test_zero_fragments(self, tmpdir, zero_stage, dtype, offload_device):
        if offload_device == OffloadDeviceEnum.nvme:
            if zero_stage != 3:
                pytest.skip(f"Nvme offload not supported for zero stage {zero_stage}")

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "zero_optimization": {
                "stage": zero_stage,
            },
            "compile": {
                "enabled": True,
                "backend": "inductor"
            }
        }

        if offload_device == OffloadDeviceEnum.cpu:
            config_dict["zero_optimization"]["offload_optimizer"] = {"device": offload_device}
        elif offload_device == OffloadDeviceEnum.nvme:
            config_dict["zero_optimization"]["offload_optimizer"] = {
                "device": offload_device,
                "nvme_path": str(tmpdir)
            }
        if dtype == torch.float16:
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        elif dtype == torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

        compare_loss(self, config_dict, dtype)
