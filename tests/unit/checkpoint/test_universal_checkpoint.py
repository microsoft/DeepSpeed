# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed
from types import SimpleNamespace
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.utils import required_torch_version
from deepspeed.checkpoint import UNIVERSAL_CHECKPOINT_INFO
from deepspeed.checkpoint.ds_to_universal import main as convert_to_universal

from unit.common import DistributedTest, DistributedFixture
from unit.simple_model import *

from unit.checkpoint.common import compare_opt_state_dicts, compare_model_states

import pytest


@pytest.mark.parametrize("load_optim", [False, True])
class TestZeROUniversalCheckpoint(DistributedTest):
    world_size = 2

    def test_universal_checkpoint_no_change(self, tmpdir, load_optim):
        ds_config = {
            "train_batch_size": 2,
            "optimizer": {
                "type": 'Adam'
            },
            "zero_optimization": {
                "stage": 2,
            }
        }
        if get_accelerator().is_fp16_supported():
            ds_config["fp16"] = {"enabled": True, "initial_scale_power": 8}
        elif get_accelerator().is_bf16_supported():
            ds_config["bf16"] = {"enabled": True}
        hidden_dim = 10
        test_step = 8

        # torch 1.2.* stores raw tensor id numbers in checkpoint state which leads to
        # false positive mismatches in checkpoint state comparisons.
        # Newer torch versions store tensor ids as 0, 1, 2, ...
        expected_mismatch_keys = [] if required_torch_version(min_version=1.4) else ['params']
        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=ds_config, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=test_step,
                                        hidden_dim=hidden_dim,
                                        device=model.device)
        for batch in data_loader:
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        if load_optim:
            trained_sd = model.optimizer.optimizer.state_dict()
        #     opt_state_dict_file = f'opt-state-dict_rank{dist.get_rank()}'
        #     torch.save(model.optimizer.optimizer.state_dict(), os.path.join(tmpdir, opt_state_dict_file))

        client_state = {}
        client_state[UNIVERSAL_CHECKPOINT_INFO] = {}
        client_state['iteration'] = test_step
        model.save_checkpoint(tmpdir, client_state=client_state)

        cp_dir = os.path.join(tmpdir, f"global_step{test_step}")
        univ_cp_dir = f"{cp_dir}_universal"

        args = SimpleNamespace(input_folder=cp_dir,
                               output_folder=univ_cp_dir,
                               num_extract_workers=1,
                               num_merge_workers=1,
                               keep_temp_folder=False,
                               strict=True)
        convert_to_universal(args)

        ds_config["checkpoint"] = {"load_universal": True}
        loaded_model = SimpleModel(hidden_dim)
        loaded_model, _, _, _ = deepspeed.initialize(config=ds_config, model=loaded_model, model_parameters=loaded_model.parameters())
        loaded_model.load_checkpoint(tmpdir, load_optimizer_states=load_optim)

        compare_model_states(model, loaded_model, compare_optimizer=load_optim, load_module_only=False)

        if load_optim:
            curr_sd = loaded_model.optimizer.optimizer.state_dict()
            compare_opt_state_dicts(curr_sd, trained_sd, expected_mismatch_keys)
