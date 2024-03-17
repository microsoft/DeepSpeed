# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed
from types import SimpleNamespace
from torch.utils._pytree import tree_map

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.utils import required_torch_version
from deepspeed.checkpoint import UNIVERSAL_CHECKPOINT_INFO
from deepspeed.checkpoint.ds_to_universal import main as convert_to_universal

from unit.common import DistributedTest, DistributedFixture
from unit.simple_model import *

from unit.checkpoint.common import compare_opt_state_dicts, compare_model_states, compare_state_dicts

import pytest
import deepspeed.comm as dist


def get_expected_mismatch_keys():
    # torch 1.2.* stores raw tensor id numbers in checkpoint state which leads to
    # false positive mismatches in checkpoint state comparisons.
    # Newer torch versions store tensor ids as 0, 1, 2, ...
    return [] if required_torch_version(min_version=1.4) else ['params']


def gather_opt_state(optimizer_state):

    def gather_tensor(t):
        if torch.is_tensor(t):
            buffer = [torch.zeros_like(t.flatten()) for _ in range(dist.get_world_size())]
            dist.all_gather(buffer, t.flatten())
            return torch.cat(buffer)
        else:
            return t

    return tree_map(gather_tensor, optimizer_state)


def train_save_convert(ds_config, hidden_dim, load_optim, tmpdir, checkpoint_tag):
    test_step = 8

    model = SimpleModel(hidden_dim)
    model, _, _, _ = deepspeed.initialize(config=ds_config, model=model, model_parameters=model.parameters())
    data_loader = random_dataloader(model=model, total_samples=test_step, hidden_dim=hidden_dim, device=model.device)
    for batch in data_loader:
        loss = model(batch[0], batch[1])
        model.backward(loss)
        model.step()

    sd = model.optimizer.optimizer.state_dict() if load_optim else None

    client_state = {}
    client_state[UNIVERSAL_CHECKPOINT_INFO] = {}
    client_state['iteration'] = test_step
    model.save_checkpoint(tmpdir, tag=checkpoint_tag, client_state=client_state)

    cp_dir = os.path.join(tmpdir, checkpoint_tag)
    univ_cp_dir = f"{cp_dir}_universal"

    args = SimpleNamespace(input_folder=cp_dir,
                           output_folder=univ_cp_dir,
                           num_extract_workers=1,
                           num_merge_workers=1,
                           keep_temp_folder=False,
                           strict=True)
    convert_to_universal(args)

    model_state = model.state_dict()
    optimizer_state = None
    if load_optim:
        optimizer_state = gather_opt_state(model.optimizer.optimizer.state_dict())

    if dist.get_rank() == 0:
        torch.save((model_state, optimizer_state), os.path.join(tmpdir, "baseline_state.pt"))

    dist.barrier()

    return model, sd


@pytest.fixture
def ds_config(zero_stage):
    ds_config = {
        "train_batch_size": 2,
        "optimizer": {
            "type": 'Adam'
        },
        "zero_optimization": {
            "stage": zero_stage,
        }
    }
    if get_accelerator().is_fp16_supported():
        ds_config["fp16"] = {"enabled": True, "initial_scale_power": 8}
    elif get_accelerator().is_bf16_supported():
        ds_config["bf16"] = {"enabled": True}
    return ds_config


@pytest.mark.parametrize("load_optim", [False, True])
class TestZeROUniversalCheckpoint(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize('zero_stage', [1, 2])
    def test_universal_checkpoint_no_change(self, tmpdir, ds_config, zero_stage, load_optim):
        hidden_dim = 10
        tag = "test_tag"
        trained_model, trained_sd = train_save_convert(ds_config, hidden_dim, load_optim, tmpdir, tag)

        ds_config["checkpoint"] = {"load_universal": True}
        loaded_model = SimpleModel(hidden_dim)
        loaded_model, _, _, _ = deepspeed.initialize(config=ds_config,
                                                     model=loaded_model,
                                                     model_parameters=loaded_model.parameters())
        loaded_model.load_checkpoint(tmpdir, tag=f"{tag}_universal", load_optimizer_states=load_optim)

        compare_model_states(trained_model, loaded_model, compare_optimizer=load_optim, load_module_only=False)

        if load_optim:
            curr_sd = loaded_model.optimizer.optimizer.state_dict()
            compare_opt_state_dicts(curr_sd, trained_sd, get_expected_mismatch_keys())


class _baseline(DistributedFixture):
    world_size = None

    def run(self, tmpdir, ds_config, zero_stage, tag, load_optim):
        hidden_dim = 10
        train_save_convert(ds_config, hidden_dim, load_optim, tmpdir, tag)


class baseline_ws1(_baseline):
    world_size = 1


class baseline_ws2(_baseline):
    world_size = 2


class baseline_ws4(_baseline):
    world_size = 4


class TestExample2(DistributedTest):

    # # These tests are divided by baseline model worldsize and test model worldsize
    @pytest.mark.world_size(1)
    @pytest.mark.parametrize("zero_stage", [1])
    @pytest.mark.parametrize("load_optim", [True])
    @pytest.mark.parametrize("tag", ["test_tag"])
    def test_world_size_2to1(self, baseline_ws2, tmpdir, tag, ds_config, load_optim):
        print(f"model={baseline_ws2}")
        hidden_dim = 10
        loaded_model_state, loaded_optimizer_state = torch.load(f"{tmpdir}/baseline_state.pt")

        ds_config["checkpoint"] = {"load_universal": True}
        univ_model = SimpleModel(hidden_dim)
        univ_model, _, _, _ = deepspeed.initialize(config=ds_config,
                                                   model=univ_model,
                                                   model_parameters=univ_model.parameters())
        univ_model.load_checkpoint(tmpdir, tag=f"{tag}_universal", load_optimizer_states=load_optim)

        model_state = univ_model.state_dict()
        compare_state_dicts(model_state, loaded_model_state)

        if load_optim:
            optimizer_state = gather_opt_state(univ_model.optimizer.optimizer.state_dict())
            print(f"loaded_optimizer_state={loaded_optimizer_state} optimizer_state={optimizer_state}")

            compare_opt_state_dicts(optimizer_state, loaded_optimizer_state, get_expected_mismatch_keys())
