# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import deepspeed
import pytest
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum

from unit.common import DistributedTest
from copy import deepcopy
from transformers import BertConfig, BertForMaskedLM
from torch._dynamo.utils import clone_inputs


def rand_int_tensor(device, low, high, shape):
    return torch.randint(
        low,
        high,
        shape,
        device=device,
        dtype=torch.int64,
        requires_grad=False,
    )


def generate_inputs_for_model(model, model_name, bs, device, include_loss_args=False):
    seq_length = 512  # Bert
    # seq_length = 1024 # GPT2
    vocab_size = model.config.vocab_size
    input = rand_int_tensor(device, 0, vocab_size, (bs, seq_length))
    input_dict = {"input_ids": input}

    if include_loss_args:
        if model_name.endswith("PreTraining"):
            label_name = "next_sentence_label"
            input_dict["labels"] = (rand_int_tensor(device, 0, vocab_size, (bs, seq_length)), )
            input_dict[label_name] = rand_int_tensor(device, 0, 1, (bs, ))
        elif (model_name.endswith("MaskedLM") or model_name.endswith("HeadModel") or model_name.endswith("CausalLM")
              or model_name.endswith("DoubleHeadsModel")):
            input_dict["labels"] = rand_int_tensor(device, 0, vocab_size, (bs, seq_length))
        else:
            raise NotImplementedError(f"Class {model_name} unsupported for training test ")

    return input_dict


class TestCompilerCorrectness(DistributedTest):
    world_size = 2
    non_daemonic_procs = True

    @pytest.mark.parametrize('dtype', [torch.float32])
    #@pytest.mark.parametrize("zero_stage", [0])
    @pytest.mark.parametrize("zero_stage", [1, 2, 3])
    @pytest.mark.parametrize('offload_device', [OffloadDeviceEnum.none, OffloadDeviceEnum.cpu, OffloadDeviceEnum.nvme])
    #@pytest.mark.parametrize('offload_device', [OffloadDeviceEnum.none])
    #@pytest.mark.parametrize("compiler_mode", ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    @pytest.mark.parametrize("compiler_mode", ["default"])
    #@pytest.mark.parametrize("fullgraph", ["True", "False"])
    @pytest.mark.parametrize("fullgraph", [False])
    def test(self, zero_stage, dtype, offload_device, compiler_mode, fullgraph):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU does not support this test yet")

        if offload_device == OffloadDeviceEnum.nvme:
            if zero_stage != 3:
                pytest.skip(f"Nvme offload not supported for zero stage {zero_stage}")

        hidden_dim = 1024
        batch_size = 32
        config_dict = {
            "train_micro_batch_size_per_gpu": 8,
            "gradient_accumulation_steps": 1,
            "steps_per_print": 1,
            "zero_optimization": {
                "stage": zero_stage,
                "contiguous_gradients": True,
                "overlap_comm": True,
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-5
                }
            },
            "compile": {
                "enabled": False,
                "backend": "inductor",
                "kwargs": {
                    "mode": compiler_mode,
                }
            },
            "zero_allow_untested_optimizer": True,
            "wall_clock_breakdown": False
        }

        model = BertForMaskedLM(BertConfig(num_hidden_layers=1, hidden_dim=hidden_dim))
        example_inputs = generate_inputs_for_model(model,
                                                   "BertForMaskedLM",
                                                   batch_size,
                                                   "cuda",
                                                   include_loss_args=True)
        to_be_compiled_model = deepcopy(model)
        ds_model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        config_dict["compile"]["enabled"] = True
        config_dict["compile"]["backend"] = "inductor"
        ds_compiled_model, _, _, _ = deepspeed.initialize(config=config_dict,
                                                          model=to_be_compiled_model,
                                                          model_parameters=to_be_compiled_model.parameters())

        ds_loss = ds_model(**example_inputs)
        ds_model.backward(ds_loss[0])
        ds_model.step()

        cloned_inputs = clone_inputs(example_inputs)
        ds_compile_loss = ds_compiled_model(**cloned_inputs)
        ds_compiled_model.backward(ds_compile_loss[0])
        ds_compiled_model.step()
        print("ds_loss norm = ", ds_loss[0].norm(), " ds_compile_loss_norm", ds_compile_loss[0].norm())
        assert torch.allclose(ds_compile_loss[0], ds_loss[0], atol=1e-5)
