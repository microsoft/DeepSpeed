# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy

import pytest
import torch
import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import groups
from unit.common import DistributedTest


def skip_on_device():
    if get_accelerator().device_name() == "xpu":
        pytest.skip("XPU requires a higher version for test")


def build_qwen3_model():
    from transformers import AutoModelForCausalLM, Qwen3Config

    # Construct locally so the test does not download model configuration or weights. q_norm/k_norm act on the
    # sharded heads and the decoder norms on the split sequence, so both gradient reductions are exercised.
    config = Qwen3Config(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    return config, AutoModelForCausalLM.from_config(config)


# plain: separate collectives; overlap: one autograd node per block with the backward reduce-scatter overlapped.
MODES = {
    "plain": {
        "sequence_sharding_overlap": False
    },
    "overlap": {
        "sequence_sharding_overlap": True
    },
}


def build_config(zero_stage, mode):
    return {
        "train_micro_batch_size_per_gpu": 2,
        "tensor_parallel": {
            "autotp_size": 2,
            "sequence_sharding": True,
            **MODES[mode],
        },
        # Plain SGD keeps a weight difference proportional to the gradient difference; Adam's normalized update
        # turns rounding-level gradient differences into 1e-5 weight differences.
        "optimizer": {
            "type": "SGD",
            "params": {
                "lr": 0.1
            },
        },
        "zero_allow_untested_optimizer": True,
        # The reference does not clip; DeepSpeed clips the global gradient norm to 1.0 by default.
        "gradient_clipping": 0.0,
        "zero_optimization": {
            "stage": zero_stage
        },
    }


def tp_shared_input_ids(vocab_size, seq_len, step):
    generator = torch.Generator().manual_seed(step)
    input_ids = torch.randint(0, vocab_size, (2, seq_len), generator=generator)
    input_ids = input_ids.to(get_accelerator().current_device_name())
    dist.broadcast(input_ids,
                   src=groups.get_tensor_model_parallel_src_rank(),
                   group=groups.get_tensor_model_parallel_group())
    return input_ids


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("zero_stage", [1, 3])
class TestAutoTPSequenceSharding(DistributedTest):
    world_size = 2

    def test_matches_single_device_training(self, zero_stage, mode):
        """AutoTP with sequence sharding trains like the unsharded model under plain PyTorch SGD."""
        skip_on_device()
        pytest.importorskip("transformers")

        config, model = build_qwen3_model()
        reference = copy.deepcopy(model).to(get_accelerator().current_device_name())
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               model_parameters=model.parameters(),
                                               config=build_config(zero_stage, mode))
        # The decoder norms must run on this rank's half of the sequence, not on the whole of it.
        norm_lengths = []
        engine.module.model.layers[0].input_layernorm.register_forward_hook(
            lambda module, args, output: norm_lengths.append(args[0].shape[1]))

        # The first step learns which linears read each block input; later steps run them in one autograd node.
        for step in range(3):
            input_ids = tp_shared_input_ids(config.vocab_size, 16, step)
            loss = engine(input_ids=input_ids, labels=input_ids).loss
            engine.backward(loss)
            engine.step()

            reference_loss = reference(input_ids=input_ids, labels=input_ids).loss
            reference_loss.backward()
            reference_optimizer.step()
            reference_optimizer.zero_grad()

            torch.testing.assert_close(loss.detach(), reference_loss.detach(), rtol=1e-5, atol=1e-5)

        assert norm_lengths == [16 // self.world_size] * 3
        if zero_stage < 3:
            # The decoder norms see half of the sequence on each rank; their summed gradient keeps them equal to
            # the reference after the updates.
            norm = engine.module.model.layers[0].input_layernorm.weight
            reference_norm = reference.model.layers[0].input_layernorm.weight
            torch.testing.assert_close(norm.detach(), reference_norm.detach(), rtol=1e-5, atol=1e-6)


class TestAutoTPSequenceShardingRejects(DistributedTest):
    world_size = 2

    def test_sequence_not_divisible_by_tp(self):
        skip_on_device()
        pytest.importorskip("transformers")

        config, model = build_qwen3_model()
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               model_parameters=model.parameters(),
                                               config=build_config(zero_stage=1, mode="overlap"))
        input_ids = tp_shared_input_ids(config.vocab_size, 15, 0)
        # pytest.raises inside a DistributedTest worker hangs the run when nothing is raised.
        try:
            engine(input_ids=input_ids, labels=input_ids)
            raised = False
        except ValueError as error:
            raised = "not divisible by the tensor-parallel size" in str(error)
        assert raised
