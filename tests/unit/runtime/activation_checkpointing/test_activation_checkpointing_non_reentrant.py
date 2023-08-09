# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# TODO: add tests with model parallelism for activation partitioning and other features.

from deepspeed.runtime.activation_checkpointing.checkpointing import non_reentrant_checkpoint
import test_activation_checkpointing
from test_activation_checkpointing import *
from test_activation_checkpointing import (
    _bool_to_float, _compute, _match_outputs, _mixed_mask,
    _prep_inputs, _test_activation_checkpoint, _test_activation_checkpoint_ordering
)

ckpt = non_reentrant_checkpoint

# both bool and float are important, as bool is not differentiable
@pytest.mark.parametrize('mask', [
    _mixed_mask(),
    _bool_to_float(_mixed_mask()),
])
class TestActivationCheckpointWithoutGrad(DistributedTest):
    world_size = 1

    def test_ckpt_inputs1_outputs1(self, mask):
        module = torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        inputs = torch.rand(HIDDEN_DIM)
        _test_activation_checkpoint(module, inputs)

    def test_ckpt_inputs2_outputs1(self, mask):
        module = MaskedLinear(HIDDEN_DIM, HIDDEN_DIM)
        inputs = torch.rand(HIDDEN_DIM)
        _test_activation_checkpoint(module, inputs, mask)

    def test_ckpt_inputs2_outputs2(self, mask):
        module = MaskedLinearSeq(HIDDEN_DIM, HIDDEN_DIM)
        inputs = torch.rand(HIDDEN_DIM)
        _test_activation_checkpoint(module, inputs, mask)

    def test_ckpt_inputs2_outputs3(self, mask):
        module = MaskedLinearSeqDup(HIDDEN_DIM, HIDDEN_DIM)
        inputs = torch.rand(HIDDEN_DIM)
        _test_activation_checkpoint(module, inputs, mask)

    def test_ckpt_arg_none(self, mask):
        module = DropMaskLinear(HIDDEN_DIM, HIDDEN_DIM)
        inputs = (torch.rand(HIDDEN_DIM), None)
        _test_activation_checkpoint(module, *inputs)


@pytest.mark.parametrize('non_tensor', [None, 2, True, (None, 2.5), (None, True, torch.randn(HIDDEN_DIM))])
class TestCheckpointNonTensorWithoutGrad(DistributedTest):
    world_size = 1

    def test_ckpt_non_tensor_input(self, non_tensor):
        module = LinearNonTensorInput(HIDDEN_DIM, HIDDEN_DIM)
        inputs = torch.rand(HIDDEN_DIM)
        _test_activation_checkpoint(module, inputs, non_tensor)

    def test_ckpt_non_tensor_output(self, non_tensor):
        module = LinearNonTensorOutput(non_tensor)
        inputs = torch.rand(HIDDEN_DIM)
        _test_activation_checkpoint(module, inputs)


@pytest.mark.parametrize('non_tensor_output', [
    None, (torch.randn(HIDDEN_DIM), 2.5), (None, torch.randn(HIDDEN_DIM), True), (None, True, torch.randn(HIDDEN_DIM))
])
class TestCheckpointNonTensorOutputOrderingWithoutGrad(DistributedTest):
    world_size = 1

    def test_ckpt_non_tensor_output_ordering(self, non_tensor_output):
        module = LinearNonTensorOutput(non_tensor_output)
        inputs = torch.rand(HIDDEN_DIM)

        # First return is a tensor
        ordering = [True]
        if type(non_tensor_output) in [list, tuple]:
            ordering += [torch.is_tensor(t) for t in non_tensor_output]
        else:
            ordering += [torch.is_tensor(non_tensor_output)]
        _test_activation_checkpoint_ordering(module, ordering, inputs)