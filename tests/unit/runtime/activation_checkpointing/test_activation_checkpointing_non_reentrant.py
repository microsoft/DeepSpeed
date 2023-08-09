# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# TODO: add tests with model parallelism for activation partitioning and other features.

from deepspeed.runtime.activation_checkpointing.checkpointing import non_reentrant_checkpoint
from test_activation_checkpointing import *
from test_activation_checkpointing import (_mixed_mask, _bool_to_float, _prep_inputs, _match_outputs)


def _compute(module, *inputs, do_checkpoint=False):
    if do_checkpoint:
        outputs = non_reentrant_checkpoint(module, *inputs)
    else:
        outputs = module(*inputs)

    if torch.is_tensor(outputs):
        outputs = (outputs, )

    sum(o.sum() for o in outputs if torch.is_tensor(o) and o.requires_grad).backward()

    grads = [p.grad for p in module.parameters()]
    input_grads = [inp.grad for inp in inputs if torch.is_tensor(inp)]

    return {
        'outputs': outputs,
        'module_grads': grads,
        'input_grads': input_grads,
    }


def _test_activation_checkpoint(module, *inputs):
    # Move to device
    module.to(get_accelerator().device_name())

    # Get rid of dropouts until we fork the RNG between tests.
    module.eval()

    module_ = deepcopy(module)
    inputs_ = _prep_inputs(*inputs)
    base = _compute(module_, *inputs_, do_checkpoint=False)

    module_ = deepcopy(module)
    inputs_ = _prep_inputs(*inputs)
    test = _compute(module_, *inputs_, do_checkpoint=True)

    for group in base.keys():
        for b, t in zip(base[group], test[group]):
            _match_outputs(b, t)


def _test_activation_checkpoint_ordering(module, expected_ordering, *inputs):
    # Move to device
    module.to(get_accelerator().device_name())

    # Get rid of dropouts until we fork the RNG between tests.
    module.eval()

    module_ = deepcopy(module)
    inputs_ = _prep_inputs(*inputs)
    test = _compute(module_, *inputs_, do_checkpoint=True)

    outputs = test['outputs']
    test_ordering = []
    for item in outputs:
        if type(item) in [list, tuple]:
            test_ordering += [torch.is_tensor(t) for t in item]
        else:
            test_ordering += [torch.is_tensor(item)]

    assert expected_ordering == test_ordering


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
