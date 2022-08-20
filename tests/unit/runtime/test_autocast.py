import pytest
import torch
from deepspeed.runtime.zero.linear import LinearModuleForZeroStage3


def _skip_autocast_cpu_test(half_op):
    if not torch.cuda.is_available(
    ) and half_op:  # "addmv_impl_cpu" not implemented for 'Half'
        return True
    else:
        return False


@pytest.mark.parametrize('half_op', [False, True])
def test_missing_amp_autocast(tmpdir, half_op):
    if _skip_autocast_cpu_test(half_op):
        pytest.skip("CPU autocast is not supported for half")
    hidden_dim = 4
    input = torch.randn(hidden_dim)
    ds_linear = LinearModuleForZeroStage3(hidden_dim, hidden_dim)

    if torch.cuda.is_available():
        input = input.cuda()
        ds_linear = ds_linear.cuda()
    if half_op:
        input = input.half()
        ds_linear = ds_linear.half()

    output = ds_linear(input)
    assert output.dtype == ds_linear.weight.dtype


@pytest.mark.parametrize('half_op', [False, True])
def test_disable_autocast_linear(tmpdir, half_op):
    if _skip_autocast_cpu_test(half_op):
        pytest.skip("CPU autocast is not supported for half")
    amp = pytest.importorskip("torch.cuda.amp")

    hidden_dim = 4
    input = torch.randn(hidden_dim)
    ds_linear = LinearModuleForZeroStage3(hidden_dim, hidden_dim)

    if torch.cuda.is_available():
        input = input.cuda()
        ds_linear = ds_linear.cuda()
    if half_op:
        input = input.half()
        ds_linear = ds_linear.half()

    with amp.autocast(False):
        output = ds_linear(input)
        assert output.dtype == ds_linear.weight.dtype


@pytest.mark.parametrize('half_input, half_weight',
                         [(False,
                           False),
                          (False,
                           True),
                          (True,
                           False),
                          (True,
                           True)])
def test_autocast_linear(tmpdir, half_input, half_weight):
    if not torch.cuda.is_available():
        pytest.skip("amp autocast is not supported for CPU")
    amp = pytest.importorskip("torch.cuda.amp")

    hidden_dim = 4
    input = torch.randn(hidden_dim).cuda()
    ds_linear = LinearModuleForZeroStage3(hidden_dim, hidden_dim).cuda()

    if half_input:
        input = input.half()

    if half_weight:
        ds_linear = ds_linear.half()

    with amp.autocast():
        output = ds_linear(input)
        assert output.dtype == torch.half
