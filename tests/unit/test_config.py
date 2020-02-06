# A test on its own
import torch

# A test on its own
import deepspeed


def test_cuda():
    assert (torch.cuda.is_available())


def test_check_version():
    assert hasattr(deepspeed, "__git_hash__")
    assert hasattr(deepspeed, "__git_branch__")
    assert hasattr(deepspeed, "__version__")


# Demonstration of pytest's paramaterization
@pytest.mark.parametrize('number,color', [(1138, 'purple')])
def test_dist_args(number, color):
    """Outer test function with inputs from pytest.mark.parametrize(). Uses a distributed
    helper function.
    """
    @distributed_test(world_size=2)
    def _test_dist_args_helper(x, color='red'):
        assert dist.get_world_size() == 2
        assert x == 1138
        assert color == 'purple'

    """Ensure that we can parse args to distributed_test decorated functions. """
    _test_dist_args_helper(number, color=color)

@pytest.mark.parametrize('batch,micro_batch,gas,success', [(32,16,1,True),(32,8,2,True),(32,16,2,False)])
def test_batch_config(batch, micro_batch, gas, success):

    @distributed_test(world_size=2)
    def _test_batch_config(batch, micro_batch, gas):
        
    """Ensure that we can parse args to distributed_test decorated functions. """
    _test_dist_args_helper(number, color=color)



