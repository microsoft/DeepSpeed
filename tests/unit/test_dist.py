import torch.distributed as dist

from common import distributed_test


@distributed_test(world_size=3)
def test_init():
    assert dist.is_initialized()
    assert dist.get_world_size() == 3
    assert dist.get_rank() < 3


@distributed_test(world_size=2)
def _test_dist_args_helper(x, color='red'):
    assert dist.get_world_size() == 2
    assert x == 1138
    assert color == 'purple'


def test_dist_args(number, color):
    """Ensure that we can parse args to distributed_test decorated functions. """
    _test_dist_args_helper(number, color=color)
