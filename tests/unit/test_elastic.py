import pytest
import deepspeed
from deepspeed.git_version_info import version as __version__


def test_old_version():
    ds_config = {
        "elasticity": {
            "enabled": True,
            "max_train_batch_size": 10000,
            "micro_batch_sizes": [8,
                                  12,
                                  16,
                                  17],
            "min_gpus": 32,
            "max_gpus": 1500,
            "min_time": 20,
            "version": 0.1
        }
    }

    final_batch_size, valid_gpus = deepspeed.elasticity.get_compatible_gpus(ds_config, target_deepspeed_version=__version__)
    assert len(valid_gpus) == 23
    assert final_batch_size == 9792
