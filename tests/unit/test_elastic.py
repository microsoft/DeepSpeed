import pytest
import deepspeed
from deepspeed.git_version_info import version as ds_version

base_ds_config = {
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


def test_basic_10k():
    ds_config = base_ds_config.copy()
    final_batch_size, valid_gpus = deepspeed.elasticity.get_compatible_gpus(
        ds_config=ds_config,
        target_deepspeed_version=ds_version)

    for gpu_num in valid_gpus:
        assert final_batch_size % gpu_num == 0, f"Batch {final_batch_size} is not divisible by GPU count {gpu_num}"
        batch_per_gpu = final_batch_size // gpu_num
        found_valid_mbsize = False

        for mb in ds_config['elasticity']['micro_batch_sizes']:
            if batch_per_gpu % mb == 0:
                found_valid_mb = True
                break
        assert found_valid_mb, "No valid mb found"

    assert len(valid_gpus) == 23
    assert final_batch_size == 9792


def test_old_version():
    ds_config = base_ds_config.copy()
    with pytest.raises(deepspeed.elasticity.config.ElasticityError):
        final_batch_size, valid_gpus = deepspeed.elasticity.get_compatible_gpus(
            ds_config=ds_config,
            target_deepspeed_version="0.2")


def test_disabled():
    ds_config = base_ds_config.copy()
    ds_config['elasticity']['enabled'] = False
    with pytest.raises(deepspeed.elasticity.config.ElasticityError):
        final_batch_size, valid_gpus = deepspeed.elasticity.get_compatible_gpus(
            ds_config=ds_config,
            target_deepspeed_version=ds_version)


def test_valid_world_size():
    ds_config = base_ds_config.copy()
    final_batch_size, valid_gpus, mbsize = deepspeed.elasticity.get_compatible_gpus(
            ds_config=ds_config,
            target_deepspeed_version=ds_version,
            world_size=64)
    assert mbsize == 17


def test_invalid_world_size():
    ds_config = base_ds_config.copy()
    with pytest.raises(deepspeed.elasticity.config.ElasticityIncompatibleWorldSize):
        final_batch_size, valid_gpus, mbsize = deepspeed.elasticity.get_compatible_gpus(
            ds_config=ds_config,
            target_deepspeed_version=ds_version,
            world_size=128)


def test_future_elastic_version():
    ds_config = base_ds_config.copy()
    ds_config['elasticity']['version'] = '0.2'
    with pytest.raises(deepspeed.elasticity.config.ElasticityError):
        final_batch_size, valid_gpus, mbsize = deepspeed.elasticity.get_compatible_gpus(
            ds_config=ds_config,
            target_deepspeed_version=ds_version)


def test_proper_mbsz():
    ds_config = base_ds_config.copy()
    ds_config["elasticity"]["max_train_batch_size"] = 32
    ds_config["elasticity"]["micro_batch_sizes"] = [1, 2, 3, 7]
    ds_config["elasticity"]["min_gpus"] = 1
    final_batch_size, valid_gpus, mbsize = deepspeed.elasticity.get_compatible_gpus(
        ds_config=ds_config,
        target_deepspeed_version=ds_version,
        world_size=7)
    assert mbsize == 3
