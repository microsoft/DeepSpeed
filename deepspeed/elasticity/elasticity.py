"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
import numpy as np

from .config import ElasticityConfig, ElasticityConfigError, ElasticityError, \
    ElasticityIncompatibleWorldSize
from .constants import ELASTICITY, ENABLED, ENABLED_DEFAULT, LATEST_ELASTICITY_VERSION, \
    MINIMUM_DEEPSPEED_VERSION
from ..git_version_info import version as __version__
'''Thirty eight smallest highly composite numbers.
The list should be enough to support up to 720K batch
size'''
HCN_LIST = [
    1,
    2,
    4,
    6,
    12,
    24,
    36,
    48,
    60,
    120,
    180,
    240,
    360,
    720,
    840,
    1260,
    1680,
    2520,
    5040,
    7560,
    10080,
    15120,
    20160,
    25200,
    27720,
    45360,
    50400,
    55440,
    83160,
    110880,
    166320,
    221760,
    277200,
    332640,
    498960,
    554400,
    665280,
    720720
]


def get_candidate_batch_sizes(base_list, max_acceptable_batch_size):
    candidate_batch_size = []

    #brute force is fine here. We are working with very small lists
    for base in base_list:
        batch_size = base
        for hcn in HCN_LIST:
            new_batch_size = base * hcn
            if new_batch_size > max_acceptable_batch_size:
                break
            batch_size = new_batch_size
        candidate_batch_size.append(batch_size)
    return list(set(candidate_batch_size))


def get_valid_gpus(batch_size, micro_batches, min_valid_gpus, max_valid_gpus):
    valid_gpus = []
    for micro_batch in micro_batches:
        if batch_size % micro_batch == 0:

            max_gpus = batch_size // micro_batch
            if max_gpus >= min_valid_gpus and max_gpus <= max_valid_gpus:
                valid_gpus.append(max_gpus)

            for i in range(1, max_gpus // 2 + 1):
                if max_gpus % i == 0:
                    if i >= min_valid_gpus and i <= max_valid_gpus:
                        valid_gpus.append(i)
    valid_gpus = set(valid_gpus)
    valid_gpus = sorted(list(valid_gpus))

    #print(f"Get valid gpus batch size: {batch_size}, micro_batches: {micro_batches} valid_gpus: {valid_gpus}")

    return valid_gpus


def get_best_candidates(candidate_batch_sizes,
                        micro_batches,
                        min_gpus,
                        max_gpus,
                        prefer_larger):

    max_valid_gpus = 0
    valid_gpus = None
    final_batch_size = int(min(micro_batches))

    for batch_size in candidate_batch_sizes:

        current_valid_gpus = get_valid_gpus(batch_size,
                                            micro_batches,
                                            min_gpus,
                                            max_gpus)

        if (len(current_valid_gpus) > max_valid_gpus
                or (len(current_valid_gpus) == max_valid_gpus and
                    ((prefer_larger and batch_size > final_batch_size) or
                     (not prefer_larger and batch_size < final_batch_size)))):
            max_valid_gpus = len(current_valid_gpus)
            valid_gpus = current_valid_gpus
            final_batch_size = batch_size

    return final_batch_size, valid_gpus


def _get_compatible_gpus_v01(micro_batches,
                             max_acceptable_batch_size,
                             min_gpus=None,
                             max_gpus=None,
                             prefer_larger=True):
    '''We use two heuristics to compute the batch size
        1. We use the Lowest Common Multiple of the micro-batches
    as the base batch size and scale it by a HCN such that the result is
    the largest batch size less than the max_acceptable batch size
        2. We use each of the micro batches as a base and scale it
    by a HCN such that the result is the largest batch size less than the
    max_acceptable batch size.

    We then use brute force to count the number of compatible GPU count for
    each of the aforementioned cases, and return the batch size with the most number of
    compatible GPU counts in the min-max GPU range if provided, other wise
    we return the batch size with the most number of total compatible GPU counts.

    Returns:
        final_batch_size
        valid_gpus
    '''

    if min_gpus is None:
        min_gpus = int(1)

    if max_gpus is None:
        max_gpus = int(max_acceptable_batch_size / min(micro_batches))

    assert all(mb <= max_acceptable_batch_size for mb in micro_batches ), \
            f"All micro batches must be less than \
            or equal to max_acceptable_batch_size: {max_acceptable_batch_size}"

    lcm = np.lcm.reduce(micro_batches)

    base_list = []
    base_list.extend(micro_batches)
    base_list.append(lcm)

    candidate_batch_sizes = get_candidate_batch_sizes(base_list,
                                                      max_acceptable_batch_size)

    final_batch_size, valid_gpus = get_best_candidates(
        candidate_batch_sizes,
        micro_batches,
        min_gpus,
        max_gpus,
        prefer_larger)

    return final_batch_size, valid_gpus


def _compatible_ds_version_check(target_deepspeed_version: str):
    min_major, min_minor, min_patch = MINIMUM_DEEPSPEED_VERSION.split(".")
    target_version_list = list(map(int, target_deepspeed_version.split(".")))
    assert len(target_version_list) >= 2, "Unable to parse version number, expecting" \
        f"major.minor[.patch] format but received {target_deepspeed_version}"

    if len(target_deepspeed_version) == 2:
        trg_major, trg_minor = target_version_list
        trg_patch = 0
    else:
        trg_major, trg_minor, trg_patch = target_version_list

    err_str = f"Target deepspeed version of {target_deepspeed_version} is not compatible " \
        f"with minimum version {MINIMUM_DEEPSPEED_VERSION} supporting elasticity."
    if trg_major < min_major:
        raise ElasticityError(err_str)
    if trg_minor < min_minor:
        raise ElasticityError(err_str)
    if trg_patch < min_patch:
        raise ElasticityError(err_str)


def elasticity_enabled(ds_config: dict):
    if ELASTICITY not in ds_config:
        return False
    return ds_config[ELASTICITY].get(ENABLED, ENABLED_DEFAULT)


def get_compatible_gpus(ds_config: dict, target_deepspeed_version: str, world_size=0):
    """[summary]

    Args:
        ds_config (dict): DeepSpeed config dictionary/json
        target_deepspeed_version (str): When called from scheduling
            infrastructure we want to ensure that the target deepspeed version is
            compatible with the elasticity version used in the backend.
        world_size (int, optional): Intended/current world size, will do some sanity
            checks to ensure world size is actually valid with the config.

    Raises:
        ElasticityConfigError: Missing required elasticity config or elasticity disabled
        ElasticityError: If target deepspeed version is not compatible with current version

    Returns:
        final_batch_size (int): total batch size used for training
        valid_gpus (list(int)): list of valid GPU counts with this config
        micro_batch_size (int, optional): if world_size is provided will return
            specific micro batch size
    """
    if ELASTICITY not in ds_config:
        raise ElasticityConfigError(f"'{ELASTICITY}' is missing from config json," \
            " please add it if running an elastic training job.")

    elastic_config_dict = ds_config[ELASTICITY]
    if not elastic_config_dict.get(ENABLED, ENABLED_DEFAULT):
        raise ElasticityConfigError("Elasticity is disabled, please enable it " \
            "('enabled':true) if running an elastic training job.")

    elastic_config = ElasticityConfig(elastic_config_dict)

    if elastic_config.version > LATEST_ELASTICITY_VERSION:
        raise ElasticityConfigError("Attempting to run elasticity version " \
            f"{elastic_config.version} but runtime only supports up " \
            f"to {LATEST_ELASTICITY_VERSION}")

    # Ensure target deepspeed version works with intended elasticity version
    if not _compatible_ds_version_check(target_deepspeed_version):
        raise ElasticityError("Unable to run elasticity on target deepspeed version of" \
            f" {target_deepspeed_version}, currently {__version__}")

    if elastic_config.version == 0.1:
        final_batch_size, valid_gpus = _get_compatible_gpus_v01(
            micro_batches=elastic_config.micro_batches,
            max_acceptable_batch_size=elastic_config.max_acceptable_batch_size,
            min_gpus=elastic_config.min_gpus,
            max_gpus=elastic_config.max_gpus,
            prefer_larger=elastic_config.prefer_larger_batch_size)
    else:
        raise NotImplementedError(
            f"Unable to find elastic logic for version: {elastic_config.version}")

    if world_size > 0:
        if world_size not in valid_gpus:
            raise ElasticityIncompatibleWorldSize("World size is not valid " \
        f"with the current list of valid GPU counts: {valid_gpus}")

        # pick largest valid micro batch size
        micro_batch_size = None
        for mbsz in reversed(elastic_config.micro_batches):
            if final_batch_size % world_size % mbsz == 0:
                micro_batch_size = mbsz
                break
        assert micro_batch_size is not None, "Unable to find divisible micro batch size" \
            f" world_size={world_size}, final_batch_size={final_batch_size}, and " \
            f" micro_batches={elastic_config.micro_batches}."
        return final_batch_size, valid_gpus, micro_batch_size

    return final_batch_size, valid_gpus


def small_test_config():
    ds_config = {
        "elasticity": {
            "enabled": True,
            "max_train_batch_size": 2000,
            "micro_batch_sizes": [2,
                                  4,
                                  6],
            "min_gpus": 1,
            "max_gpus": 10000,
            "min_time": 20,
            "version": 0.1
        }
    }

    final_batch_size, valid_gpus = get_compatible_gpus(ds_config)


def small_test():
    micro_batches = [8, 12, 16, 17]
    max_acceptable_batch_size = 10000
    min_gpus = 32
    max_gpus = 1500

    micro_batches = sorted(list(set(micro_batches)), reverse=True)


    final_batch, compatible_gpu_counts = _get_compatible_gpus_v01(micro_batches,
                                max_acceptable_batch_size,
                                min_gpus = min_gpus,
                                max_gpus = max_gpus)

    print(
        f"Final Batch: {final_batch}, Micro batches {micro_batches}, compatible gpus {compatible_gpu_counts} total gpus {len(compatible_gpu_counts)}"
    )

    for gpu_num in compatible_gpu_counts:
        assert final_batch % gpu_num == 0, f"Batch {final_batch} is not divisible by GPU count {gpu_num}"
        batch_per_gpu = final_batch // gpu_num
        found_valid_mb = False

        for mb in micro_batches:
            if batch_per_gpu % mb == 0:
                found_valid_mb = True
                print(
                    f"GPU count : {gpu_num} Micro_batch: {mb} GAS = {batch_per_gpu/mb}")
                break
        assert found_valid_mb, "No valid mb found"


if __name__ == "__main__":
    small_test()
