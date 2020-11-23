import numpy as np


class DeepSpeedElasticityConfig(object):
    def __init__(self, param_dict, world_size):
        super(DeepSpeedElasticityConfig, self).__init__()

        self.world_size = world_size

        self.micro_batches = []
        self.max_acceptable_batch_size = None
        self.min_gpus = None
        self.max_gpus = None
        self.prefer_larger_batch_size = True

        #batch size computed based on elasticity config
        self.computed_batch_size = None

        #micro batch size for this run computed based on
        #elasticity configs, and the world size
        self.computed_micro_batch = None

        #gradient accumulation steps for this run computed based on
        #elasticity configs, and the world size
        self.computed_gradient_accumulation_step = None

        self._initialize(param_dict)

    def _initialize(self, param_dict):
        pass


'''Thirty eight smallest highly composite numbers.
The list should be enough to support up to 720K batch
size'''

hcn_list = [
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
    global hcn_list

    candidate_batch_size = []

    #brute force is fine here. We are working with very small lists
    for base in base_list:
        batch_size = base
        for hcn in hcn_list:
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


def get_compatible_gpus(micro_batches,
                        max_acceptable_batch_size,
                        min_gpus=None,
                        max_gpus=None,
                        prefer_larger=True,
                        model_parallel=1,
                        pipeline_parallel=1):
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
    assert model_parallel == 1, f"Does not currently support model parallel \
                                not equal to 1, but was provided \
                                {model_parallel}"

    assert pipeline_parallel == 1, f"Does not currently support model parallel \
                                not equal to 1, but was provided \
                                {pipeline_parallel}"

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

    final_batch_size, valid_gpus = get_best_candidates(candidate_batch_sizes, micro_batches,
                                                        min_gpus, max_gpus, prefer_larger)

    return final_batch_size, valid_gpus


#def get_compatible_gpus(ds_config_file):


def small_test():
    micro_batches = [8, 12, 16, 17]
    max_acceptable_batch_size = 10000
    min_gpus = 32
    max_gpus = 1500

    micro_batches = sorted(list(set(micro_batches)), reverse=True)


    final_batch, compatible_gpu_counts = get_compatible_gpus(micro_batches,
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


#small_test()
