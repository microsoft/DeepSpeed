# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed import comm as dist
global num_kv_heads


def set_num_kv_heads(num):
    global num_kv_heads
    num_kv_heads = num


def get_num_kv_heads():
    global num_kv_heads
    return num_kv_heads


def get_shard_size(total_size, mp_size, rank=None):
    global num_kv_heads
    # When we have num_kv_heads defined, uneven division is possible, otherwise enforce near even division
    if rank == None:
        rank = dist.get_rank()
    if num_kv_heads != None and total_size % num_kv_heads == 0:
        my_slices = (num_kv_heads // mp_size) + (1 if rank < (num_kv_heads % mp_size) else 0)
        return total_size * my_slices // num_kv_heads
    else:
        return total_size // mp_size + (1 if rank < (total_size % mp_size) else 0)


def get_shard_size_list(total_size, mp_size):
    shard_sizes = []
    for i in range(mp_size):
        shard_sizes.append(get_shard_size(total_size, mp_size, i))
    return shard_sizes
