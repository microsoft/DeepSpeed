# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
sys.path.append("../..")

import torch
import mpu

from commons import initialize_distributed
from commons import print_separator


def test_initialize_model_parallel(model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing initialize_model_parallel with size {} ...'.format(
            model_parallel_size))
    model_parallel_size_ = min(model_parallel_size,
                               torch.distributed.get_world_size())
    assert not mpu.model_parallel_is_initialized()
    mpu.initialize_model_parallel(model_parallel_size_)
    assert mpu.model_parallel_is_initialized()

    # Checks.
    def check(group, world_size, rank):
        assert world_size == torch.distributed.get_world_size(group=group)
        assert rank == torch.distributed.get_rank(group=group)

    # Model parallel.
    world_size = model_parallel_size_
    rank = torch.distributed.get_rank() % model_parallel_size_
    assert world_size == mpu.get_model_parallel_world_size()
    assert rank == mpu.get_model_parallel_rank()
    check(mpu.get_model_parallel_group(), world_size, rank)


    # Data parallel.
    world_size = torch.distributed.get_world_size() // model_parallel_size_
    rank = torch.distributed.get_rank() // model_parallel_size
    assert world_size == mpu.get_data_parallel_world_size()
    assert rank == mpu.get_data_parallel_rank()
    check(mpu.get_data_parallel_group(), world_size, rank)

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('>> passed the test :-)')


def test_get_model_parallel_src_rank(model_parallel_size_):

    if torch.distributed.get_rank() == 0:
        print('> testing get_model_parallel_src_rank with size {} ...'.format(
            model_parallel_size_))
    model_parallel_size = min(model_parallel_size_,
                              torch.distributed.get_world_size())
    assert not mpu.model_parallel_is_initialized()
    mpu.initialize_model_parallel(model_parallel_size)
    assert mpu.model_parallel_is_initialized()

    # Checks
    src_rank = torch.distributed.get_rank() - mpu.get_model_parallel_rank()
    assert mpu.get_model_parallel_src_rank() == src_rank

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('>> passed the test :-)')


if __name__ == '__main__':

    initialize_distributed()
    world_size = torch.distributed.get_world_size()
    model_parallel_size = 1
    while model_parallel_size <= world_size:
        print_separator('test initialize model parallel')
        test_initialize_model_parallel(model_parallel_size)
        print_separator('test model parallel source rank')
        test_get_model_parallel_src_rank(model_parallel_size)
        model_parallel_size *= 2
