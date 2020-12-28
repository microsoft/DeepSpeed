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

import functools
import operator
import sys
sys.path.append("../..")

import torch
import mpu
from mpu import data as data_utils

from commons import initialize_distributed
from commons import print_separator


def test_boradcast_data(model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing boradcast_data with model parallel size {} ...'.
              format(model_parallel_size))

    mpu.initialize_model_parallel(model_parallel_size)
    torch.manual_seed(1234 + mpu.get_data_parallel_rank())
    model_parallel_size = mpu.get_model_parallel_world_size()

    key_size_t = {'key1': [7, 11],
                  'key2': [8, 2, 1],
                  'key3': [13],
                  'key4': [5, 1, 2],
                  'key5': [5, 12]}
    keys = list(key_size_t.keys())

    data = {}
    data_t = {}
    for key in key_size_t:
        data[key] = torch.LongTensor(size=key_size_t[key]).random_(0, 1000)
        data_t[key] = data[key].clone()
    data['keyX'] = torch.FloatTensor(size=(5, )).random_(0, 1000)
    data_t['keyX'] = data['keyX'].clone()
    if mpu.get_model_parallel_rank() != 0:
        data = None

    data_utils._check_data_types(keys, data_t, torch.int64)
    key_size, key_numel, \
        total_numel = data_utils._build_key_size_numel_dictionaries(keys, data)
    for key in keys:
        assert key_size[key] == key_size_t[key]
    total_numel_t = 0
    for key in keys:
        target_size = functools.reduce(operator.mul, key_size_t[key], 1)
        assert key_numel[key] == target_size
        total_numel_t += target_size
    assert total_numel == total_numel_t

    data_b = data_utils.broadcast_data(keys, data, torch.int64)
    for key in keys:
        tensor = data_t[key].cuda()
        assert data_b[key].sub(tensor).abs().max() == 0

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
        print_separator('test test boradcast data')
        test_boradcast_data(model_parallel_size)
        model_parallel_size *= 2


