# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.runtime import utils as ds_utils


def test_partition_balanced():
    """
    params: weights = [1, 1, 1, 1], num_parts = 4
    result: [0, 1, 2, 3, 4]
    """
    result = ds_utils.partition_balanced(weights=[1, 1, 1, 1], num_parts=4)
    assert result == [0, 1, 2, 3, 4]
    """
    params: weights = [1, 1, 1, 1, 1], num_parts = 4
    result: [0, 1, 2, 3, 5] or [0, 2, 3, 4, 5]
    """
    result = ds_utils.partition_balanced(weights=[1, 1, 1, 1, 1], num_parts=4)
    assert result in [[0, 1, 2, 3, 5], [0, 2, 3, 4, 5]]
    """
    params: weights = [1, 1, 1, 1, 0, 1], num_parts = 4
    result: [0, 1, 2, 3, 6] or [0, 2, 3, 4, 6]
    """
    result = ds_utils.partition_balanced(weights=[1, 1, 1, 1, 0, 1], num_parts=4)
    assert result in [[0, 1, 2, 3, 6], [0, 2, 3, 4, 6]]
