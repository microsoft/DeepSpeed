# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

from pydantic import ValidationError

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.ragged import DSStateManagerConfig
from ...v2.inference_test_utils import skip_on_inference_v2

pytestmark = pytest.mark.skipif(skip_on_inference_v2(),
                                reason=f'Inference V2 not supported by {get_accelerator().device_name()}.')


@pytest.mark.inference_v2
def test_negative_max_tracked_sequences() -> None:
    with pytest.raises(ValidationError):
        DSStateManagerConfig(max_tracked_sequences=-1)


@pytest.mark.inference_v2
def test_zero_max_tracked_sequences() -> None:
    with pytest.raises(ValidationError):
        DSStateManagerConfig(max_tracked_sequences=0)


@pytest.mark.inference_v2
def test_negative_max_ragged_batch_size() -> None:
    with pytest.raises(ValidationError):
        DSStateManagerConfig(max_ragged_batch_size=-1)


@pytest.mark.inference_v2
def test_zero_max_ragged_batch_size() -> None:
    with pytest.raises(ValidationError):
        DSStateManagerConfig(max_ragged_batch_size=0)


@pytest.mark.inference_v2
def test_negative_max_ragged_sequence_count() -> None:
    with pytest.raises(ValidationError):
        DSStateManagerConfig(max_ragged_sequence_count=-1)


@pytest.mark.inference_v2
def test_zero_max_ragged_sequence_count() -> None:
    with pytest.raises(ValidationError):
        DSStateManagerConfig(max_ragged_sequence_count=0)


@pytest.mark.inference_v2
def test_too_small_max_ragged_batch_size() -> None:
    with pytest.raises(ValidationError):
        DSStateManagerConfig(max_ragged_batch_size=512, max_ragged_sequence_count=1024)


@pytest.mark.inference_v2
def test_too_small_max_tracked_sequences() -> None:
    with pytest.raises(ValidationError):
        DSStateManagerConfig(max_tracked_sequences=512, max_ragged_sequence_count=1024)
