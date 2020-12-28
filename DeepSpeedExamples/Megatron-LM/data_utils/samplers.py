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
"""batch samplers that work with either random or sequential data samplers"""
import math
import os
import sys

import torch
from torch.utils import data
import numpy as np

class RandomSampler(data.sampler.Sampler):
    r"""
    Based off of pytorch RandomSampler and DistributedSampler. Essentially a RandomSampler,
    but this class lets the user set an epoch like DistributedSampler
    Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False
    """

    def __init__(self, data_source, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.epoch = -1

        if self._num_samples is not None and replacement is False:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))
        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        g = torch.Generator()
        if self.epoch >= 0:
            g.manual_seed(self.epoch)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64, generator=g).tolist())
        return iter(torch.randperm(n, generator=g).tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class DistributedBatchSampler(data.sampler.BatchSampler):
    """
    similar to normal implementation of distributed sampler, except implementation is at the
    batch sampler level, instead of just the sampler level. This allows wrapping of arbitrary
    data samplers (sequential, random, WeightedRandomSampler, etc.) with this batch sampler.
    """
    def __init__(self, sampler, batch_size, drop_last, rank=-1, world_size=2, wrap_last=False):
        super(DistributedBatchSampler, self).__init__(sampler, batch_size, drop_last)
        if rank == -1:
            assert False, 'should not be here'
            rank = torch.distributed.get_rank()
        self.rank = rank
        self.world_size = world_size
        self.sampler.wrap_around = 0
        self.wrap_around = 0
        self.wrap_last = wrap_last
        self.start_iter = 0

    def __iter__(self):
        batch = []
        last_batch = None
        i = 0
        for idx in self.data_iterator(self.sampler, wrap_around=False):
            batch.append(idx)
            if len(batch) == self.batch_size:
                tbatch = self._batch(batch)
                if i >= self.start_iter:
                    yield tbatch
                    self.start_iter = 0
                i += 1
                last_batch = np.array(list(tbatch))
                batch = []
        batch_len = len(batch)
        if batch_len > 0 and not self.drop_last:
            if self.wrap_last:
                self.sampler.wrap_around -= (self.batch_size)
                self.wrap_around += (len(batch))
                self.wrap_around %= self.batch_size
                if isinstance(self.sampler, TransposedSampler):
                    for i, idx in enumerate(self.data_iterator(self.sampler, wrap_around=True)):
                        if i == 0:
                            continue
                        batch.append(idx)
                        new_batch_len = len(batch)
                        if len(batch) == self.batch_size:
                            break
            yield self._batch(batch)
        if self.wrap_last:
            self.sampler.wrap_around += self.batch_size

    def data_iterator(self, _iter, wrap_around=False):
        """iterates through data and handles wrap around"""
        for i, idx in enumerate(_iter):
            if i < self.wrap_around%self.batch_size:
                continue
            if wrap_around:
                self.wrap_around += 1
                self.wrap_around %= self.batch_size
            yield idx

    def _batch(self, batch):
        """extracts samples only pertaining to this worker's batch"""
        start = self.rank*self.batch_size//self.world_size
        end = (self.rank+1)*self.batch_size//self.world_size
        return batch[start:end]
