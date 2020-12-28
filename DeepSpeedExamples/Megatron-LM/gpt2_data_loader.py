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

import json
import os

import numpy as np
import torch
from torch.multiprocessing import Lock
from torch.utils.data import Dataset

import mpu
from data_utils.samplers import DistributedBatchSampler
from data_utils.tokenization_gpt2 import GPT2Tokenizer


def make_gpt2_dataloaders(args):

    # Input parameters.
    input_data_sizes_file = args.input_data_sizes_file
    seq_length = args.seq_length
    initial_seed = args.seed

    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    global_batch_size = args.batch_size * world_size
    num_workers = args.num_workers

    def make_data_loader_(data_path):
        # Build the dataset.
        dataset = GPT2Dataset(data_path, input_data_sizes_file,
                              seq_length, initial_seed)
        # Use a simple sampler with distributed batch sampler.
        sampler = torch.utils.data.SequentialSampler(dataset)
        batch_sampler = DistributedBatchSampler(sampler=sampler,
                                                batch_size=global_batch_size,
                                                drop_last=True,
                                                rank=rank,
                                                world_size=world_size)
        # Torch dataloader.
        return torch.utils.data.DataLoader(dataset,
                                           batch_sampler=batch_sampler,
                                           num_workers=num_workers,
                                           pin_memory=True)

    train = make_data_loader_(args.train_data_path)
    valid = make_data_loader_(args.val_data_path)
    test = make_data_loader_(args.test_data_path)

    args.do_train = False
    args.do_valid = False
    args.do_test = False

    if train is not None:
        args.do_train = True
    if valid is not None:
        args.do_valid = True
    if test is not None:
        args.do_test = True

    # Tokenizer.
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=args.cache_dir)
    eod_token = tokenizer.encoder['<|endoftext|>']
    num_tokens = eod_token + 1

    return (train, valid, test), num_tokens, eod_token


class GPT2Dataset(Dataset):

    def __init__(self, data_path, sizes_filename, seq_length,
                 initial_seed, max_epochs=100):
        # Input parameters.
        self.data_path = data_path
        self.sizes_filename = sizes_filename
        self.seq_length = seq_length
        self.initial_seed = initial_seed
        self.max_epochs = max_epochs
        # Lock for building the dataset.
        self.lock = Lock()

        # Shard stuff.
        # Dictionary from shard nameto its size (number of element).
        self.master_shard_size_dict = None
        # Dictionary from shard name to modified size so it is
        # divisible by self.seq_length.
        self.shard_size_dict = None
        # Long array (self.max_epochs * num-shards) populated
        # randomly with shard names.
        self.shards_name = None
        # Start index of the data for a shard.
        self.shards_start_index = None
        self.build_shard_mappings_()
        self.data_length = self.shards_start_index[-1]

        # Data.
        self.shards_data = [None]*self.shards_name.size
        self.shards_sample_index = [None]*self.shards_name.size

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        # Find which shard we need.
        shard_index = np.searchsorted(self.shards_start_index,
                                      idx, side='right') - 1
        # data index in the shard.
        data_idx = idx - self.shards_start_index[shard_index]
        # Load the shard if it is not in memory.
        #self.lock.acquire()
        if self.shards_data[shard_index] is None:
            print('global rank {} is building data for shard index {} ...'.
                  format(torch.distributed.get_rank(), shard_index))
            self.build_dataset_(shard_index)
        #assert self.shards_data[shard_index] is not None
        #self.lock.release()
        # Start index.
        start_index = self.shards_sample_index[shard_index][data_idx]
        # Add one for label shift.
        end_index = start_index + self.seq_length + 1
        data = self.shards_data[shard_index][start_index:end_index]
        return {'text': np.array(data, dtype=np.int64)}

    def build_dataset_(self, shard_index):
        # Garbage collect so we don't use a lot of memory.
        # Leave the last one in case other threads have not catche up yet.
        #for i in range(shard_index - 1):
        for i in range(shard_index):
            self.shards_data[i] = None
            self.shards_sample_index[i] = None
        # Read the shard.
        filename = os.path.join(self.data_path, self.shards_name[shard_index])
        print('loading {}'.format(filename))
        data = np.load(filename, allow_pickle=True)
        # Shuffle the data
        rng = np.random.RandomState(self.initial_seed + shard_index)
        rng.shuffle(data)
        # Flatten.
        data = np.hstack(data)
        size = (data.shape[0] - 1) // self.seq_length
        last_index = size * self.seq_length + 1
        data = data[0:last_index]
        self.shards_data[shard_index] = data
        indices = np.arange(size) * self.seq_length
        rng.shuffle(indices)
        self.shards_sample_index[shard_index] = indices

    def build_shard_mappings_(self):
        # Load the sizes file.
        sizes_filename = os.path.join(self.data_path, self.sizes_filename)
        if torch.distributed.get_rank() == 0:
            print(' > loading sizes from {}'.format(sizes_filename))
        with open(sizes_filename, 'r') as f:
            self.master_shard_size_dict = json.load(f)
        if torch.distributed.get_rank() == 0:
            print('   found {} shards'.format(len(self.master_shard_size_dict)))
        # Adjust sizes to be a multiple of seq_length.
        self.shard_size_dict = self.master_shard_size_dict.copy()
        total_samples = 0
        for shard in self.shard_size_dict:
            size = self.shard_size_dict[shard]
            size = ((size - 1) // self.seq_length) * self.seq_length
            total_samples += size // self.seq_length
            self.shard_size_dict[shard] = size
        if torch.distributed.get_rank() == 0:
            print('   found {} samples in the dataset'.format(total_samples))
        # Build a list of shards.
        shards_ = np.sort(np.array(list(self.shard_size_dict.keys())))
        rng = np.random.RandomState(self.initial_seed)
        self.shards_name = np.copy(shards_)
        rng.shuffle(self.shards_name)
        for i in range(1, self.max_epochs):
            shards_c = np.copy(shards_)
            rng.shuffle(shards_c)
            self.shards_name = np.append(self.shards_name, shards_c)
        # Build the global indexing.
        self.shards_start_index = np.zeros(self.shards_name.size, dtype=np.int)
        self.shards_start_index[0] = 0
        for i in range(1, self.shards_name.size):
            shard = str(self.shards_name[i-1])
            size = self.shard_size_dict[shard]
            self.shards_start_index[i] = self.shards_start_index[i-1] + \
                                         size // self.seq_length

'''
if __name__ == '__main__':

    print('gpt2 data loader ...')
    path = '/raid/mshoeybi/data/gpt2/adlr/reddit_all_ftfy_lg200/npys'

    dataset = GPT2Dataset(path, 'sizes.txt', 1024, 1234, 100)
    print('dataset contains {} samples'.format(dataset.data_length))

    for i in range(len(dataset)):
        if i % 512000 == 0:
            print(i)
        data = dataset[i]
'''
