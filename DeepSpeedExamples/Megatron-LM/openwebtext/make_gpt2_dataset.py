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
import numpy as np
import time
import os
import sys

from tokenizer import Tokenizer


def tokenize_corpus(filename, np_filename, print_interval=10000):

    print(' > tokenizing {}'.format(filename))

    tokenizer = Tokenizer(cache_dir='./cache')

    tokenized_docs = []
    num_docs = 0
    num_tokens = 0
    start_time = time.time()
    with open(filename, 'r') as f:
        for line in f:
            try:
                myjson = json.loads(line)
                url = myjson['url']
                sample = myjson['text']
                tokens = tokenizer.tokenize_document(sample)
                tokenized_docs.append(np.array(tokens, dtype=np.uint16))
                num_docs += 1
                num_tokens += len(tokens)
                if num_docs % print_interval == 0:
                    print('    processed {:9d} documents in {:.2f} (s) so far'.
                          format(num_docs, time.time() - start_time),
                          flush=True)
            except Exception as e:
                print('    skipping ', line, e)

    print('  >> processed {} document with total of {} tokens ...'.format(
        num_docs, num_tokens))

    tokenized_docs = np.array(tokenized_docs, dtype=object)
    np.save(np_filename, tokenized_docs, allow_pickle=True)
    print('  >> saved the tokenzed document to {} ...'.format(np_filename))


if __name__ == '__main__':

    print('building gpt2 dataset ...')

    path = sys.argv[1]
    shard = sys.argv[2]

    input_filename = os.path.join(path,
                                  'shards/shard_{:04d}'.format(int(shard)))
    output_filename = os.path.join(path,
                                  'npys/shard_{:04d}.npy'.format(int(shard)))
    print('will be reading {}'.format(input_filename))
    print('and will write the results to {}'.format(output_filename))

    tokenize_corpus(input_filename, output_filename)


