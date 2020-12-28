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
sys.path.append('..')

from data_utils.tokenization_gpt2 import GPT2Tokenizer


class Tokenizer:

    def __init__(self, cache_dir=None):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2',
                                                       cache_dir=cache_dir)
        self.tokenizer.max_len = int(1e12)
        self.eod_token = self.tokenizer.encoder['<|endoftext|>']
        assert self.eod_token < 65535, 'vocab size will not fit in uint16'
        print('> GPT2 tokenizer with {} vocab size and eod token {} ...'.format(
            len(self.tokenizer.encoder), self.eod_token))

    def tokenize_document(self, document):
        tokens = self.tokenizer.encode(document)
        tokens.append(self.eod_token)
        return tokens
