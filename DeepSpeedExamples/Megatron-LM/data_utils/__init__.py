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
"""utils for creating datasets"""
import os
import math

from .samplers import DistributedBatchSampler
from .datasets import json_dataset, csv_dataset, split_ds, ConcatDataset, SplitDataset, bert_sentencepair_dataset, GPT2Dataset
from .lazy_loader import exists_lazy, make_lazy, lazy_array_loader
from .tokenization import Tokenization, CommandToken, Tokenizer, CharacterLevelTokenizer, BertWordPieceTokenizer, GPT2BPETokenizer, make_tokenizer
from . import corpora

TRAIN_DATA = 0
VAL_DATA = 1
TEST_DATA = 2

def should_split(split):
    """
    given split proportions checks if should split
    Examples:
    >>> should_split([10,0,0]) 
    False
    >>> should_split([1,.1,.2])
    True
    """
    return max(split)/sum(split) != 1.

def get_ext(path):
    """gets path extension"""
    return os.path.splitext(path)[1]

def get_dataset(path, **kwargs):
    """gets dataset object based on keyword args and file at `path`"""
    if supported_corpus(path):
        return corpora.NAMED_CORPORA[path](**kwargs)
    ext = get_ext(path)
    if '.json' in ext:
        text = json_dataset(path, **kwargs)
    elif ext in ['.csv', '.tsv']:
        text = csv_dataset(path, **kwargs)
    else:
        raise NotImplementedError('data file type %s is not supported'%(ext))
    return text

def supported_corpus(corpus_name):
    """checks if corpus name is defined in `corpora.py`"""
    return corpus_name in corpora.NAMED_CORPORA

def make_dataset(path, seq_length, text_key, label_key, lazy=False, process_fn=None, split=[1.],
                delim=',', loose=False, binarize_sent=False, drop_unlabeled=False, tokenizer=None,
                tokenizer_type='CharacterLevelTokenizer', tokenizer_model_path=None, vocab_size=None,
                model_type='bpe', pad_token=0, character_converage=1.0, non_binary_cols=None, **kwargs):
    """function to create datasets+tokenizers for common options"""
    if isinstance(process_fn, str):
        process_fn = eval(process_fn)
    if non_binary_cols is not None:
        # multilabel dataset support (only for csvs)
        label_key = non_binary_cols
    def get_dataset_from_path(path_):
        if lazy:
            # get lazily loaded dataset
            named_corpora = False
            if supported_corpus(path_):
                named_corpora = True
                name = path_
                path_ = corpora.NAMED_CORPORA[path_].PATH
            if not exists_lazy(path_, data_type='data'):
                # create cached version of dataset for lazy loading if it doesn't exist
                text = get_dataset(name if named_corpora else path_, text_key=text_key, label_key=label_key, binarize_sent=binarize_sent,
                    delim=delim, drop_unlabeled=drop_unlabeled, loose_json=loose)
                make_lazy(path_, text.X, data_type='data')
            text = lazy_array_loader(path_, data_type='data', map_fn=process_fn)
        else:
            # get dataset
            text = get_dataset(path_, text_key=text_key, label_key=label_key, binarize_sent=binarize_sent,
                    delim=delim, drop_unlabeled=drop_unlabeled, loose_json=loose, preprocess_fn=process_fn)
        return text
    # get one or multiple datasets and concatenate
    if isinstance(path, str):
        path = [path]
    datasets = [get_dataset_from_path(p) for p in path]
    if len(datasets) == 1:
        ds = datasets[0]
    else:
        ds = ConcatDataset(datasets)
    # make tokenizer for dataset
    if tokenizer is None:
        tokenizer = make_tokenizer(tokenizer_type, ds, tokenizer_model_path, vocab_size, model_type, 
                                    pad_token, character_converage, **kwargs)

    ds_type = ''
    if 'ds_type' in kwargs:
        ds_type = kwargs['ds_type']
    ds.SetTokenizer(tokenizer)
    # Split dataset into train/val/test (and wrap bert dataset)
    if should_split(split):
        ds = split_ds(ds, split)
        if ds_type.lower() == 'bert':
            presplit_sentences = kwargs['presplit_sentences'] if 'presplit_sentences' in kwargs else False
            ds = [bert_sentencepair_dataset(d, max_seq_len=seq_length, presplit_sentences=presplit_sentences)  if d is not None else None  for d in ds]
        elif ds_type.lower() == 'gpt2':
            ds = [GPT2Dataset(d, max_seq_len=seq_length) if d is not None else None for d in ds]
    else:
        if ds_type.lower() == 'bert':
            presplit_sentences = kwargs['presplit_sentences'] if 'presplit_sentences' in kwargs else False
            ds = bert_sentencepair_dataset(ds, max_seq_len=seq_length, presplit_sentences=presplit_sentences)
        elif ds_type.lower() == 'gpt2':
            ds = GPT2Dataset(ds, max_seq_len=seq_length)
    return ds, tokenizer
