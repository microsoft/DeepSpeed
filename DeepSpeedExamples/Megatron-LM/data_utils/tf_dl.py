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
"""PyTorch DataLoader for TFRecords"""

import queue
import threading

import tensorflow as tf
tf.enable_eager_execution()
import torch
import numpy as np

class TFRecordDataLoader(object):
    def __init__(self, records, batch_size, max_seq_len, max_preds_per_seq, train, num_workers=2, seed=1, threaded_dl=False):
        assert max_preds_per_seq is not None, "--max-preds-per-seq MUST BE SPECIFIED when using tfrecords"
        tf.set_random_seed(seed)
        if isinstance(records, str):
            records  = [records]

        self.record_converter = Record2Example({"input_ids": tf.FixedLenFeature([max_seq_len], tf.int64),
                                                "input_mask": tf.FixedLenFeature([max_seq_len], tf.int64),
                                                "segment_ids": tf.FixedLenFeature([max_seq_len], tf.int64),
                                                "masked_lm_positions": tf.FixedLenFeature([max_preds_per_seq], tf.int64),
                                                "masked_lm_ids": tf.FixedLenFeature([max_preds_per_seq], tf.int64),
                                                "masked_lm_weights": tf.FixedLenFeature([max_preds_per_seq], tf.float32),
                                                "next_sentence_labels": tf.FixedLenFeature([1], tf.int64)})

        #Instantiate dataset according to original BERT implementation
        if train:
            self.dataset = tf.data.Dataset.from_tensor_slices(tf.constant(records))
            self.dataset = self.dataset.repeat()
            self.dataset = self.dataset.shuffle(buffer_size=len(records))

            # use sloppy tfrecord dataset
            self.dataset = self.dataset.apply(
                tf.contrib.data.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy=train,
                    cycle_length=min(num_workers, len(records))))
            self.dataset = self.dataset.shuffle(buffer_size=100)
        else:
            self.dataset = tf.data.TFRecordDataset(records)
            self.dataset = self.dataset.repeat()

        # Instantiate dataloader (do not drop remainder for eval)
        loader_args = {'batch_size': batch_size, 
                       'num_parallel_batches': num_workers,
                       'drop_remainder': train}
        self.dataloader = self.dataset.apply(tf.contrib.data.map_and_batch(self.record_converter, **loader_args))
        self.threaded_dl = threaded_dl
        self.num_workers = num_workers

    def __iter__(self):
        if self.threaded_dl:
            data_iter = iter(MultiprocessLoader(self.dataloader, self.num_workers))
            for item in data_iter:
                yield item
        else:
            data_iter = iter(self.dataloader)
            for item in data_iter:
                yield convert_tf_example_to_torch_tensors(item)

class Record2Example(object):
    def __init__(self, feature_map):
        self.feature_map = feature_map

    def __call__(self, record):
        """Decodes a BERT TF record to a TF example."""
        example = tf.parse_single_example(record, self.feature_map)
        for k, v in list(example.items()):
            if v.dtype == tf.int64:
                example[k] = tf.to_int32(v)
        return example

def convert_tf_example_to_torch_tensors(example):
    item = {k: (v.numpy()) for k,v in example.items()}
    mask = np.zeros_like(item['input_ids'])
    mask_labels = np.ones_like(item['input_ids'])*-1
    for b, row in enumerate(item['masked_lm_positions'].astype(int)):
        for i, idx in enumerate(row):
            if item['masked_lm_weights'][b, i] != 0:
                mask[b, idx] = 1
                mask_labels[b, idx] = item['masked_lm_ids'][b, i]
    output = {'text': item['input_ids'], 'types': item['segment_ids'],'is_random': item['next_sentence_labels'],
            'pad_mask': 1-item['input_mask'], 'mask': mask, 'mask_labels': mask_labels}
    return {k: torch.from_numpy(v) for k,v in output.items()}

class MultiprocessLoader(object):
    def __init__(self, dataloader, num_workers=2):
        self.dl = dataloader
        self.queue_size = 2*num_workers

    def __iter__(self):
        output_queue = queue.Queue(self.queue_size)
        output_thread = threading.Thread(target=_multiproc_iter,
                                         args=(self.dl, output_queue))
        output_thread.daemon = True
        output_thread.start()

        while output_thread.is_alive():
            yield output_queue.get(block=True)
        else:
            print(RuntimeError('TF record data loader thread exited unexpectedly'))

def _multiproc_iter(dl, output_queue):
    data_iter = iter(dl)
    for item in data_iter:
        tensors = convert_tf_example_to_torch_tensors(item)
        output_queue.put(tensors, block=True)