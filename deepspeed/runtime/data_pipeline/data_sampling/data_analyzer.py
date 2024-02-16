# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
from collections import defaultdict
import csv
import time
from multiprocessing import Process, Manager
import numpy as np
import torch
from torch.utils.data import BatchSampler, SequentialSampler, DataLoader, Subset

from deepspeed.utils import logger
import deepspeed.comm as dist
from .indexed_dataset import MMapIndexedDataset, valid_dtypes
from .utils import split_dataset, split_index, create_mmap_dataset_builder, close_mmap_dataset_builder, find_fit_int_dtype


class DataAnalyzer(object):

    def __init__(self,
                 dataset,
                 num_workers=1,
                 worker_id=0,
                 num_threads=1,
                 num_threads_reduce=1,
                 specific_threads=[],
                 batch_size=1,
                 metric_names=[],
                 metric_functions=[],
                 metric_types=[],
                 metric_dtypes=[],
                 save_path="./",
                 collate_fn=None,
                 custom_map_init=None,
                 custom_map_update=None,
                 custom_map_finalize=None,
                 custom_reduce=None,
                 sample_indices=None):
        super().__init__()
        self.dataset = dataset
        self.num_workers = num_workers
        self.worker_id = worker_id
        self.num_threads = num_threads
        self.num_threads_reduce = num_threads_reduce
        self.specific_threads = specific_threads
        self.batch_size = batch_size
        self.metric_names = metric_names
        self.metric_functions = metric_functions
        self.metric_types = metric_types
        self.metric_dtypes = metric_dtypes
        self.save_path = save_path
        self.collate_fn = collate_fn
        self.custom_map_init = custom_map_init
        self.custom_map_update = custom_map_update
        self.custom_map_finalize = custom_map_finalize
        self.custom_reduce = custom_reduce
        self.sample_indices = sample_indices

    def init_metric_results(self, thread_id, metric_names, metric_types, metric_dtypes, save_path, worker_id):
        metric_results = []
        for m_idx in range(len(metric_names)):
            metric_name, metric_type, metric_dtype = metric_names[m_idx], \
                metric_types[m_idx], metric_dtypes[m_idx]
            assert metric_dtype in valid_dtypes, f"metric_dtype {metric_dtype} not supported. Supported dtypes {valid_dtypes}"
            metric_save_path = f"{save_path}/{metric_name}/worker{worker_id}_thread{thread_id}/"
            os.makedirs(metric_save_path, exist_ok=True)
            if metric_type == 'single_value_per_sample':
                sample_to_metric_fname = f"{metric_save_path}/{metric_name}_sample_to_metric"
                sample_to_metric_builder = create_mmap_dataset_builder(sample_to_metric_fname, metric_dtype)
                metric_to_sample_fname = f"{metric_save_path}/{metric_name}_metric_to_sample"
                os.system(f"rm -rf {metric_to_sample_fname}*")
                metric_to_sample_dict = defaultdict(list)
                metric_results.append({
                    "sample_to_metric_fname": sample_to_metric_fname,
                    "sample_to_metric_builder": sample_to_metric_builder,
                    "metric_to_sample_fname": metric_to_sample_fname,
                    "metric_to_sample_dict": metric_to_sample_dict
                })
            elif metric_type == 'accumulate_value_over_samples':
                metric_value = None
                metric_value_fname = f"{metric_save_path}/{metric_name}_metric_value"
                metric_results.append({"metric_value": metric_value, "metric_value_fname": metric_value_fname})
        return metric_results

    def update_metric_results(self,
                              data,
                              metric_types,
                              metric_dtypes,
                              metric_functions,
                              metric_results,
                              batch_start_idx=0):
        for m_idx in range(len(metric_types)):
            metric_type, metric_dtype, metric_function, metric_result = metric_types[m_idx], \
                metric_dtypes[m_idx], metric_functions[m_idx], metric_results[m_idx]
            metric_values = metric_function(data)

            assert torch.is_tensor(metric_values) or isinstance(metric_values, np.ndarray), \
                    "metric_function must return a tensor or array"
            assert metric_values.dtype == metric_dtype, \
                    f"metric_function result dtype {metric_values.dtype} does not match metric_dtype {metric_dtype}"
            if isinstance(metric_values, np.ndarray):
                metric_values = torch.from_numpy(metric_values)

            if metric_type == 'single_value_per_sample':
                for row in range(metric_values.size()[0]):
                    sample_idx = batch_start_idx + row  # sample idx following dataset iteration order
                    if 'index' in data:  # Megatron use case, sample idx provided in 'index' field
                        sample_idx = data['index'][row][0].item()
                    elif self.sample_indices is not None:  # user defined shuffling of indices
                        sample_idx = self.sample_indices[sample_idx]
                    metric_result["sample_to_metric_builder"].add_item(metric_values[row].reshape(-1))
                    metric_result["metric_to_sample_dict"][metric_values[row].item()].append(sample_idx)
                for m_value in metric_result["metric_to_sample_dict"]:
                    if len(metric_result["metric_to_sample_dict"][m_value]) > 100:
                        metric_fname = metric_result["metric_to_sample_fname"]
                        with open(f"{metric_fname}_{m_value}.csv", 'a') as f:
                            writer = csv.writer(f)
                            writer.writerows([metric_result["metric_to_sample_dict"][m_value]])
                        metric_result["metric_to_sample_dict"][m_value] = []
            elif metric_type == 'accumulate_value_over_samples':
                if metric_result["metric_value"] is None:
                    metric_result["metric_value"] = metric_values
                else:
                    metric_result["metric_value"].add_(metric_values)

    def finalize_metric_results(self, metric_types, metric_dtypes, metric_results):
        for m_idx in range(len(metric_types)):
            metric_type, metric_dtype, metric_result = metric_types[m_idx], \
                metric_dtypes[m_idx], metric_results[m_idx]
            if metric_type == 'single_value_per_sample':
                metric_fname = metric_result["sample_to_metric_fname"]
                close_mmap_dataset_builder(metric_result["sample_to_metric_builder"], metric_fname)
                for m_value in metric_result["metric_to_sample_dict"]:
                    if len(metric_result["metric_to_sample_dict"][m_value]) > 0:
                        metric_fname = metric_result["metric_to_sample_fname"]
                        with open(f"{metric_fname}_{m_value}.csv", 'a') as f:
                            writer = csv.writer(f)
                            writer.writerows([metric_result["metric_to_sample_dict"][m_value]])
                        metric_result["metric_to_sample_dict"][m_value] = []
            elif metric_type == 'accumulate_value_over_samples':
                if metric_result["metric_value"] is not None:
                    metric_value_builder = create_mmap_dataset_builder(metric_result["metric_value_fname"],
                                                                       metric_dtype)
                    metric_value_builder.add_item(metric_result["metric_value"].reshape(-1))
                    close_mmap_dataset_builder(metric_value_builder, metric_result["metric_value_fname"])

    def run_map_helper(self, thread_id):
        start_idx, end_idx = self.thread_splits[thread_id][0], \
            self.thread_splits[thread_id][1]
        logger.info(f"worker {self.worker_id} thread {thread_id}: start working " \
            f"on data subset {start_idx} to {end_idx}")
        thread_dataset = Subset(self.dataset, list(range(start_idx, end_idx)))
        sampler = BatchSampler(SequentialSampler(thread_dataset), batch_size=self.batch_size, drop_last=False)
        iterator = iter(
            DataLoader(thread_dataset,
                       batch_sampler=sampler,
                       num_workers=0,
                       collate_fn=self.collate_fn,
                       pin_memory=False))
        if self.custom_map_init is None:
            metric_results = self.init_metric_results(thread_id, self.metric_names, self.metric_types,
                                                      self.metric_dtypes, self.save_path, self.worker_id)
        else:
            metric_results = self.custom_map_init(thread_id, self.metric_names, self.metric_types, self.metric_dtypes,
                                                  self.save_path, self.worker_id)
        total_sample = len(thread_dataset)
        processed_sample = 0
        start = time.time()
        while True:
            try:
                data = next(iterator)
                batch_start_idx = start_idx + processed_sample
                if self.custom_map_update is None:
                    self.update_metric_results(data, self.metric_types, self.metric_dtypes, self.metric_functions,
                                               metric_results, batch_start_idx)
                else:
                    self.custom_map_update(data, self.metric_types, self.metric_dtypes, self.metric_functions,
                                           metric_results, batch_start_idx)
                processed_sample += self.batch_size
                duration = (time.time() - start) / 3600.0
                remain_duration = duration * total_sample / processed_sample - duration
                logger.info(
                    f"worker {self.worker_id} thread {thread_id}: {processed_sample} " \
                    f"out of {total_sample} processed in {duration:.2f} hr, " \
                    f"estimated to finish in {remain_duration:.2f} hr")
            except StopIteration:
                logger.info(f"worker {self.worker_id} thread {thread_id}: reach end of file")
                break
        if self.custom_map_finalize is None:
            self.finalize_metric_results(self.metric_types, self.metric_dtypes, metric_results)
        else:
            self.custom_map_finalize(self.metric_types, self.metric_dtypes, metric_results)
        logger.info(f"worker {self.worker_id} thread {thread_id}: finished")

    def run_map(self):
        self.worker_splits, self.thread_splits = split_dataset(self.dataset, self.num_workers, self.worker_id,
                                                               self.num_threads)
        if len(self.specific_threads) > 0:
            threads_to_run = self.specific_threads
        else:
            threads_to_run = list(range(self.num_threads))
        if self.num_threads > 1:
            p = []
            for thread in threads_to_run:
                p.append(Process(target=self.run_map_helper, args=(thread, )))
                p[thread].start()

            for thread in threads_to_run:
                p[thread].join()
        else:
            assert self.num_threads == 1
            self.run_map_helper(0)

    def get_metric_value_percentiles(self, metric_name, num_sample_per_value, total_num_samples):
        logger.info(f"Checking the value percentiles of metric {metric_name}...")
        processed_samples = 0
        current_percentile = 5
        for key in sorted(num_sample_per_value.keys()):
            processed_samples += num_sample_per_value[key]
            if processed_samples >= total_num_samples * current_percentile / 100.0:
                logger.info(f"Metric {metric_name} {current_percentile}th percentile: {key}")
                current_percentile += 5

    def merge_gather_map_stats(self, num_workers, num_threads, num_threads_reduce, t_idx_reduce, metric_save_path,
                               metric_name, return_dict):
        results = []
        for w_idx in range(num_workers):
            for t_idx in range(num_threads):
                if (w_idx * num_threads + t_idx) % num_threads_reduce == t_idx_reduce:
                    w_metric_save_path = f"{metric_save_path}/worker{w_idx}_thread{t_idx}/"
                    w_sample_to_metric_fname = f"{w_metric_save_path}/{metric_name}_sample_to_metric"
                    w_sample_to_metric = MMapIndexedDataset(w_sample_to_metric_fname, skip_warmup=True)
                    unique_v = list(np.unique(w_sample_to_metric))
                    sample_to_metric_count = len(w_sample_to_metric)
                    logger.info(f"Finished gathering map stats from worker {w_idx} thread {t_idx}.")
                    results.append([unique_v, sample_to_metric_count])
        return_dict[t_idx_reduce] = results

    def merge_sample_to_metric(self, t_idx_reduce, metric_save_path, metric_name, metric_value_dtype,
                               map_worker_thread):
        sample_to_metric_fname = f"{metric_save_path}/{metric_name}_sample_to_metric_thread{t_idx_reduce}"
        sample_to_metric_builder = create_mmap_dataset_builder(sample_to_metric_fname, metric_value_dtype)
        for w_t in map_worker_thread:
            w_metric_save_path = f"{metric_save_path}/worker{w_t[0]}_thread{w_t[1]}/"
            w_sample_to_metric_fname = f"{w_metric_save_path}/{metric_name}_sample_to_metric"
            w_data = MMapIndexedDataset(w_sample_to_metric_fname, skip_warmup=True)
            for row in range(len(w_data)):
                sample_to_metric_builder.add_item(torch.tensor(w_data[row].astype(np.int64), dtype=torch.long))
            logger.info(f"Finished merge_sample_to_metric from worker {w_t[0]} thread {w_t[1]}.")
        close_mmap_dataset_builder(sample_to_metric_builder, sample_to_metric_fname)

    def merge_metric_to_sample(self, t_idx_reduce, metric_save_path, metric_name, sample_idx_dtype, metric_value_dtype,
                               unique_metric_values, num_workers, num_threads):
        index_to_sample_fname = f"{metric_save_path}/{metric_name}_index_to_sample_thread{t_idx_reduce}"
        index_to_sample_builder = create_mmap_dataset_builder(index_to_sample_fname, sample_idx_dtype)
        index_to_metric_fname = f"{metric_save_path}/{metric_name}_index_to_metric_thread{t_idx_reduce}"
        index_to_metric_builder = create_mmap_dataset_builder(index_to_metric_fname, metric_value_dtype)
        for unique_v in unique_metric_values:
            samples = []
            for w_idx in range(num_workers):
                for t_idx in range(num_threads):
                    w_metric_save_path = f"{metric_save_path}/worker{w_idx}_thread{t_idx}/"
                    w_metric_to_sample_fname = f"{w_metric_save_path}/{metric_name}_metric_to_sample_{unique_v}.csv"
                    if os.path.isfile(w_metric_to_sample_fname):
                        with open(w_metric_to_sample_fname, 'r') as f:
                            datareader = csv.reader(f)
                            for row in datareader:
                                samples += [int(x) for x in row]
            index_to_sample_builder.add_item(torch.tensor(samples, dtype=torch.long))
            index_to_metric_builder.add_item(torch.tensor([unique_v], dtype=torch.long))
            logger.info(f"Finished reducing metric {metric_name} value {unique_v}.")
        close_mmap_dataset_builder(index_to_sample_builder, index_to_sample_fname)
        close_mmap_dataset_builder(index_to_metric_builder, index_to_metric_fname)

    def merge_map_results(self, dataset, metric_names, metric_types, save_path, num_workers, num_threads,
                          num_threads_reduce):
        total_num_samples = len(dataset)
        sample_idx_dtype = find_fit_int_dtype(0, total_num_samples - 1)
        logger.info(
            f"Total number of data samples: {total_num_samples}. Will use {sample_idx_dtype} to store the sample indexes."
        )
        for m_idx in range(len(metric_names)):
            metric_name, metric_type = metric_names[m_idx], metric_types[m_idx]
            if metric_type == 'single_value_per_sample':
                metric_save_path = f"{save_path}/{metric_name}/"
                sample_to_metric_count = 0
                unique_metric_values = set([])
                manager = Manager()
                return_dict = manager.dict()
                p = []
                for t_idx_reduce in range(num_threads_reduce):
                    p.append(
                        Process(target=self.merge_gather_map_stats,
                                args=(
                                    num_workers,
                                    num_threads,
                                    num_threads_reduce,
                                    t_idx_reduce,
                                    metric_save_path,
                                    metric_name,
                                    return_dict,
                                )))
                    p[t_idx_reduce].start()
                for t_idx_reduce in range(num_threads_reduce):
                    p[t_idx_reduce].join()
                for t_idx_reduce in range(num_threads_reduce):
                    results = return_dict[t_idx_reduce]
                    for res in results:
                        unique_metric_values = unique_metric_values.union(set(res[0]))
                        sample_to_metric_count += res[1]
                value_max = max(unique_metric_values)
                value_min = min(unique_metric_values)
                assert sample_to_metric_count == total_num_samples, "The number of samples in map result files are not correct. It's possible that some map worker didn't finish successfully."
                metric_value_dtype = find_fit_int_dtype(value_min, value_max)
                logger.info(
                    f"Metric {metric_name} has values between {value_min} and {value_max}. Will use {metric_value_dtype} to store the metric values."
                )

                # sample_to_metric
                map_worker_thread = []
                for w_idx in range(num_workers):
                    for t_idx in range(num_threads):
                        map_worker_thread.append([w_idx, t_idx])
                thread_splits = split_index(0, len(map_worker_thread), num_threads_reduce)
                p = []
                for t_idx_reduce in range(num_threads_reduce):
                    start_idx, end_idx = thread_splits[t_idx_reduce][0], thread_splits[t_idx_reduce][1]
                    p.append(
                        Process(target=self.merge_sample_to_metric,
                                args=(
                                    t_idx_reduce,
                                    metric_save_path,
                                    metric_name,
                                    metric_value_dtype,
                                    map_worker_thread[start_idx:end_idx],
                                )))
                    p[t_idx_reduce].start()
                for t_idx_reduce in range(num_threads_reduce):
                    p[t_idx_reduce].join()

                sample_to_metric_fname = f"{metric_save_path}/{metric_name}_sample_to_metric"
                sample_to_metric_builder = create_mmap_dataset_builder(sample_to_metric_fname, metric_value_dtype)
                for t_idx_reduce in range(num_threads_reduce):
                    chunk_fname = f"{metric_save_path}/{metric_name}_sample_to_metric_thread{t_idx_reduce}"
                    logger.info(f"Merging file {chunk_fname}")
                    sample_to_metric_builder.merge_file_(chunk_fname)
                close_mmap_dataset_builder(sample_to_metric_builder, sample_to_metric_fname)
                sample_to_metric = MMapIndexedDataset(sample_to_metric_fname, skip_warmup=True)
                assert len(sample_to_metric) == total_num_samples

                # metric_to_sample
                unique_metric_values = list(sorted(unique_metric_values))
                thread_splits = split_index(0, len(unique_metric_values), num_threads_reduce)
                p = []
                for t_idx_reduce in range(num_threads_reduce):
                    start_idx, end_idx = thread_splits[t_idx_reduce][0], thread_splits[t_idx_reduce][1]
                    p.append(
                        Process(target=self.merge_metric_to_sample,
                                args=(
                                    t_idx_reduce,
                                    metric_save_path,
                                    metric_name,
                                    sample_idx_dtype,
                                    metric_value_dtype,
                                    unique_metric_values[start_idx:end_idx],
                                    num_workers,
                                    num_threads,
                                )))
                    p[t_idx_reduce].start()
                for t_idx_reduce in range(num_threads_reduce):
                    p[t_idx_reduce].join()
                index_to_sample_fname = f"{metric_save_path}/{metric_name}_index_to_sample"
                index_to_sample_builder = create_mmap_dataset_builder(index_to_sample_fname, sample_idx_dtype)
                index_to_metric_fname = f"{metric_save_path}/{metric_name}_index_to_metric"
                index_to_metric_builder = create_mmap_dataset_builder(index_to_metric_fname, metric_value_dtype)
                for t_idx_reduce in range(num_threads_reduce):
                    chunk_is_fname = f"{metric_save_path}/{metric_name}_index_to_sample_thread{t_idx_reduce}"
                    logger.info(f"Merging file {chunk_is_fname}")
                    index_to_sample_builder.merge_file_(chunk_is_fname)
                    chunk_im_fname = f"{metric_save_path}/{metric_name}_index_to_metric_thread{t_idx_reduce}"
                    logger.info(f"Merging file {chunk_im_fname}")
                    index_to_metric_builder.merge_file_(chunk_im_fname)
                close_mmap_dataset_builder(index_to_sample_builder, index_to_sample_fname)
                close_mmap_dataset_builder(index_to_metric_builder, index_to_metric_fname)
                num_sample_per_value = {}
                index_to_sample = MMapIndexedDataset(index_to_sample_fname, skip_warmup=True)
                index_to_metric = MMapIndexedDataset(index_to_metric_fname, skip_warmup=True)
                index_to_sample_merged_fname = f"{metric_save_path}/{metric_name}_index_to_sample_percentile_merged"
                index_to_sample_merged_builder = create_mmap_dataset_builder(index_to_sample_merged_fname,
                                                                             sample_idx_dtype)
                for v_idx in range(len(index_to_sample)):
                    if v_idx > 0:
                        assert index_to_metric[v_idx] > index_to_metric[v_idx - 1]
                    num_sample_per_value[index_to_metric[v_idx][0]] = len(index_to_sample[v_idx])
                assert sum(num_sample_per_value.values()) == total_num_samples
                merge_step = max(1, len(index_to_sample) // 100)
                for v_idx in range(0, len(index_to_sample), merge_step):
                    merged_samples = np.copy(
                        np.concatenate(index_to_sample[v_idx:min(len(index_to_sample), (v_idx + merge_step))],
                                       axis=None))
                    index_to_sample_merged_builder.add_item(
                        torch.tensor(merged_samples.astype(np.int64), dtype=torch.long))
                    logger.info(f"Finished merging index_to_sample {v_idx} to {v_idx+merge_step}.")
                close_mmap_dataset_builder(index_to_sample_merged_builder, index_to_sample_merged_fname)
                self.get_metric_value_percentiles(metric_name, num_sample_per_value, total_num_samples)
            elif metric_type == 'accumulate_value_over_samples':
                metric_save_path = f"{save_path}/{metric_name}/"
                metric_value = None
                for w_idx in range(num_workers):
                    for t_idx in range(num_threads):
                        w_metric_save_path = f"{metric_save_path}/worker{w_idx}_thread{t_idx}/"
                        w_metric_value_fname = f"{w_metric_save_path}/{metric_name}_metric_value"
                        w_metric_value = MMapIndexedDataset(w_metric_value_fname, skip_warmup=True)
                        if metric_value is None:
                            metric_value = np.copy(w_metric_value[0])
                        else:
                            metric_value += np.copy(w_metric_value[0])
                value_max = int(max(metric_value))
                value_min = int(min(metric_value))
                metric_value_dtype = find_fit_int_dtype(value_min, value_max)
                metric_value_fname = f"{metric_save_path}/{metric_name}_metric_value"
                metric_value_builder = create_mmap_dataset_builder(metric_value_fname, metric_value_dtype)
                metric_value_builder.add_item(torch.tensor(metric_value.astype(np.int64), dtype=torch.long))
                close_mmap_dataset_builder(metric_value_builder, metric_value_fname)

    def run_reduce(self):
        if self.custom_reduce is None:
            self.merge_map_results(self.dataset, self.metric_names, self.metric_types, self.save_path,
                                   self.num_workers, self.num_threads, self.num_threads_reduce)
        else:
            self.custom_reduce(self.dataset, self.metric_names, self.metric_types, self.save_path, self.num_workers,
                               self.num_threads, self.num_threads_reduce)

    def run_map_reduce(self, comm_group=None):
        self.run_map()
        # wait for the mapping operation, where all nodes outputs their own (partial) result files
        dist.barrier(group=comm_group)
        if self.worker_id == 0:
            self.run_reduce()
        # wait for the reduce, where rank 0 merges all (partial) files. Dataset can then be used by all nodes.
        dist.barrier(group=comm_group)
        

class DistributedDataAnalyzer(object):

    def __init__(self,
            dataset,
            num_workers=1,
            worker_id=0,
            batch_size=1,
            metric_names=[],
            metric_functions=[],
            metric_types=[],
            save_path="./",
            collate_fn=None,
            device='cuda',
            comm_group=None,
            sample_indices=None,
        ) -> None:
        self.dataset = dataset
        self.comm_group = comm_group
        self.batch_size = batch_size
        self.metric_names = metric_names
        self.metric_functions = metric_functions
        self.metric_types = metric_types
        self.save_path = save_path
        self.collate_fn = collate_fn
        self.device = device
        self.sample_indices = sample_indices

        # comm_group and num_workers/worker_id are mutually exclusive
        self.comm_group = comm_group
        if comm_group is not None:
            self.num_workers = comm_group.size()
            self.worker_id = comm_group.rank()
        else:
            self.num_workers = num_workers
            self.worker_id = worker_id


    def run_map_reduce(self):

        # setup individual dataloaders
        worker_splits, _ = split_dataset(self.dataset, self.num_workers, self.worker_id, num_threads=1)
        start_idx, end_idx = worker_splits[self.worker_id]
        logger.info(f"worker {self.worker_id}: start working on data subset {start_idx} to {end_idx}")
        worker_dataset = Subset(self.dataset, list(range(start_idx, end_idx)))
        sampler = BatchSampler(SequentialSampler(worker_dataset), batch_size=self.batch_size, drop_last=False)
        dataloader = DataLoader(dataset=worker_dataset, batch_sampler=sampler,
                                num_workers=0, collate_fn=self.collate_fn, pin_memory=False)

        # set initial results list
        metric_results = []
        for metric_type in self.metric_types:
            assert metric_type in ['single_value_per_sample', 'accumulate_value_over_samples'], \
                f"metric_type {metric_type} not implemented."
            metric_results.append( [] if metric_type == 'single_value_per_sample' else None )

        # iterate dataloader and store metric results
        batch_start_idx = start_idx
        for data in dataloader:
            for m_idx in range(len(self.metric_names)):
                metric_type, metric_function, metric_result = \
                    self.metric_types[m_idx], self.metric_functions[m_idx], metric_results[m_idx]
                metric_values = metric_function(data)
                assert torch.is_tensor(metric_values) or isinstance(metric_values, np.ndarray), \
                    "metric_function must return a tensor or array"
                if isinstance(metric_values, np.ndarray):
                    metric_values = torch.from_numpy(metric_values)
                assert metric_values.dtype in valid_dtypes, \
                    f"metric_function result dtype {metric_values.dtype} not supported. Supported dtypes {valid_dtypes}"

                if metric_type == 'single_value_per_sample':
                    for row in range(metric_values.size()[0]):
                        value = metric_values[row].item()
                        sample_idx = batch_start_idx + row  # sample idx following dataset iteration order
                        if isinstance(data, dict) and 'index' in data:  # Megatron use case, idx provided in 'index' field
                            sample_idx = data['index'][row][0].item()
                        elif self.sample_indices is not None:  # user defined shuffling of indices
                            sample_idx = self.sample_indices[sample_idx]
                        metric_result.append((value, sample_idx))
                elif metric_type == 'accumulate_value_over_samples':
                    if metric_result is None:
                        metric_result = metric_values
                    else:
                        metric_result.add_(metric_values)
            batch_start_idx += self.batch_size

        # compute dtype for sample ids
        total_num_samples = len(self.dataset)
        sample_idx_dtype = find_fit_int_dtype(0, total_num_samples - 1)
        logger.info(f"Total number of data samples: {total_num_samples}.")
        logger.info(f"Will use {sample_idx_dtype} to store the sample indexes.")

        # convert to list of tensors
        metric_results = [ torch.tensor(m).to(self.device) for m in metric_results ]

        for m_idx in range(len(self.metric_names)):
            metric_values, metric_name, metric_type = \
                metric_results[m_idx], self.metric_names[m_idx], self.metric_types[m_idx]
            metric_save_path = f"{self.save_path}/{metric_name}/"
            if metric_type == 'single_value_per_sample':


                # sample_to_metric maps sample ids to metric values, as a list of metric values
                sample_to_metric_fname = f"{metric_save_path}/{metric_name}_sample_to_metric"
                values = [torch.tensor([x]) for x in metric_values[:,0]]
                self.file_write_ordered(values, sample_to_metric_fname, torch.long)

                # index_to_metric and index_to_sample serialize a dicitonary from metric to samples
                # index_to_metric stores a key per row, index_to_sample stores the values per row

                # distributed sorting by values, gives an ordered disjoint subset of keys on nodes
                metric_values = DistributedDataAnalyzer.dist_sample_sort(metric_values, self.comm_group, self.num_workers)
                metric_to_sample_dict = {}
                for value, sample in metric_values:
                    if value.item() not in metric_to_sample_dict:
                        metric_to_sample_dict[value.item()] = []
                    metric_to_sample_dict[value.item()].append(sample.item())
                values = [torch.tensor([x]) for x in metric_to_sample_dict.keys()]
                samples = [torch.tensor(metric_to_sample_dict[x]) for x in metric_to_sample_dict.keys()]
                index_to_metric_fname = f"{metric_save_path}/{metric_name}_index_to_metric" #dict keys
                index_to_sample_fname = f"{metric_save_path}/{metric_name}_index_to_sample" #dict values
                self.file_write_ordered(values, index_to_metric_fname, torch.long)
                self.file_write_ordered(samples, index_to_sample_fname, torch.long)

                # get unique values across all ranks and compute the values dtype based on min/max
                # value_min, value_max = DistributedDataAnalyzer.dist_min_max(values, self.comm_group)
                # metric_value_dtype = find_fit_int_dtype(value_min, value_max)

            elif metric_type == 'accumulate_value_over_samples':
                metric_value_fname = f"{metric_save_path}/{metric_name}_metric_value"
                dist.all_reduce(metric_values, op=dist.ReduceOp.SUM, group=self.comm_group)
                if self.worker_id == 0:
                    metric_values = metric_values.cpu().numpy()
                    builder = create_mmap_dataset_builder(metric_value_fname, torch.long)
                    builder.add_item(metric_values)
                    close_mmap_dataset_builder(builder, metric_value_fname)


    def file_write_ordered(self, tensor_list, fname, numpy_dtype):
        """ save a distributed tensor to a single file, iteratively, ordered by rank """

        # each not has a list of rows (tensors) to be written to the file.
        # we will serialize it to communicate it in one comm step.

        tkwargs = dict(dtype=torch.int64, device=self.device)

        # 1. gather on rank 0 the number of rows to be sent/recv
        row_count = torch.tensor(len(tensor_list), **tkwargs)
        row_counts = torch.zeros(self.num_workers, **tkwargs)
        dist.all_gather_into_tensor(row_counts, row_count, group=self.comm_group)
        assert row_counts[self.worker_id]==row_count, "all_gather failed" #sanity check

        # 3. gather on rank 0 the sizes of the rows to be sent/recv
        # (all_gather requires all tensors to be of same size so we need pad them)
        max_size = max(row_counts)
        row_lens, row_len = None, torch.zeros(max_size, **tkwargs)
        if self.worker_id == 0: # create padded recv buffers
            row_lens = [torch.zeros(max_size, **tkwargs)]*self.num_workers
        row_len[0:len(tensor_list)] = torch.tensor([len(l) for l in tensor_list], **tkwargs)
        dist.gather(row_len, row_lens, dst=0, group=self.comm_group)
        if self.worker_id == 0: # remove padding from buffers
            row_lens = [r[:s] for r,s in zip(row_lens, row_counts)]

        # 4. gather on rank 0 of the total size (sum of all row lengths) to be received
        size = torch.tensor(sum(row_len).item(), **tkwargs)
        sizes = torch.zeros(self.num_workers, **tkwargs)
        print("XXX", self.worker_id)
        dist.all_gather_into_tensor(sizes, size, group=self.comm_group)
        assert sizes[self.worker_id]==size, "all_gather did not return the same sizes" #sanity check

        # method to deserializes a buffer into rows of different lengths and write them to file
        def write_recv_buffer_to_file(recv_buffer, src, builder):
            assert self.worker_id == 0, "only rank 0 can write to file"
            for row_len in row_lens[src]:
                builder.add_item(recv_buffer[:row_len].cpu())
                recv_buffer = recv_buffer[row_len:]

        # 5. rank 0 receives all tensors sequentially and writes them to the file
        buffer = torch.cat(tensor_list, dim=0).to(self.device) #serialize list into buffer
        if self.worker_id == 0:
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            builder = create_mmap_dataset_builder(fname, numpy_dtype)
            write_recv_buffer_to_file(buffer, 0, builder)

        for src in range(1, self.num_workers):
            dist.barrier(group=self.comm_group)
            if src == self.worker_id:
                dist.send(buffer, 0, group=self.comm_group) # send tensor
            elif self.worker_id == 0:
                buffer = torch.zeros(sizes[src].item(), dtype=buffer.dtype, device=buffer.device)
                dist.recv(buffer, src=src, group=self.comm_group)
                write_recv_buffer_to_file(buffer, src, builder)

        # rank 0 closes the file
        if self.worker_id == 0:
            close_mmap_dataset_builder(builder, fname) # close file
        dist.barrier(self.comm_group)


    @staticmethod
    def dist_min_max(tensor, comm_group):
        """ given a distributed tensor, return the min/max values across all ranks"""

        value_min, value_max = tensor.min(), tensor.max()
        dist.reduce(value_min, 0, op=dist.ReduceOp.MIN, group=comm_group)
        dist.reduce(value_max, 0, op=dist.ReduceOp.MAX, group=comm_group)
        return value_min.item(), value_max.item()


    @staticmethod
    def dist_sample_sort(tensor, comm_group, num_workers, n_samples=100):
        """ perform a distributed random sort of a tensor, and returns the sorted partial tensor"""
        device, dims = tensor.device, tensor.size()[1]

        # 1 - Sort locally
        tensor = torch.sort(tensor, dim=0)[0]

        # 2 - collect few samples per rank
        idx = torch.round(torch.linspace(0, len(tensor) - 1, n_samples+1)).to(int)
        samples = tensor[idx[:-1]][:,0].contiguous().to(device) #only first column, all but last row

        # 2 - Allgather samples
        all_samples = [torch.zeros(n_samples, dtype=samples.dtype, device=device)] * num_workers
        dist.all_gather(all_samples, samples, group=comm_group)
        all_samples = torch.cat(all_samples, dim=0).to(device)

        # 3 - Sort all samples and collect the ranges of each rank as equidistant
        all_samples = all_samples.sort()[0]
        idx = torch.round(torch.linspace(0, len(all_samples) - 1, num_workers + 1)).to(int)
        ranges = all_samples[idx] # range of each rank r as ranges[r] <= x < ranges[r+1]
        ranges[-1] += 1  # increase upper limit of last rank so that x < ranges[r+1].

        # 4 - collect elements to send to each rank, based on the rank ranges
        send = []
        for rank in range(num_workers):
            mask = (tensor[:,0] >= ranges[rank]) & (tensor[:,0] < ranges[rank+1])
            send.append(tensor[mask])

        # 5. all to all to communicate the sizes to be sent/recv
        send_count = [ torch.tensor([len(tensor)*dims], dtype=torch.int64, device=device) for tensor in send]
        recv_count = list(torch.zeros([num_workers], dtype=torch.int64, device=device).chunk(num_workers))
        dist.all_to_all(recv_count, send_count, group=comm_group)

        # 6. all to all to communicate the elements to be sent/recv as a single tensor
        send = torch.cat(send, dim=0).flatten().to(device)
        recv = torch.zeros( sum(recv_count), dtype=send.dtype).to(device)
        dist.all_to_all_single(recv, send, recv_count, send_count, group=comm_group)
        del send

        # 7. the received tensor is the 1D disjoint subset of the distributed tensor
        return recv.view(-1, dims)

