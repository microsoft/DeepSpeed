import os
import random

import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from bert_dataset_provider import BertDatasetProviderInterface
from turing.dataset import PreTrainingDataset, PretrainDataType
from data_worker import AsyncWorker


class BingBertDatasetProvider(BertDatasetProviderInterface):
    def __init__(self, args):
        self.tokenizer = args.tokenizer
        self.refresh_bucket_size = args.refresh_bucket_size
        self.datasampler = RandomSampler if args.local_rank == -1 else DistributedSampler
        self.num_workers = args.config['training']['num_workers']

        # Initialize dataset paths
        self.dataset_paths = []
        for dataset in ['wiki_pretrain_dataset', 'bc_pretrain_dataset']:
            self.dataset_paths.append(
                os.path.join(args.data_path_prefix,
                             args.config["data"]["datasets"][dataset]))

        self.max_seq_length = args.max_seq_length
        self.max_predictions_per_seq = args.max_predictions_per_seq

        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.train_micro_batch_size_per_gpu = args.train_micro_batch_size_per_gpu
        self.local_rank = args.local_rank
        self.global_rank = dist.get_rank()
        self.world_size = 1 if self.local_rank == -1 else dist.get_world_size()
        self.logger = args.logger

        self.dataloaders = {}
        self.dataset_iterator = []

        # Configure asynchronous data loading
        self.async_dataloading = 'async_worker' in args.config['training']
        self.async_worker = None

        if self.global_rank == 0:
            self.logger.info(
                f"BingBertDatasetProvider - Initialization:  async data loading {self.async_dataloading}"
            )

    def get_shard(self, index, shuffle=True):
        datalengths = []
        batches_per_dataset = []

        for i, dataset_path in enumerate(self.dataset_paths):
            pretrain_dataset = PreTrainingDataset(
                tokenizer=self.tokenizer,
                folder=dataset_path,
                logger=self.logger,
                max_seq_length=self.max_seq_length,
                index=index,
                data_type=PretrainDataType.NUMPY,
                max_predictions_per_seq=self.max_predictions_per_seq)

            datalengths.append(len(pretrain_dataset))
            batches_per_dataset.append(
                self._get_effective_batch(len(pretrain_dataset)))
            self.dataloaders[i] = self._get_dataloader(pretrain_dataset)

        dataset_batches = []
        for i, batch_count in enumerate(batches_per_dataset):
            dataset_batches.extend([i] * batch_count)

        # shuffle
        if shuffle:
            random.shuffle(dataset_batches)

        self.dataset_iterator = []
        for dataset_batch_type in dataset_batches:
            self.dataset_iterator.extend([dataset_batch_type] *
                                         self.gradient_accumulation_steps *
                                         self.refresh_bucket_size)

        if self.async_dataloading:
            self.async_worker = AsyncWorker(self.dataloaders,
                                            self.dataset_iterator)
            self.async_worker.start()

        return self.dataset_iterator, sum(datalengths)

    def release_shard(self, index):
        if self.async_dataloading:
            self.async_worker.stop()

    def prefetch_shard(self, index):
        pass

    def get_batch(self, batch_iter):
        if self.async_dataloading:
            return self.async_worker.get()
        return next(self.dataloaders[batch_iter])

    def prefetch_batch(self):
        if self.async_dataloading:
            self.async_worker.prefetch()

    def _get_dataloader(self, dataset: Dataset):
        return (
            x
            for x in DataLoader(dataset,
                                batch_size=self.train_micro_batch_size_per_gpu,
                                sampler=self.datasampler(dataset),
                                num_workers=self.num_workers))

    def _get_effective_batch(self, total):
        return total // self.world_size // self.train_micro_batch_size_per_gpu // self.gradient_accumulation_steps // self.refresh_bucket_size
