import random
import torch
from deepspeed.utils import logger
from torch.utils.data import DistributedSampler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
import deepspeed


# see https://github.com/facebookresearch/fairseq/blob/b5a039c292facba9c73f59ff34621ec131d82341/fairseq/data/data_utils.py#L282
# see how to set new batch size here:
# https://github.com/microsoft/DeepSpeed/issues/2798#issuecomment-1435475061
# engine.set_train_micro_batch_size and set_train_batch_size (only changes grad acc steps) in
# https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/engine.py
# TODO we need same batch size per GPU per grad step!


def batch_by_size(
    metric_values,
    max_metric_value_per_batch,
    sample_ids=None,
    min_batch_size=1,
    max_batch_size=None,
    shuffle_metric_values=False,
    order_by_metric_value=False,
    dataloader_num_replicas=1,
    gradient_accumulation_steps=1,
    required_microbatches_of_same_size=False,
    verbose=False,
    ):

    """
    Yield mini-batches of indices bucketed by size. Batches may contain sequences of different lengths.
    Similar to "Attention is all you need", Section 5.1: 
    "Sentence pairs were batched together by approximate sequence length. Each training batch
    contained a set of sentence pairs containing approximately X source tokens and X target tokens"

    Arguments:
    - `metric_values`: a list of difficulties (metric values) for every sample in the dataset;
    - `max_metric_value_per_batch`: upper cap in total difficulty in a batch;
    - `sample_ids`: user-defined ids of the samples in metric_values. If not provided,
      automatically assigns a sequential order;
    - `min_batch_size`: smallest allowed size of a batch;
    - `min_batch_size`: largest allowed size of a batch;
    - `shuffle_metric_values`: shuffle metric values before packing samples into batches;
    - `order_by_metric_value`: order samples by ascending metric values before packing into batches;
    - `dataloader_num_replicas`: number of dataloaders
    - `gradient_accumulation_steps`: number of gradient accumulation steps;
    - `required_microbatches_of_same_size`: enable if each mini-batch (in a total of `batch_size_multiple`
       micro-batches per batch), should have all micro-batches with the same batch size.
       Required for pipeline parallelism (as activation shapes is uniform across mini-batches), or
       in regular data parallelism if we want the same number of samples per accumulation step.

    Returns a list of the ids of each micro-batch and a list of effective batch sizes.
    """

    assert not shuffle_metric_values or not order_by_metric_value, \
        "either sort_metric_values or shuffle_metric_values can be True, not both."

    sample_ids = sample_ids or list(range(len(metric_values))) 
    metrics = list(zip(metric_values, sample_ids))

    if shuffle_metric_values:
        random.shuffle(metrics)
    if order_by_metric_value:
        metrics = sorted(metrics)

    # go through metrics and warn user and filter samples that alone exceed the max batch threshold
    long_ids = [ idx for val, idx in metrics if val>max_metric_value_per_batch ]
    if len(long_ids)>0:
        logger.warning(f"Data indices {long_ids} ignored as metrics exceed {max_metric_value_per_batch}.")
        logger.info(f"Original dataset length: {len(metrics)}. New dataset length: {len(long_ids)}")
        metrics = [ m for m in metrics if m[1] not in long_ids ]

    def is_microbatch_valid(metrics):
        if len(metrics) < min_batch_size: return False # insufficient sample count
        if max_batch_size and len(metrics)>max_batch_size: return False # too many samples
        if sum([m[0] for m in metrics]) > max_metric_value_per_batch: return False # exceeds max
        return True
    
    # go through all samples and pack then in microbatches of metric sums below the threshold
    # `required_microbatches_of_same_size` means all minibatches in a batch must be of equal size
    num_microbatches_per_batch = dataloader_num_replicas * gradient_accumulation_steps
    equal_size_multiple = num_microbatches_per_batch if required_microbatches_of_same_size else 1
    microbatches = []
    batch_init = 0
    while batch_init < len(metrics):

        # we iterate over possible effective batch sizes (groups of microbatches of same size)
        for batch_size in range(equal_size_multiple, len(metrics), equal_size_multiple):

            # attempt effective batch
            batch = metrics[batch_init:batch_init+batch_size]

            # pick interleaved samples for each microbatch to help with load balancing
            # (in the ordered use case), and to replicate what the distributed sampler does.
            microbatch = [ batch[b::equal_size_multiple] for b in range(equal_size_multiple) ]

            # if they are all valid micro-batches, keep them until you find longer mbatches, if any
            is_batch_valid = all([is_microbatch_valid(mb) for mb in microbatch] )
            if not is_batch_valid:
                break

        if not is_batch_valid: batch_size -= equal_size_multiple #ignore last iteration (not valid)
        batch = metrics[batch_init:batch_init+batch_size]
        microbatch = [ batch[b::equal_size_multiple] for b in range(equal_size_multiple) ]
        batch_init += sum( [ len(l) for l in microbatch ] )
        microbatches += microbatch

    # make sure we give the same number of (micro-)batches to each dataloader by trimming dataset
    microbatches = microbatches[:len(microbatches) - len(microbatches) % num_microbatches_per_batch]

    #compute the effective batch size for each microbatch.
    batch_sizes, batch_metrics, microbatch_sample_ids = [], [], []
    for rank in range(0, len(microbatches), num_microbatches_per_batch):
        microbatch = microbatches[rank: rank+num_microbatches_per_batch]
        batch_size = sum([len(mb) for mb in microbatch])
        batch_metric = sum([m[0] for m in microbatch[0]])
        batch_sample_ids = [ [m[1] for m in metrics] for metrics in microbatch]
        batch_sizes.append(batch_size)
        batch_metrics.append(batch_metric)
        microbatch_sample_ids += batch_sample_ids
        if verbose:
            print(f"Batch size {batch_size} samples, metric value {batch_metric}, samples: {batch_sample_ids}")

    # return the sample ids of each microbatch, and the batch sizes 
    assert len(batch_sizes) == len(microbatch_sample_ids)//num_microbatches_per_batch
    return microbatch_sample_ids, batch_sizes, batch_metrics
        

def scale_lr(base_batch_size, batch_size, base_lr=1, method="linear"):
    """ given a reference lr and batch_size, compute the new LR for a given batch size """
    if method == "linear":
        # Linear Scaling Rule: "When the minibatch size is multiplied by k, multiply the learning,
        # rate by k" (Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour, Goyal et al)
        return base_lr * batch_size / base_batch_size
    if method == "sqrt":
        # Square Root scaling: "when multiplying the batch size by k, multiply the learning rate
        # by √k, to keep the variance in the gradient expectation constant" 
        # (A. Krizhevsky. One weird trick for parallelizing convolutional neural networks)
        return base_lr * torch.sqrt(batch_size / base_batch_size)
    raise ValueError("Unknown scaling method: {}".format(method))


def dataloader_for_variable_batch_size(dataset, 
        microbatch_sample_ids, dataloader_rank, dataloader_num_replicas, dataloader_collate_fn,
        dataloader_num_workers=2, dataloader_pin_memory=False): 

        # equidistantly distribute the microbatches across the replicas in an interleaved fashion.
        sampler = DistributedSampler(
            dataset=microbatch_sample_ids,
            num_replicas=dataloader_num_replicas,
            rank=dataloader_rank,
            shuffle=False,
            drop_last=False,
            )

        # collate function applies wraps user defined collate function to the variable batch data  
        def collate_fn_wrapper(batch_sample_ids, dataset, collate_fn=None):
            # batch is a list of sample ids per microbatch
            assert len(batch_sample_ids)==1, "only 1 element should be returned by the sampler."
            batch_data = [dataset[idx] for idx in batch_sample_ids[0]]
            return collate_fn(batch_data) if collate_fn else batch_data

        collate_fn = lambda b: collate_fn_wrapper(b, dataset, dataloader_collate_fn)

        dataloader = DataLoader(
                dataset=microbatch_sample_ids,
                sampler=sampler,
                num_workers = dataloader_num_workers,
                collate_fn = collate_fn,
                pin_memory=dataloader_pin_memory,
            )

        deepspeed_io_kwargs = dict(dataset=dataset,
                        batch_size=1,
                        pin_memory=dataloader_pin_memory,
                        data_sampler=sampler,
                        collate_fn=collate_fn,
                        num_local_io_workers=dataloader_num_workers)
        
        return dataloader, deepspeed_io_kwargs
        

class StubLRScheduler(LRScheduler):
    """ a stub LR scheduler that does not change the LR, keeps it constant """
    def get_lr(self) -> float:
        return self.base_lrs

def lr_scheduler_for_variable_batch_size(
        base_batch_size, batch_sizes, lr_scaling_method='linear',
        optimizer=None, lr_scheduler_class=None, **lr_scheduler_kwargs):
    """
    returns a class that provides an LR scheduler that scales learning rate at every
    epoch taking into account the batch size of each epoch.
    If learning rate is constant, ie no LR scheduler, then `optimizer` must be provided.
    Otherwise, the base `LRScheduler` must be provided as  `lr_scheduler_class`.

    Arguments:
    - `base_batch_size`: the batch size that the base LR in the optimizer or scheduler refers to;
    - `lr_scaling_method`: method to use to scale LR - see `scale_lr()`;
    - `batch_sizes`: the effective batch size of each batch in the dataloader;
    - `optimizer` and `lr_scheduler_class`: the base LR scheduler. It not provided,
       will use the constant LRs from the optimizer's param groups instead. If provided,
       the initialization of the scheduler will be done with `lr_scheduler_kwargs`.

    Returns the new LRScheduler
    """

    class VariableBatchSizeLR(lr_scheduler_class or StubLRScheduler):

        def __init__(self, optimizer, **lr_scheduler_kwargs):
            self.batch_sizes = batch_sizes
            self.base_batch_size = base_batch_size
            self.lr_scaling_method = lr_scaling_method
            self.unscaled_lrs = [p['lr'] for p in optimizer.param_groups]
            super().__init__(optimizer=optimizer, **lr_scheduler_kwargs)

        def state_dict(self):
            return {
                'base': super().state_dict(),
                'base_batch_size': self.base_batch_size,
                'lr_scaling_method': self.lr_scaling_method,
                'unscaled_lrs': self.unscaled_lrs,
                'batch_sizes': self.batch_sizes
                }

        def load_state_dict(self, state_dict):
            super().load_state_dict(state_dict['base'])
            self.base_batch_size = state_dict['base_batch_size']
            self.lr_scaling_method = state_dict['lr_scaling_method']
            self.unscaled_lrs = state_dict['unscaled_lrs']
            self.batch_sizes = state_dict['batch_sizes']

        def step(self, epoch=None):
            
            # call the base scheduler's step method to get LR for next epoch
            # note: optimizer.step preceeds lr_scheduler.step(), so the stepping workflow is:
            # init: lr_scheduler.step(0) --> set LR for epoch 0
            # epoch 0: optimizer.step(); lr_scheduler.step(1) --> set LR for epoch 1
            # epoch 1: optimizer.step(); lr_scheduler.step(2) --> set LR for epoch 2

            # reset unscaled LRs (to the original scheduler's one) for the current epoch
            for param_group, lr in zip(self.optimizer.param_groups, self.unscaled_lrs):
                param_group['lr'] = lr
            self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

            super().step(epoch) # set lr, _step_count and last_epoch (for next epoch), _last_lr
            self.unscaled_lrs = self.get_last_lr()[:] # backup next epoch LRs, cloned

            # scale the learning rate for next epoch for each parameter group.
            # if we reach the last element, assume looping of data, ie refer to the first element
            if self.last_epoch % len(self.batch_sizes) == 0:
                print("RESET")
            batch_size = self.batch_sizes[self.last_epoch % len(self.batch_sizes)]
            lr_multiplier = scale_lr(self.base_batch_size, batch_size, method=lr_scaling_method)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= lr_multiplier
            self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
            print(f"LRs: {self.unscaled_lrs}, scaled by {lr_multiplier}, scaled LR: {self.get_last_lr()}")


    if lr_scheduler_class is None:
        assert optimizer is not None, "optimizer must be provided if lr_scheduler_class is not"
    else:
        assert issubclass(lr_scheduler_class, LRScheduler), "lr_scheduler should be a LRScheduler"

    if optimizer is None:
        assert lr_scheduler_class is not None, "lr_scheduler_class must be provided if optimizer is not"
        optimizer = lr_scheduler_kwargs['optimizer']

    return VariableBatchSizeLR(optimizer=optimizer, **lr_scheduler_kwargs)


def get_dataloader_and_lr_scheduler_for_variable_batch_size(
        dataset,
        dataset_metric_values,
        max_metric_value_per_batch,
        base_batch_size,
        sample_ids=None,
        lr_scaling_method="linear",
        min_batch_size=1,
        max_batch_size=None,
        shuffle_metric_values=False,
        order_by_metric_value=False,
        gradient_accumulation_steps=1,
        pipeline_parallelism=False,
        dataloader_rank=0,
        dataloader_num_replicas=1,
        dataloader_num_workers=0,
        dataloader_collate_fn=None,
        dataloader_pin_memory=False,
        optimizer=None,
        lr_scheduler_class=None,
        lr_scheduler_kwargs={},
        verbose=False,
):

        # batch_by_size returns the effective batch size and the sample ids for each microbatch.
        # We will use the sample ids to retrieve the batches from the dataset, 
        # and the effective batch size to retrieve the scaled learning rate for each batch
        # Note: pipelining in DeepSpeed takes the first micro-batch activation shape as reference.
        # So we need to make sure batch size remains contant across all microbatches in a batch.
        microbatch_sample_ids, batch_sizes, batch_metrics = batch_by_size(
            metric_values=dataset_metric_values,
            max_metric_value_per_batch=max_metric_value_per_batch,
            sample_ids=sample_ids,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            shuffle_metric_values=shuffle_metric_values,
            order_by_metric_value=order_by_metric_value,
            dataloader_num_replicas=dataloader_num_replicas,
            gradient_accumulation_steps=gradient_accumulation_steps,
            required_microbatches_of_same_size=pipeline_parallelism,
            verbose=verbose,
        )

        dataloader, deepspeed_io_kwargs = dataloader_for_variable_batch_size(
            dataset=dataset,
            microbatch_sample_ids=microbatch_sample_ids,
            dataloader_rank=dataloader_rank,
            dataloader_num_replicas=dataloader_num_replicas,
            dataloader_collate_fn=dataloader_collate_fn,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_pin_memory=dataloader_pin_memory,
        )
            
        lr_scheduler = lr_scheduler_for_variable_batch_size(
            base_batch_size=base_batch_size,
            batch_sizes=batch_sizes,
            lr_scaling_method=lr_scaling_method,
            optimizer=optimizer,
            lr_scheduler_class=lr_scheduler_class,
            **lr_scheduler_kwargs)

        return dataloader, lr_scheduler, deepspeed_io_kwargs


if __name__ == "__main__":

    # A small example/test on how to use this module

    from torch.utils.data import Dataset
    class TestData(Dataset):
        """ A test dataset with sequences of random length, and their sum as the target"""
        def __init__(self, seq_count, min_seq_len=1, max_seq_len=21):
            self.seqs = [ torch.ones(random.randrange(min_seq_len,max_seq_len)) for _ in range(seq_count) ]
        
        __len__ = lambda self: len(self.seqs)
        __getitem__ = lambda self, idx: [self.seqs[idx], self.seqs[idx].sum()]

        # collate_fn merges sequences, padded to the max length, or trimmed/paded to a value
        @staticmethod
        def collate_fn(batch, max_seq_len=None):
            # if max_seq_len in enforces, trim/pad them to the max_len specified
            if max_seq_len is not None:
                for i, (seq, _) in enumerate(batch):
                    batch[i][0] = torch.nn.ConstantPad1d((0, max_seq_len - seq.shape[0]), 0)(seq)
            seqs, labels = zip(*batch)
            padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
            labels = torch.tensor(labels)
            return padded, labels

    import torch.nn as nn
    import torch.nn.functional as F 
    class TestFeedForward(nn.Module):

        def __init__(self):
            super(TestFeedForward, self).__init__()
            # an affine operation: y = Wx + b
            self.fc1 = nn.Linear(max_seq_len, 128)
            self.fc2 = nn.Linear(128, 128)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return x.sum(dim=1)


    max_seq_len=15
    dataset = TestData(seq_count=100, min_seq_len=5, max_seq_len=max_seq_len)
    max_metric_value_per_batch=40
    dataloader_num_workers=2
    gradient_accumulation_steps=2
    base_batch_size=8
    model = TestFeedForward().to("cuda")
    base_lr=1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

    metric_values = [ len(s[0]) for s in dataset] # difficulty = input sequence length
    dataloader, lr_scheduler, deepspeed_io_kwargs = get_dataloader_and_lr_scheduler_for_variable_batch_size(
            dataset=dataset,
            dataset_metric_values=metric_values,
            base_batch_size=base_batch_size,
            max_metric_value_per_batch=max_metric_value_per_batch,
            dataloader_rank=0,
            dataloader_num_replicas=1,
            sample_ids=None,
            pipeline_parallelism=False,
            lr_scaling_method="linear",
            min_batch_size=1,
            max_batch_size=None,
            shuffle_metric_values=False,
            order_by_metric_value=False,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataloader_num_workers=0,
            dataloader_collate_fn=lambda b : TestData.collate_fn(b, max_seq_len=max_seq_len),
            dataloader_pin_memory=False,
            optimizer=optimizer,
            # lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
            # lr_scheduler_kwargs=dict(optimizer=optimizer, step_size=1, gamma=0.1),
            verbose=True,
    )
 
    # test with PyTorch
    dataloader_it = iter(dataloader)
    with torch.set_grad_enabled(True):
        for epoch in range(10):
            try:
                for batch_id in range(len(dataloader)//gradient_accumulation_steps):
                    for microbatch_id in range(gradient_accumulation_steps):
                        inputs, labels = next(dataloader_it)
                        inputs, labels = inputs.to("cuda"), labels.to("cuda")
                        outputs = model(inputs)
                        loss = F.mse_loss(outputs, labels)
                        loss.backward()
                        print(f"Epoch {epoch}, batch {batch_id}, microbatch {microbatch_id}, loss {loss.item()}, LRs {lr_scheduler.get_last_lr()}")
                    optimizer.step()
                    optimizer.zero_grad()  
                    lr_scheduler.step()
            except StopIteration:
                # if we run out of data, we restart from the very first batch
                dataloader_it = iter(dataloader)
                continue

    # Test with DeepSpeed
    config = {
        "train_batch_size": base_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "optimizer": { "type": "Adam", "params": { "lr": base_lr, } },
    }
    engine, optimizer, _, _ = deepspeed.initialize(config=config,
        model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    # engine.training_dataloader = dataloader
    engine.deepspeed_io(**deepspeed_io_kwargs)
    # engine.training_dataloader = engine.deepspeed_io()

    dataloader_it = iter(engine.training_dataloader)
    for epoch in range(10):
        try:
            for batch_id in range(len(engine.training_dataloader)//gradient_accumulation_steps):
                for microbatch_id in range(gradient_accumulation_steps):
                    inputs, labels = next(dataloader_it)
                    inputs, labels = inputs.to("cuda"), labels.to("cuda")
                    outputs = engine(inputs)
                    loss = F.mse_loss(outputs, labels)
                    engine.backward(loss)
                    engine.step()
                    print(f"Epoch {epoch}, batch {batch_id}, microbatch {microbatch_id}, loss {loss.item()}, LRs {lr_scheduler.get_last_lr()}")
        except StopIteration:
            # if we run out of data, we restart from the very first batch
            dataloader_it = iter(engine.training_dataloader)
            continue

            