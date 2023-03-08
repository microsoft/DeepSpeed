import torch
import deepspeed
import transformers
import datasets
import os
import time
import argparse
import numpy as np

from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod


def get_parameters_in_billions(model):
    gpus_per_model = int(os.getenv("WORLD_SIZE", "1"))
    approx_parameters_in_billions = sum(
        [
            sum(
                [
                    p.ds_numel if hasattr(p, "ds_id") else p.nelement()
                    for p in model_module.parameters()
                ]
            )
            for model_module in model
        ]
    )
    return approx_parameters_in_billions * gpus_per_model / (1e9)


def throughput_calculator(
    model,
    micro_batch_size,
    iteration_time,
    total_iterations,
    seq_length,
    hidden_size,
    num_layers,
    vocab_size,
):
    gpus_per_model = int(os.getenv("WORLD_SIZE", "1"))
    batch_size = micro_batch_size * gpus_per_model
    samples_per_model = batch_size * seq_length
    model_replica_count = 1
    approx_parameters_in_billions = (
        None if (model is None) else get_parameters_in_billions(model)
    )
    elapsed_time_per_iter = iteration_time / total_iterations
    samples_per_second = batch_size / elapsed_time_per_iter

    # General TFLOPs formula (borrowed from Equation 3 in Section 5.1 of
    # https://arxiv.org/pdf/2104.04473.pdf).
    # The factor of 4 is when used with activation check-pointing,
    # otherwise it will be 3.
    checkpoint_activations_factor = 3
    flops_per_iteration = (
        24
        * checkpoint_activations_factor
        * batch_size
        * seq_lenth
        * num_layers
        * (hidden_size**2)
    ) * (
        1.0
        + (seq_len / (6.0 * hidden_size))
        + (vocab_size / (16.0 * num_layers * hidden_size))
    )
    tflops = flops_per_iteration / (elapsed_time_per_iter * gpus_per_model * (10**12))
    return samples_per_second, tflops, approx_parameters_in_billions


class GPTLMLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )


class SyntheticTokenDataset(Dataset):
    def __init__(self, size, vocab_size, seq_length) -> None:
        super().__init__()
        self.size = size
        self.vocab_size = vocab_size
        self.seq_length = seq_length

    def __len__(self):
        return self.size

    def __getitem__(self, _):
        input_ids = torch.randint(self.vocab_size, (self.seq_length,))
        attention_mask = torch.ones((self.seq_length,), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


class BenchModelSetup(ABC):
    def __init__(self, args, model_config, ds_config):
        self.args = args
        self.model_config = model_config
        self.ds_config = ds_config

    @property
    @abstractmethod
    def model(self):
        ...

    @property
    @abstractmethod
    def dataset(self):
        ...

    @property
    @abstractmethod
    def criterion(self):
        ...


class GPT2ModelSetup(BenchModelSetup):
    @property
    def model(self):
        hf_config = transformers.GPT2Config(**self.model_config)
        hf_model = transformers.GPT2LMHeadModel(hf_config)
        return hf_model

    @property
    def dataset(self):
        batch_size = self.ds_config.get("train_micro_batch_size_per_gpu")
        steps = self.args.steps_per_epoch
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        size = batch_size * world_size * steps
        vocab_size = self.model_config.get("vocab_size")
        seq_length = self.model_config.get("seq_length")
        return SyntheticTokenDataset(size, vocab_size, seq_length)

    @property
    def criterion(self):
        return GPTLMLoss()


model_classes = {"gpt2": GPT2ModelSetup}


def deepspeed_bench(model_setup):
    hf_model = model_setup.model
    dataset = model_setup.dataset
    criterion = model_setup.criterion

    model, _, _, _ = deepspeed.initialize(
        model=hf_model,
        model_parameters=hf_model.parameters(),
        config=model_setup.ds_config,
    )
    model.train()

    data = DataLoader(
        dataset, batch_size=model_setup.ds_config["train_micro_batch_size_per_gpu"]
    )

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    times = []
    for step, batch in enumerate(data):
        time_start = time.time_ns()
        for key, val in batch.items():
            batch[key] = val.to(f"cuda:{local_rank}")
        outputs = model(**batch)

        loss = criterion(outputs.logits, batch["labels"])
        model.backward(loss)
        model.step()
        time_end = time.time_ns()

        if step >= model_setup.args.warmup:
            times.append((time_end - time_start) / (10**9))

    return times


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, help="hf model name")
    parser.add_argument(
        "--steps-per-epoch", "-s", type=int, default=50, help="steps per epoch"
    )
    parser.add_argument("--epochs", "-e", type=int, default=10, help="number of epochs")
    parser.add_argument(
        "--warmup",
        "-w",
        type=int,
        default=5,
        help="number of steps before measuring performance",
    )
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    args = parser.parse_args()
    return args


def main():
    gpt2_config = {
        "seq_length": 1024,
        "vocab_size": 50257,
        "hidden_size": 768,
        "num_heads": 12,
        "depth": 12,
        "nume": 124439808,
        "checkpoint": False,
        "evaluation": "ppl",
        "use_cache": False,
    }
    ds_config = {
        "train_micro_batch_size_per_gpu": 16,
        "optimizer": {"type": "Adam", "params": {"lr": 1e-4}},
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
    }
    args = parse_args()

    model_setup = model_classes[args.model](
        model_config=gpt2_config, ds_config=ds_config, args=args
    )
    times = deepspeed_bench(model_setup)
    print(np.mean(times))


if __name__ == "__main__":
    main()
