"""Copyright The Microsoft DeepSpeed Team"""

# Usage:
# deepspeed --num_gpus 1 --model-name facebook/opt1.3b --steps 100 --offload --batch-size-per-gpu 32 --zero-stage 3
# Supports models: GPT2 (see model_config.py for all variants), facebook/opt-*

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
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.ops.adam import DeepSpeedCPUAdam

from model_configs import gpt2_configs


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

    def forward(self, outputs, batch):
        logits = outputs.logits
        labels = batch.get("labels")
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
    def __init__(self, args, model_config={}, ds_config={}):
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
        steps = self.args.steps
        size = batch_size * steps
        vocab_size = self.model_config.get("vocab_size")
        seq_length = self.model_config.get("seq_length")
        return SyntheticTokenDataset(size, vocab_size, seq_length)

    @property
    def criterion(self):
        return GPTLMLoss()


class OPTModelSetup(BenchModelSetup):
    @property
    def model(self):
        hf_config = transformers.AutoConfig.from_pretrained(self.args.model_name)
        self.model_config["vocab_size"] = hf_config.vocab_size
        return transformers.OPTForCausalLM(config=hf_config)

    @property
    def dataset(self):
        batch_size = self.ds_config.get("train_micro_batch_size_per_gpu")
        steps = self.args.steps
        size = batch_size * steps
        vocab_size = self.model_config.get("vocab_size")
        seq_length = self.model_config.get("seq_length")
        return SyntheticTokenDataset(size, vocab_size, seq_length)

    @property
    def criterion(self):
        return lambda outputs, batch: outputs.loss


def get_model_class(model_name):
    if "gpt2" in model_name.lower():
        return GPT2ModelSetup
    elif "opt" in model_name.lower():
        return OPTModelSetup
    else:
        raise NotImplementedError(f"This benchmark does not yet support {model_name}")


def get_model_config(model_name):
    if "gpt2" in model_name.lower():
        return gpt2_configs[model_name.lower()]
    if "opt" in model_name.lower():
        return {"seq_length": 1024}
    else:
        raise NotImplementedError(f"This benchmark does not yet support {model_name}")


def deepspeed_bench(model_setup):
    hf_model = model_setup.model
    dataset = model_setup.dataset
    criterion = model_setup.criterion

    optimizer = DeepSpeedCPUAdam(hf_model.parameters(), lr=1e-4)
    hf_model.gradient_checkpointing_enable()
    model, optimizer, _, _ = deepspeed.initialize(
        model=hf_model,
        model_parameters=hf_model.parameters(),
        config=model_setup.ds_config,
        optimizer=optimizer,
    )
    model.train()

    data = DataLoader(
        dataset, batch_size=model_setup.ds_config["train_micro_batch_size_per_gpu"]
    )

    times = []
    for step, batch in enumerate(data):
        time_start = time.time_ns()
        for key, val in batch.items():
            batch[key] = val.to(f"cuda:{model_setup.args.local_rank}")
        outputs = model(**batch)
        loss = criterion(outputs, batch)
        model.backward(loss)
        optimizer.step()
        time_end = time.time_ns()

        if step >= model_setup.args.warmup:
            times.append((time_end - time_start) / (10**9))

    # return throughput_calculator(
    #    model_setup.model,
    #    model_setup.ds_config["train_micro_batch_size_per_gpu"],
    #    np.sum(times),
    #    step,
    #    model_setup.model_config["seq_length"],
    #    model_setup.model_config["hidden_size"],
    #    model_setup.model_config["depth"],
    #    model_setup.model_config["vocab_size"],
    # )
    return times


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", "-m", type=str, help="hf model name")
    parser.add_argument("--steps", "-s", type=int, default=50, help="total steps")
    parser.add_argument(
        "--warmup",
        "-w",
        type=int,
        default=5,
        help="number of steps before measuring performance",
    )
    parser.add_argument(
        "--zero-stage",
        "-z",
        type=int,
        default=3,
        choices=(0, 1, 2, 3),
        help="zero stage",
    )
    parser.add_argument(
        "--offload", "-o", action="store_true", help="enable cpu offloading"
    )
    parser.add_argument(
        "--batch-size-per-gpu",
        "-b",
        type=int,
        default=16,
        help="per-gpu micro batch size",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.getenv("LOCAL_RANK", "0")),
        help="local rank",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=int(os.getenv("WORLD_SIZE", "1")),
        help="world size",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model_config = get_model_config(args.model_name)
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size_per_gpu,
        "train_batch_size": args.batch_size_per_gpu * args.world_size,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {
            "stage": args.zero_stage,
        },
        "steps_per_print": 100000000,
        "fp16": {"enabled": True, "initial_scale_power": 10},
    }
    if args.offload:
        ds_config["zero_optimization"]["offload_optimizer"] = {"device": "cpu"}
        ds_config["zero_optimization"]["offload_param"] = {"device": "cpu"}

    model_setup = get_model_class(args.model_name)(
        model_config=model_config, ds_config=ds_config, args=args
    )

    times = deepspeed_bench(model_setup)
    print(f"Average time per step: {np.mean(times)}")


if __name__ == "__main__":
    main()
