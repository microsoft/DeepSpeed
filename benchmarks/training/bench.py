import torch
import deepspeed
import transformers
import datasets
import os
import time
import numpy as np

from torch.utils.data import Dataset, DataLoader

def get_parameters_in_billions(model):
    gpus_per_model = int(os.getenv("WORLD_SIZE", "1"))
    approx_parameters_in_billions = sum([sum([p.ds_numel if hasattr(p,'ds_id') else  p.nelement() for p in model_module.parameters()])
                                        for model_module in model])
    return approx_parameters_in_billions*gpus_per_model/(1e9)

def throughput_calculator(model, micro_batch_size, iteration_time, total_iterations, seq_length, hidden_size, num_layers, vocab_size):
    gpus_per_model = int(os.getenv("WORLD_SIZE", "1"))
    batch_size = micro_batch_size * gpus_per_model
    samples_per_model = batch_size * seq_length
    model_replica_count = 1
    approx_parameters_in_billions = None if (model is None) else get_parameters_in_billions(model)
    elapsed_time_per_iter = iteration_time/total_iterations
    samples_per_second = batch_size / elapsed_time_per_iter

    # General TFLOPs formula (borrowed from Equation 3 in Section 5.1 of
    # https://arxiv.org/pdf/2104.04473.pdf).
    # The factor of 4 is when used with activation check-pointing,
    # otherwise it will be 3.
    checkpoint_activations_factor = 3
    flops_per_iteration = (24 * checkpoint_activations_factor * batch_size * seq_lenth * num_layers * (hidden_size**2)) * (1. + (seq_len / (6. * hidden_size)) + (vocab_size / (16. * num_layers * hidden_size)))
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
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

model_classes = {"gpt2": transformers.GPT2LMHeadModel}

model_configs = {"gpt2": transformers.GPT2Config}

model_criterion = {"gpt2": GPTLMLoss}


class SyntheticDataset(Dataset):
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


class ModelFromHF(torch.nn.Module):
    def __init__(self, config, model_cls):
        super().__init__()
        self.module = model_cls(config)
        # if CONFIG['model'].get('checkpoint'):
        #    self.module.apply(self.set_checkpointing)

    def set_checkpointing(self, module):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = True

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        return output.logits


class DeepSpeedInit(object):
    def __init__(self, model_name):
        self.model_name = model_name

    def get_data(self, batch_size, steps, vocab_size, seq_length):
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        size = batch_size * world_size * steps
        return SyntheticDataset(size, vocab_size, seq_length)

    def get_model(self, config_kwargs, ds_config):
        model_config = model_configs[self.model_name](**config_kwargs)
        with deepspeed.zero.Init(config_dict_or_path=ds_config):
            hf_model = ModelFromHF(model_config, model_classes[self.model_name])
        model, _, _, _ = deepspeed.initialize(
            model=hf_model, model_parameters=hf_model.parameters(), config=ds_config
        )
        return model

    def get_criterion(self):
        return model_criterion[self.model_name]()


def main():
    gpt_config = {
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

    ds = DeepSpeedInit(model_name="gpt2")

    data_set = ds.get_data(batch_size=16, steps=100,
        vocab_size=gpt_config["vocab_size"],
        seq_length=gpt_config["seq_length"],
    )
    data = DataLoader(data_set, batch_size=ds_config["train_micro_batch_size_per_gpu"])

    model = ds.get_model(gpt_config, ds_config)
    model.train()

    criterion = ds.get_criterion()

    times = []
    iters = 100
    warmup = 5
    for step, batch in enumerate(data):
        if step >= iters:
            break
        time_start = time.time_ns()
        for key, val in batch.items():
            batch[key] = val.to("cuda:0")
        outputs = model(**batch)

        loss = criterion(outputs, batch["labels"])
        model.backward(loss)
        model.step()
        time_end = time.time_ns()

        if step >= warmup:
            times.append((time_end - time_start) / (10 ** 9))

    print(np.mean(times))

if __name__ == "__main__":
    main()
