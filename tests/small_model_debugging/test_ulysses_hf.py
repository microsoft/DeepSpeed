# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed.comm as dist
import deepspeed
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import numpy as np
import transformers
#from datasets import load_dataset
from deepspeed.sequence.cross_entropy import vocab_sequence_parallel_cross_entropy

#MODEL_NAME = "meta-llama/Llama-2-7b-hf"
MODEL_NAME = "facebook/opt-125m"

DS_CONFIG = {
    'bf16': {
        'enabled': True
    },
    'optimizer': {
        'type': 'AdamW',
        'params': {
            'lr': 2e-05,
            'betas': [0.9, 0.999],
            'eps': 1e-08,
            'weight_decay': 0.0
        }
    },
    'scheduler': {
        'type': 'WarmupLR',
        'params': {
            'warmup_min_lr': 0,
            'warmup_max_lr': 2e-05,
            'warmup_num_steps': 0
        }
    },
    'zero_optimization': {
        'stage': 2,
        'overlap_comm': True,
        'contiguous_gradients': True,
        'sub_group_size': 1e9,
        'sub_group_size': 1000000000.0,
        'reduce_bucket_size': 16777216,
        'stage3_prefetch_bucket_size': 15099494.4,
        'stage3_max_live_parameters': 1000000000.0,
        'stage3_max_reuse_distance': 1000000000.0,
        'stage3_gather_16bit_weights_on_model_save': True,
        'zero_hpz_partition_size': 1
    },
    'gradient_accumulation_steps': 1,
    'gradient_clipping': 1.0,
    'steps_per_print': float('inf'),
    'train_batch_size': 8,
    'train_micro_batch_size_per_gpu': 4,
    'wall_clock_breakdown': False,
}


def load_and_prepare_data(model_name):
    """Load model, tokenizer and dataset, and prepare data loader."""
    from datasets import load_dataset
    # Load model and tokenizer
    config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.num_hidden_layers = 9
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 config=config,
                                                 trust_remote_code=True,
                                                 torch_dtype=config.torch_dtype,
                                                 attn_implementation="flash_attention_2")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load and tokenize dataset
    dataset = load_dataset("wikitext", 'wikitext-103-raw-v1', split='train[:1%]').filter(lambda x: x['text'])

    def tokenize_function(examples):
        # Tokenize and ensure 'labels' are the same as 'input_ids'
        max_length = 256
        tokenized_output = tokenizer(examples["text"],
                                     padding="longest",
                                     truncation=True,
                                     return_tensors='pt',
                                     max_length=max_length)
        tokenized_output["labels"] = tokenized_output["input_ids"].clone()
        return tokenized_output

    tokenized_dataset = dataset.map(tokenize_function, batched=True).filter(lambda x: x['text'])
    #tokenized_dataset.set_format('torch', columns=['input_ids', 'labels'])
    tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Create data loader
    data_loader = DataLoader(tokenized_dataset, batch_size=2, shuffle=False)
    #print(f"data {data_loader.dataset.__dict__}")
    return model, data_loader


def get_loss(model, data_loader, config_dict, step=10):
    """Train the model and calculate average loss."""
    # Initialize DeepSpeed
    model, _, _, _ = deepspeed.initialize(model=model,
                                          model_parameters=model.parameters(),
                                          config=config_dict,
                                          dist_init_required=True,
                                          mesh_param=(2, 4))
    spg = model.get_sequence_parallel_group()
    seq_parallel_world_size = dist.get_world_size(spg)
    seq_parallel_rank = dist.get_rank(spg)
    dist.barrier()
    model.train()

    # Training loop
    losses = []
    for n, batch in enumerate(data_loader):
        if n >= step:
            break

        seq_length = batch["input_ids"].size(1)
        assert seq_length % seq_parallel_world_size == 0
        sub_seq_length = seq_length // seq_parallel_world_size
        sub_seq_start = seq_parallel_rank * sub_seq_length
        sub_seq_end = (seq_parallel_rank + 1) * sub_seq_length

        batch["input_ids"] = batch["input_ids"][:, sub_seq_start:sub_seq_end]
        batch["labels"] = batch["labels"][:, sub_seq_start:sub_seq_end]

        batch = {k: v.to(model.device) for k, v in batch.items()}
        #if dist.get_rank() == 0:
        #print(f"Model: {model} Module: {model.module}")
        original_forward = model.forward

        def sp_forward(**batch):
            loss_mask = batch['attention_mask']
            batch['attention_mask'] = None
            #[b,s] => [s,b]
            labels = batch['labels'].transpose(0, 1).contiguous()
            outputs = original_forward(**batch)
            #loss = outputs.loss
            #labels_1, loss_mask = labels[0], labels[1]
            #print(f"LOGITS: {outputs.logits.shape} labels: {labels.shape}")
            #label , b,s,h
            loss = vocab_sequence_parallel_cross_entropy(outputs.logits.transpose(0, 1), labels, spg)
            #loss = vocab_sequence_parallel_cross_entropy(outputs.logits, labels,spg)
            #[s,b] => [b,s]
            loss = loss.transpose(0, 1).contiguous()  ##??
            #print(f"Loss: {loss.shape} mask: {loss_mask.shape}")
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss.view(-1) * loss_mask) / loss_mask.sum()
            return loss

        model.forward = sp_forward
        loss = model(**batch)
        model.forward = original_forward
        if dist.get_rank() == 0:
            print(f"loss: {loss}")
        model.backward(loss)
        model.step()
        losses.append(loss.item())

    return np.nanmean(losses[-100:])


if __name__ == "__main__":
    torch.manual_seed(0)
    model, data_loader = load_and_prepare_data(MODEL_NAME)
    zeropp_loss = get_loss(model, data_loader, DS_CONFIG)
