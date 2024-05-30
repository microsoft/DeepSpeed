# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaDecoderLayer

from deepspeed.ops.adam import FusedAdam

MODEL_CONFIGS = {
    '7b': LlamaConfig(),
    '13b': LlamaConfig(hidden_size=5120, intermediate_size=13760, num_hidden_layers=40, num_attention_heads=40),
    '30b': LlamaConfig(hidden_size=6656, intermediate_size=17888, num_hidden_layers=60, num_attention_heads=52),
    '65b': LlamaConfig(hidden_size=8192, intermediate_size=22016, num_hidden_layers=80, num_attention_heads=64)
}


class LinearLayer_LoRA(torch.nn.Module):
    # an simple implementation of LoRA
    # for now only support Linear Layer
    def __init__(self, weight, lora_dim=0, lora_scaling=1, lora_droppout=0, bias=None):
        super(LinearLayer_LoRA, self).__init__()
        self.weight = weight
        self.bias = bias

        if lora_dim <= 0:
            raise ValueError("You are training to use LoRA, whose reduced dim should be larger than 1")

        try:
            # for zero stage 3
            rows, columns = weight.ds_shape
        except:
            rows, columns = weight.shape
        self.lora_right_weight = torch.nn.Parameter(torch.zeros(
            columns, lora_dim))  # apply transpose so in forward we do not need to transpose again
        self.lora_left_weight = torch.nn.Parameter(torch.zeros(lora_dim, rows))
        self.lora_scaling = lora_scaling / lora_dim

        if lora_droppout > 0:
            self.lora_dropout = torch.nn.Dropout(lora_droppout)
        else:
            self.lora_dropout = torch.nn.Identity()

        self.reset_parameters()
        # disable the original weight gradient
        self.weight.requires_grad = False
        # fuse LoRA to the original weight
        self.fuse_lora = False

    def eval(self):
        self.lora_dropout.eval()

    def train(self, mode=True):
        self.lora_dropout.train(mode)

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_left_weight)

    def forward(self, input):
        if self.fuse_lora:
            return F.linear(input, self.weight, self.bias)
        else:
            return F.linear(input, self.weight, self.bias) + (
                    self.lora_dropout(input) @ self.lora_right_weight @ self.lora_left_weight) * self.lora_scaling


def only_optimize_lora_parameters(model):
    # turn off the gradient of all the parameters except the LoRA parameters
    for name, param in model.named_parameters():
        if "lora_right_weight" in name or "lora_left_weight" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def convert_linear_layer_to_lora(model, part_module_name, lora_dim=0, lora_scaling=1, lora_droppout=0):
    from deepspeed.compression.helper import recursive_getattr, recursive_setattr

    repalce_name = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and part_module_name in name:
            repalce_name.append(name)
    for name in repalce_name:
        module = recursive_getattr(model, name)
        tmp = LinearLayer_LoRA(module.weight, lora_dim, lora_scaling, lora_droppout,
                               module.bias).to(module.weight.device).to(module.weight.dtype)
        recursive_setattr(model, name, tmp)
    return model


class LLama(pl.LightningModule):
    def __init__(self, parsed_args):
        super().__init__()
        self.args = parsed_args
        if self.args.num_of_model_param == '65b' or self.args.strategy == 'fsdp':
            self.configure_sharded_model = self._configure_sharded_model
        else:
            self._configure_sharded_model()
            print('init model in __init__')

        self.gradient_accumulation_steps = self.args.gradient_accumulation_steps

        if self.args.strategy == 'fsdp':
            self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        if self.args.strategy == 'fsdp':
            loss = self.model(**batch)['loss'] / self.gradient_accumulation_steps
            self.manual_backward(loss)

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            return loss
        else:
            output = self.model(**batch)
            return output['loss']

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        res = {
            'loss': output['loss']
        }
        return res

    def _configure_sharded_model(self):
        self.model = LlamaForCausalLM(MODEL_CONFIGS[self.args.num_of_model_param])
        self.model = convert_linear_layer_to_lora(self.model, "", 16)
        self.model = only_optimize_lora_parameters(self.model)

        if self.args.strategy == 'fsdp':
            if self.args.gradient_checkpointing.lower() == 'true':
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing, \
                    checkpoint_wrapper
                check_fn = lambda l: isinstance(l, LlamaDecoderLayer)
                apply_activation_checkpointing(self.model.model, checkpoint_wrapper_fn=checkpoint_wrapper,
                                               check_fn=check_fn)
            from torch.distributed.fsdp.wrap import wrap, always_wrap_policy
            self.model = wrap(self.model, auto_wrap_policy=always_wrap_policy)
        else:
            if self.args.gradient_checkpointing.lower() == 'true':
                self.model.model.gradient_checkpointing = True
            print(f'LLama model gradient_checkpointing: {self.model.model.gradient_checkpointing}')

    def configure_optimizers(self):
        if self.args.strategy == 'fsdp':
            optimizer = torch.optim.Adam(self.trainer.model.parameters(),
                                         lr=self.args.learning_rate,
                                         betas=(self.args.adam_beta1, self.args.adam_beta2),
                                         eps=self.args.adam_eps)
        else:
            optimizer = FusedAdam(self.parameters(),
                                  lr=self.args.learning_rate,
                                  betas=(self.args.adam_beta1, self.args.adam_beta2),
                                  eps=self.args.adam_eps)

        batch_per_epoch = self.args.len_dataset
        t_total = batch_per_epoch // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        warmup_ratio = self.args.warmup_ratio
        warmup_iters = int(t_total * warmup_ratio)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, warmup_iters, t_total * 0.3)
        self.optimizer = optimizer
        return [self.optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
