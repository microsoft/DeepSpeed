# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed
import argparse
from deepspeed.accelerator import get_accelerator

deepspeed.runtime.utils.see_memory_usage('pre test', force=True)

model = AutoModelForCausalLM.from_pretrained('facebook/opt-1.3b').half().to(get_accelerator().device_name())
parser = argparse.ArgumentParser()
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

deepspeed.runtime.utils.see_memory_usage('post test', force=True)

m, _, _, _ = deepspeed.initialize(model=model, args=args)

m.eval()

prompt = "Microsoft is in Washington"

tokenizer = AutoTokenizer.from_pretrained('facebook/opt-1.3b', padding_side="left")

input_tokens = tokenizer(prompt, return_tensors="pt", padding=True)

for t in input_tokens:
    if torch.is_tensor(input_tokens[t]):
        input_tokens[t] = input_tokens[t].to(get_accelerator().device_name())

outputs = m.generate(**input_tokens, do_sample=False, max_length=100)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
