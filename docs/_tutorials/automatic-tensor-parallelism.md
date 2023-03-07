---
title: "Automatic Tensor Parallelism for HuggingFace Models"
tags: inference
---

# Contents
   * [Introduction](#introduction)
   * [Example Script](#example-script)
        * [Launching](#launching)
        * [T5 11B Inference Performance Comparison](#t5-11b-inference-performance-comparison)
   * [Supported Models](#supported-models)
   * [Unsupported Models](#unsupported-models)

# Introduction
This tutorial demonstrates the new automatic tensor parallelism feature for inference. Previously, the user needed to provide an injection policy to DeepSpeed to enable tensor parallelism. DeepSpeed now supports automatic tensor parallelism for HuggingFace models by default as long as kernel injection is not enabled and an injection policy is not provided. This allows our users to improve performance of models that are not currently supported via kernel injection, without providing the injection policy. Below is an example of the new method:

```python
# ---------------------------------------
# New automatic tensor parallelism method
# ---------------------------------------
import os
import torch
import transformers
import deepspeed
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
# create the model pipeline
pipe = transformers.pipeline(task="text2text-generation", model="google/t5-v1_1-small", device=local_rank)
# Initialize the DeepSpeed-Inference engine
pipe.model = deepspeed.init_inference(
    pipe.model,
    mp_size=world_size,
    dtype=torch.float
)
output = pipe('Input String')
```

Previously, to run inference with only tensor parallelism for the models that don't have kernel injection support, you could pass an injection policy that showed the two specific linear layers on a Transformer Encoder/Decoder layer: 1) the attention output GeMM and 2) layer output GeMM. We needed these parts of the layer to add the required all-reduce communication between GPUs to merge the partial results across model-parallel ranks. Below, we show an example of this previous method:

```python
# ----------------------------------
# Previous tensor parallelism method
# ----------------------------------
import os
import torch
import transformers
import deepspeed
from transformers.models.t5.modeling_t5 import T5Block
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
# create the model pipeline
pipe = transformers.pipeline(task="text2text-generation", model="google/t5-v1_1-small", device=local_rank)
# Initialize the DeepSpeed-Inference engine
pipe.model = deepspeed.init_inference(
    pipe.model,
    mp_size=world_size,
    dtype=torch.float,
    injection_policy={T5Block: ('SelfAttention.o', 'EncDecAttention.o', 'DenseReluDense.wo')}
)
output = pipe('Input String')
```

With automatic tensor parallelism, we do not need to provide the injection policy for supported models. The injection policy will be determined at runtime and applied automatically.


# Example Script

We can observe performance improvement with automatic tensor parallelism using the [inference test suite](https://github.com/microsoft/DeepSpeedExamples/blob/master/inference/huggingface/text-generation/inference-test.py). The script includes per token latency, bandwidth, throughput and memory checks for comparison. See the [README](https://github.com/microsoft/DeepSpeedExamples/tree/master/inference/huggingface/text-generation#deepspeed-huggingface-text-generation-examples) for more information.

The script below adapts the inference test suite for testing auto tensor parallelism with t5-11b.

```python
# ----------------------------------
# test-auto-tp.py
# ----------------------------------
from argparse import ArgumentParser
import transformers
import deepspeed
import torch
import os
import time
import math
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from deepspeed.runtime.utils import see_memory_usage

parser = ArgumentParser()
parser.add_argument("--model", required=True, type=str, help="model_id")
parser.add_argument("--ds_inference", action='store_true', help="enable ds-inference")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--dtype", default="float16", type=str, choices=["float32", "float16", "int8"], help="data-type")
parser.add_argument("--test_performance", action='store_true', help="enable latency, bandwidth, and throughout testing")
parser.add_argument("--local_rank", type=int, default=0, help="local rank")
args = parser.parse_args()

class DSPipeline():
    def __init__(self,
                 model_name='t5-11b',
                 dtype=torch.float16,
                 device=-1,
                 checkpoint_path=None
                 ):
        self.model_name = model_name
        self.dtype = dtype

        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif device < 0:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", model_max_length=512)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.eval()

    def __call__(self,
                inputs=["test"]
                ):
        outputs = self.generate_outputs(inputs)
        return outputs

    def generate_outputs(self,
                         inputs=["test"]
                        ):
        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        self.model.cuda().to(self.device)
        outputs = self.model.generate(inputs["input_ids"].to(self.device))
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        average_num_tokens = 0
        for o in outputs:
            average_num_tokens += len(self.tokenizer.tokenize(o))
        average_num_tokens = average_num_tokens/args.batch_size
        return outputs, average_num_tokens

def print_perf_stats(latency_set, config, warmup=3):
    # trim warmup queries
    latency_set = list(latency_set)
    latency_set = latency_set[warmup:]
    count = len(latency_set)

    if count > 0:
        latency_set.sort()
        avg = sum(latency_set) / count
        avg = avg/args.batch_size
        num_layers = getattr(config, "num_layers", config.num_hidden_layers)
        num_parameters = num_layers * config.hidden_size * config.hidden_size * 12
        if args.dtype == "float16":
            num_bytes = 2
        elif args.dtype == "float32":
            num_bytes = 4
        else:
            num_bytes = 1

        log = open("log.txt","a")
        log.write(str(os.getenv('WORLD_SIZE', '1')) + " gpus, " + str(args.batch_size) + " batch\n")
        log.write("Avg Per Token Latency: {0:8.2f} ms\n".format(avg * 1000))
        log.write("Avg flops: {0:8.2f} TFlops/s\n".format(1/avg * num_parameters * num_bytes / 1e12))
        log.close()

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

data_type = getattr(torch, args.dtype)

pipe = DSPipeline(model_name=args.model,
                  dtype=data_type,
                  device=args.local_rank,
                  )

if local_rank == 0:
    see_memory_usage("before init", True)

if args.ds_inference:
    pipe.model = deepspeed.init_inference(
        pipe.model,
        mp_size=world_size,
        dtype=data_type,
    )

if local_rank == 0:
    see_memory_usage("after init", True)

input_sentences = [
         "DeepSpeed is a machine learning framework",
         "summarize: My friends are cool but they eat too many carbs",
         "summarize: There are many reasons to have a dog",
         "translate English to French: He is working on it",
         "summarize: My friends are cool but they eat too many carbs.",
         "translate English to German: The house is wonderful.",
         "summarize: studies have shown that owning a dog is good for you",
         "translate English to Spanish: The new movie that got Oscar this year",
         "translate English to French: In the far far distance from our galaxy,",
         "translate English to German: Peace is the only way."
]

if args.batch_size > len(input_sentences):
    # dynamically extend to support larger bs by repetition
    input_sentences *= math.ceil(args.batch_size / len(input_sentences))

inputs = input_sentences[:args.batch_size]

iters = 30 if args.test_performance else 1
times=[]
for i in range(iters):
    torch.cuda.synchronize()
    start = time.time()
    outputs, average_num_tokens = pipe(inputs)
    torch.cuda.synchronize()
    end = time.time()
    times.append(end - start)

if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    for i, o in zip(inputs, outputs):
        print(f"\nin={i}\nout={o}\n{'-'*60}")
    if args.test_performance:
        print_perf_stats(map(lambda t: t / average_num_tokens, times), pipe.model.config)

```

## Launching

Use the following command to run without DeepSpeed and without tensor parallelism. Set the `test_performance` flag to collect performance data:

```bash
deepspeed --num_gpus <num_gpus> test-auto-tp.py --name <model> --batch_size <batch_size> --test_performance
```


To enable tensor parallelism, you need to use the flag `ds_inference`:

```bash
deepspeed --num_gpus <num_gpus> test-auto-tp.py --name <model> --batch_size <batch_size> --test_performance --ds_inference
```

## T5 11B Inference Performance Comparison

The following results were collected using V100 SXM2 32GB GPUs.

### Latency

![T5 Latency Graph](/assets/images/auto-tp-chart-latency.png){: .align-center}

### Throughput

![T5 Throughput Graph](/assets/images/auto-tp-chart-throughput.png){: .align-center}

### Memory

| Test           | Memory Allocated per GPU   |
| -------------- | -------------------------- |
| No TP or 1 GPU | 21.06 GB                   |
| 2 GPU TP       | 10.56 GB                   |
| 4 GPU TP       | 5.31 GB                    |
| 8 GPU TP       | 2.69 GB                    |

# Supported Models

The following model families have been successfully tested with automatic tensor parallelism. Other models may work but have not been tested yet.

- albert
- bert
- bigbird_pegasus
- camembert
- deberta_v2
- electra
- ernie
- esm
- gpt-j
- gpt-neo
- gpt-neox
- longt5
- luke
- m2m_100
- marian
- mvp
- nezha
- openai
- opt
- pegasus
- perceiver
- plbart
- reformer
- roberta
- roformer
- splinter
- t5
- xglm
- xlm_roberta
- yoso

# Unsupported Models

The following models are not currently supported with automatic tensor parallelism. They may still be compatible with other DeepSpeed features (e.g., kernel injection for Bloom):

- bloom
- codegen
- deberta
- flaubert
- fsmt
- gpt2
- led
- longformer
- xlm
- xlnet
