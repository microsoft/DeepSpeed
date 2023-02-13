---
title: "Automatic Tensor Parallelism for HuggingFace Models"
tags: inference
---

This tutorial demonstrates the new automatic tensor parallelism feature for inference. Previously, the user needed to provide an injection policy to DeepSpeed to enable tensor parallelism. DeepSpeed now supports automatic tensor parallelism for HuggingFace models by simply setting the replace method to empty string "". This is convenient for when the injection policy of a model is not known and improving performance of models without kernel injection support.

```python
# create the model
import transformers
import deepspeed
pipe = pipeline("text2text-generation", model="google/t5-v1_1-small", device=local_rank)
# Initialize the DeepSpeed-Inference engine
pipe.model = deepspeed.init_inference(
    pipe.model,
    mp_size=world_size,
    dtype=torch.float,
    replace_method=""
)
output = pipe('Input String')
```

Previously, to run inference with only tensor parallelism for the models that we don't support kernels, you can pass an injection policy that shows the two specific linear layers on a Transformer Encoder/Decoder layer: 1) the attention output GeMM and 2) layer output GeMM. We need these parts of the layer to add the required all-reduce communication between GPUs to merge the partial results across model-parallel ranks. Below, we bring an example that shows how you can use deepspeed-inference with a T5 model:

```python
# create the model
import transformers
from transformers.models.t5.modeling_t5 import T5Block
import deepspeed
pipe = pipeline("text2text-generation", model="google/t5-v1_1-small", device=local_rank)
# Initialize the DeepSpeed-Inference engine
pipe.model = deepspeed.init_inference(
    pipe.model,
    mp_size=world_size,
    dtype=torch.float,
    injection_policy={T5Block: ('SelfAttention.o', 'EncDecAttention.o', 'DenseReluDense.wo')}
)
output = pipe('Input String')
```

With automatic tensor parallelism, we do not need to provide the injection policy and can use replace method set to empty string "" instead. The injection policy will be determined at runtime.


## Example Script

We can observe performance improvement with automatic tensor parallism using the [inference test suite](https://github.com/microsoft/DeepSpeedExamples/blob/master/inference/huggingface/text-generation/inference-test.py). The script includes per token latency, bandwidth, throughput and memory checks for comparison.


## Launching

To run without DeepSpeed and without tensor parallelism:

```bash
deepspeed --num_gpus <num_gpus> DeepSpeedExamples/inference/huggingface/text-generation/inference-test.py --name <model> --batch_size <batch_size>
```


To enable tensor parallelism, you need to set the `ds_inference` to True for the compatible models:

```bash
deepspeed --num_gpus <num_gpus> DeepSpeedExamples/inference/huggingface/text-generation/inference-test.py --name <model> --batch_size <batch_size> --ds_inference
```

## OPT 13B Inference Performance Comparison

The following results were collected using V100 SXM2 32GB GPUs.

### Max New Tokens = 50
| | Memory Allocated per GPU | Max Batch Size | Max Throughput per GPU |
|---|---|---|---|
| No TP    | 23.94 GB | 64  | 18.84 TFlops |
| 2 GPU TP | 12.23 GB | 320 | 27.17 TFlops |
| 4 GPU TP | 6.36 GB  | 664 | 27.63 TFlops |

### Max New Tokens = 1024
| | Memory Allocated per GPU | Max Batch Size | Max Throughput per GPU |
|---|---|---|---|
| No TP    | 23.94 GB | 2  | 1.65 TFlops |
| 2 GPU TP | 12.23 GB | 20 | 4.61 TFlops |
| 4 GPU TP | 6.36 GB  | 56 | 4.90 TFlops |

## Supported Models

The following model families have been successfully tested with automatic tensor parallelism. Other models may work but have not been tested yet.

- albert
- bert
- bigbird_pegasus
- camembert
- deberta_v2
- electra
- ernie
- esm
- gpt2
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

## Unsupported Models

The following models are not currently supported:

- bloom
- codegen
- deberta
- flaubert
- fsmt
- led
- longformer
- xlm
- xlnet
