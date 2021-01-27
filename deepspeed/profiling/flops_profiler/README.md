# DeepSpeed Flops Profiler

> Measures the time, number of estimated flops and parameters of each module in a PyTorch Model.

## Overview

The DeepSpeed flops profiler profiles the forward pass of a PyTorch model and prints the model graph with the measured profile attached to each module.
It shows the latency, flops, and number of parameters of the modules within the model to identify potential bottlenecks.
It also outputs the names of the top `k` modules in terms of aggregated time, flops, and number of parameters at depth `l` with `k` and `l` specified by the user.
The DeepSpeed flops profiler can be used with the DeepSpeed runtime or as a standalone package.

The output profile is computed for each batch of input and printed to the `stdout`. For each module, the measured profile is annotated after the name and is listed in the order of `number of parameters, percentage of total parameters, number of multiply-accumulate operations (MACs), percentage of total MACs, latency of the module, percentage of the totatal latency, floating point operations per second (FLOPS) of the module`. Note that the number of flops is estimated as `2 * MACs` in the profiler (each MAC operation is counted as 2 floating point operations).

Below is an example output for LeNet5 with batch size 1024:

```shell
LeNet5(
  61.71 k, 100.00% Params, 439.55 MMACs, 100.00% MACs, 25.62 ms, 100.00% time, 0.034 TFLOPS,
  (feature_extractor): Sequential(
    50.69 k, 82.15% Params, 428.37 MMACs, 97.46% MACs, 18.41 ms, 71.85% time, 0.047 TFLOPS,
    (0): Conv2d(156, 0.25% Params, 125.24 MMACs, 28.49% MACs, 10.56 ms, 41.21% time, 0.024 TFLOPS, 1, 6, kernel_size=(5, 5), stride=(1, 1))
    (1): Tanh(0, 0.00% Params, 0.0 MACs, 0.00% MACs, 2.25 ms, 8.79% time, 0.0 TFLOPS, )
    (2): AvgPool2d(0, 0.00% Params, 4.82 MMACs, 1.10% MACs, 2.47 ms, 9.63% time, 0.0039 TFLOPS, kernel_size=2, stride=2, padding=0)
    (3): Conv2d(2.42 k, 3.92% Params, 247.4 MMACs, 56.28% MACs, 1.08 ms, 4.23% time, 0.46 TFLOPS, 6, 16, kernel_size=(5, 5), stride=(1, 1))
    (4): Tanh(0, 0.00% Params, 0.0 MACs, 0.00% MACs, 497.39 us, 1.94% time, 0.0 TFLOPS, )
    (5): AvgPool2d(0, 0.00% Params, 1.64 MMACs, 0.37% MACs, 758.24 us, 2.96% time, 0.0043 TFLOPS, kernel_size=2, stride=2, padding=0)
    (6): Conv2d(48.12 k, 77.98% Params, 49.27 MMACs, 11.21% MACs, 606.35 us, 2.37% time, 0.16 TFLOPS, 16, 120, kernel_size=(5, 5), stride=(1, 1))
    (7): Tanh(0, 0.00% Params, 0.0 MACs, 0.00% MACs, 68.86 us, 0.27% time, 0.0 TFLOPS, )
  )
  (classifier): Sequential(
    11.01 k, 17.85% Params, 11.18 MMACs, 2.54% MACs, 7.03 ms, 27.43% time, 0.0032 TFLOPS,
    (0): Linear(10.16 k, 16.47% Params, 10.32 MMACs, 2.35% MACs, 2.71 ms, 10.57% time, 0.0076 TFLOPS, in_features=120, out_features=84, bias=True)
    (1): Tanh(0, 0.00% Params, 0.0 MACs, 0.00% MACs, 78.77 us, 0.31% time, 0.0 TFLOPS, )
    (2): Linear(850, 1.38% Params, 860.16 KMACs, 0.20% MACs, 4.17 ms, 16.27% time, 0.00041 TFLOPS, in_features=84, out_features=10, bias=True)
  )
)
Top 3 modules in flops at depth 2 are {'Conv2d': '421.91 MMACs', 'Linear': '11.18 MMACs', 'AvgPool2d': '6.46 MMACs'}
Top 3 modules in params at depth 2 are {'Conv2d': '50.69 k', 'Linear': '11.01 k', 'Tanh': '0'}
Top 3 modules in time at depth 2 are {'Conv2d': '12.25 ms', 'Linear': '6.88 ms', 'AvgPool2d': '3.23 ms'}
Batch size:                     1024
Number of MACs:        439.55 MMACs
Number of parameters:           61.71 k
```

## Supported Models

The flops estimation is partly inspired by [ptflops](https://github.com/sovrasov/flops-counter.pytorch) with the major difference being that the DeepSpeed flops profiler captures ```torch.nn.functional``` invoked in a module to estimate the flops. Thus the DeepSpeed flops profiler allows for customized modules in the model, e.g., ```ParallelTransformerLayerworks, ParallelSelfAttention, RowParallelLinear, etc.``` in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM). This is in contrast to tools that profile at ```torch.nn.module``` level, such as ptflops, which require users to write customized flops calculation functions for each customized module. Finally, the DeepSpeed flops profiler also supports flops computation at module level (for RNNs).

## Multi-GPU, Multi-node Runs

For models running on multi-GPU or multi-node, only the model parallelism (e.g. ```--model-parallel-size``` in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)) affects the number of flops and parameters profiled, i.e.,
`model_parallel_size * flops = total_flops` and `model_parallel_size * parameters = total_parameters`. The number of GPUs or nodes does not affect the output profile.


## Usage With the DeepSpeed Runtime

When using DeepSpeed for model training, the flops profiler can be configured in the `deepspeed_config` file. No explict API calls are needed to use the profiler. Refer to [flops profiler](https://www.deepspeed.ai/docs/config-json/#flops-profiler) for details.


### Example: Megatron-LM

For information on running Megatron-LM with DeepSpeed, please refer to our tutorial [Megatron-LM](https://github.com/microsoft/DeepSpeedExamples/tree/master/Megatron-LM)

The flops profiler can be enabled by adding the following field to the `deepspeed_config` file.

```json
{
  "flops_profiler": {
    "enabled": false,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 3
  }
}
```

An example output of 4-layer Megatron-LM model (`hidden_size = 512, num_attention_heads = 16, batch_size = 8, seq_length = 1024`) is shown below.

```shell
DistributedDataParallel(
  38.89 M, 100.00% Params, 103.62 GMACs, 100.00% MACs, 33.31 ms, 100.00% time, 6.2 TFLOPS,
  (module): FP16_Module(
    38.89 M, 100.00% Params, 103.62 GMACs, 100.00% MACs, 33.29 ms, 99.94% time, 6.2 TFLOPS,
    (module): GPT2Model(
      38.89 M, 100.00% Params, 103.62 GMACs, 100.00% MACs, 33.25 ms, 99.80% time, 6.2 TFLOPS,
      (language_model): TransformerLanguageModel(
        38.89 M, 100.00% Params, 103.62 GMACs, 100.00% MACs, 4.73 ms, 14.21% time, 4.4e+01 TFLOPS,
        (embedding): Embedding(
          26.28 M, 67.57% Params, 0 MACs, 0.00% MACs, 307.56 us, 0.92% time, 0.0 TFLOPS,
          (word_embeddings): VocabParallelEmbedding(25.76 M, 66.23% Params, 0 MACs, 0.00% MACs, 130.41 us, 0.39% time, 0.0 TFLOPS, )
          (position_embeddings): Embedding(524.29 k, 1.35% Params, 0 MACs, 0.00% MACs, 68.19 us, 0.20% time, 0.0 TFLOPS, 1024, 512)
          (embedding_dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 56.27 us, 0.17% time, 0.0 TFLOPS, p=0.1, inplace=False)
        )
        (transformer): ParallelTransformer(
          12.61 M, 32.43% Params, 103.62 GMACs, 100.00% MACs, 4.4 ms, 13.22% time, 4.7e+01 TFLOPS,
          (layers): ModuleList(
            12.61 M, 32.42% Params, 103.62 GMACs, 100.00% MACs, 0, 0.00% time, 0.0 TFLOPS,
            (0): ParallelTransformerLayer(
              3.15 M, 8.11% Params, 25.9 GMACs, 25.00% MACs, 1.18 ms, 3.54% time, 4.4e+01 TFLOPS,
              (input_layernorm): FusedLayerNorm(1.02 k, 0.00% Params, 0 MACs, 0.00% MACs, 66.52 us, 0.20% time, 0.0 TFLOPS, torch.Size([512]), eps=1e-05, elementwise_affine=True)
              (attention): ParallelSelfAttention(
                1.05 M, 2.70% Params, 8.72 GMACs, 8.42% MACs, 650.17 us, 1.95% time, 2.7e+01 TFLOPS,
                (query_key_value): ColumnParallelLinear(787.97 k, 2.03% Params, 6.44 GMACs, 6.22% MACs, 139.24 us, 0.42% time, 9.3e+01 TFLOPS, )
                (scale_mask_softmax): FusedScaleMaskSoftmax(0, 0.00% Params, 134.22 MMACs, 0.13% MACs, 108.24 us, 0.32% time, 2.5 TFLOPS, )
                (attention_dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 39.58 us, 0.12% time, 0.0 TFLOPS, p=0.1, inplace=False)
                (dense): RowParallelLinear(262.66 k, 0.68% Params, 2.15 GMACs, 2.07% MACs, 73.19 us, 0.22% time, 5.9e+01 TFLOPS, )
              )
              (post_attention_layernorm): FusedLayerNorm(1.02 k, 0.00% Params, 0 MACs, 0.00% MACs, 54.36 us, 0.16% time, 0.0 TFLOPS, torch.Size([512]), eps=1e-05, elementwise_affine=True)
              (mlp): ParallelMLP(
                2.1 M, 5.40% Params, 17.18 GMACs, 16.58% MACs, 199.08 us, 0.60% time, 1.7e+02 TFLOPS,
                (dense_h_to_4h): ColumnParallelLinear(1.05 M, 2.70% Params, 8.59 GMACs, 8.29% MACs, 60.56 us, 0.18% time, 2.8e+02 TFLOPS, )
                (dense_4h_to_h): RowParallelLinear(1.05 M, 2.70% Params, 8.59 GMACs, 8.29% MACs, 75.34 us, 0.23% time, 2.3e+02 TFLOPS, )
              )
            )
            ...
            (3): ParallelTransformerLayer(...)
          (final_layernorm): FusedLayerNorm(1.02 k, 0.00% Params, 0 MACs, 0.00% MACs, 52.69 us, 0.16% time, 0.0 TFLOPS, torch.Size([512]), eps=1e-05, elementwise_affine=True)
        )
      )
    )
  )
)
Top 3 modules in flops at depth 8 are {'ColumnParallelLinear': '60.13 GMACs', 'RowParallelLinear': '42.95 GMACs', 'FusedScaleMaskSoftmax': '536.87 MMACs'}
Top 3 modules in params at depth 8 are {'ColumnParallelLinear': '7.35 M', 'RowParallelLinear': '5.25 M', 'FusedScaleMaskSoftmax': '0'}
Top 3 modules in time at depth 8 are {'ColumnParallelLinear': '595.81 us', 'RowParallelLinear': '529.29 us', 'FusedScaleMaskSoftmax': '334.26 us'}
Number of MACs:                 103616086016
Number of parameters:           38890496
```

##  Usage Outside the DeepSpeed Runtime

The flops profiler can be used as a standalone package outside of the DeepSpeed runtime.
One can simply install DeepSpeed and import the `flops_profiler` package to use the APIs directly.
Refer to [installation of DeepSpeed](https://www.deepspeed.ai/getting-started/#installation) for installing DeepSpeed.

### In Model Inference

To profile a trained model in inference, use the `get_model_profile` function.
Examples are given below.

#### Example: AlexNet

The following example shows how to profile AlexNet using the DeepSpeed flops profiler.

```python
import torchvision.models as models
import torch
from deepspeed.profiling.flops_profiler import get_model_profile

with torch.cuda.device(0):
    model = models.alexnet()
    batch_size = 256
    macs, params = get_model_profile(model=model, # model
                                     input_res=(batch_size, 3, 224, 224), # input shape or input to the input_constructor
                                     input_constructor=None, # if specified, a constructor taking input_res is used as input to the model
                                     print_profile=True, # prints the model graph with the measured profile attached to each module
                                     print_aggregated_profile=True, # print the aggregated profile for the top modules
                                     module_depth=-1, # depth into the nested modules with -1 being the inner most modules
                                     top_modules=3, # the number of top modules to print aggregated profile
                                     warm_up=10, # the number of warm-ups before measuring the time of each module
                                     as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                     ignore_modules=None) # the list of modules to ignore in the profiling
    print("{:<30}  {:<8}".format("Batch size: ", batch_size))
    print('{:<30}  {:<8}'.format('Number of MACs: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
```

An example output:

```shell
AlexNet(
  61.1 M, 100.00% Params, 183.18 GMACs, 100.00% MACs, 6.19 ms, 100.00% time, 5.9e+01 TFLOPS,
  (features): Sequential(
    2.47 M, 4.04% Params, 168.17 GMACs, 91.81% MACs, 4.42 ms, 71.41% time, 7.6e+01 TFLOPS,
    (0): Conv2d(23.3 k, 0.04% Params, 18.04 GMACs, 9.85% MACs, 143.53 us, 2.32% time, 2.5e+02 TFLOPS, 3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(0, 0.00% Params, 49.56 MMACs, 0.03% MACs, 55.31 us, 0.89% time, 1.8 TFLOPS, inplace=True)
    (2): MaxPool2d(0, 0.00% Params, 49.56 MMACs, 0.03% MACs, 41.01 us, 0.66% time, 2.4 TFLOPS, kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(307.39 k, 0.50% Params, 57.37 GMACs, 31.32% MACs, 1.85 ms, 29.83% time, 6.2e+01 TFLOPS, 64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(0, 0.00% Params, 35.83 MMACs, 0.02% MACs, 33.14 us, 0.54% time, 2.2 TFLOPS, inplace=True)
    (5): MaxPool2d(0, 0.00% Params, 35.83 MMACs, 0.02% MACs, 44.82 us, 0.72% time, 1.6 TFLOPS, kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(663.94 k, 1.09% Params, 28.72 GMACs, 15.68% MACs, 106.1 us, 1.71% time, 5.4e+02 TFLOPS, 192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(0, 0.00% Params, 16.61 MMACs, 0.01% MACs, 24.08 us, 0.39% time, 1.4 TFLOPS, inplace=True)
    (8): Conv2d(884.99 k, 1.45% Params, 38.29 GMACs, 20.90% MACs, 1.86 ms, 30.01% time, 4.1e+01 TFLOPS, 384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(0, 0.00% Params, 11.08 MMACs, 0.01% MACs, 36.0 us, 0.58% time, 0.62 TFLOPS, inplace=True)
    (10): Conv2d(590.08 k, 0.97% Params, 25.53 GMACs, 13.94% MACs, 101.09 us, 1.63% time, 5.1e+02 TFLOPS, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(0, 0.00% Params, 11.08 MMACs, 0.01% MACs, 27.66 us, 0.45% time, 0.8 TFLOPS, inplace=True)
    (12): MaxPool2d(0, 0.00% Params, 11.08 MMACs, 0.01% MACs, 35.76 us, 0.58% time, 0.62 TFLOPS, kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(0, 0.00% Params, 2.36 MMACs, 0.00% MACs, 34.09 us, 0.55% time, 0.14 TFLOPS, output_size=(6, 6))
  (classifier): Sequential(
    58.63 M, 95.96% Params, 15.01 GMACs, 8.19% MACs, 1.7 ms, 27.49% time, 1.8e+01 TFLOPS,
    (0): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 14.31 us, 0.23% time, 0.0 TFLOPS, p=0.5, inplace=False)
    (1): Linear(37.75 M, 61.79% Params, 9.66 GMACs, 5.28% MACs, 83.21 us, 1.34% time, 2.3e+02 TFLOPS, in_features=9216, out_features=4096, bias=True)
    (2): ReLU(0, 0.00% Params, 1.05 MMACs, 0.00% MACs, 25.03 us, 0.40% time, 0.084 TFLOPS, inplace=True)
    (3): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 10.49 us, 0.17% time, 0.0 TFLOPS, p=0.5, inplace=False)
    (4): Linear(16.78 M, 27.46% Params, 4.29 GMACs, 2.34% MACs, 67.23 us, 1.09% time, 1.3e+02 TFLOPS, in_features=4096, out_features=4096, bias=True)
    (5): ReLU(0, 0.00% Params, 1.05 MMACs, 0.00% MACs, 22.17 us, 0.36% time, 0.095 TFLOPS, inplace=True)
    (6): Linear(4.1 M, 6.71% Params, 1.05 GMACs, 0.57% MACs, 1.44 ms, 23.33% time, 1.5 TFLOPS, in_features=4096, out_features=1000, bias=True)
  )
)
Top 3 modules in flops at depth 2 are {'Conv2d': '167.95 GMACs', 'Linear': '15.01 GMACs', 'ReLU': '126.26 MMACs'}
Top 3 modules in params at depth 2 are {'Linear': '58.63 M', 'Conv2d': '2.47 M', 'ReLU': '0'}
Top 3 modules in time at depth 2 are {'Conv2d': '4.05 ms', 'Linear': '1.59 ms', 'ReLU': '223.4 us'}
Batch size:                     256
Number of MACs:                 183.18 GMACs
Number of parameters:           61.1 M
```

#### Example: Bert

```python
from functools import partial
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from deepspeed.profiling.flops_profiler import get_model_profile


def bert_input_constructor(input_shape, tokenizer):
    fake_seq = ""
    for _ in range(input_shape[1] - 2):  # ignore the two special tokens [CLS] and [SEP]
        fake_seq += tokenizer.pad_token
    input_tensor = tokenizer([fake_seq] * input_shape[0],
                       padding=True,
                       truncation=True,
                       return_tensors="pt")
    return input_tensor


with torch.cuda.device(0):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    batch_size = 5
    seq_len = 128
    macs, params = get_model_profile(
        model,
        (batch_size, seq_len),
        input_constructor=partial(bert_input_constructor,
                                  tokenizer=tokenizer),
        print_profile=True,
        print_aggregated_profile=True,
    )
    print("{:<30}  {:<8}".format("Number of MACs: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
```

An example output:

```
BertForSequenceClassification(
  109.48 M, 100.00% Params, 43.49 GMACs, 100.00% MACs, 390.23 ms, 100.00% time, 0.22 TFLOPS,
  (bert): BertModel(
    109.48 M, 100.00% Params, 43.49 GMACs, 100.00% MACs, 389.92 ms, 99.92% time, 0.22 TFLOPS,
    (embeddings): BertEmbeddings(
      23.84 M, 21.77% Params, 0 MACs, 0.00% MACs, 1.48 ms, 0.38% time, 0.0 TFLOPS,
      (word_embeddings): Embedding(23.44 M, 21.41% Params, 0 MACs, 0.00% MACs, 336.89 us, 0.09% time, 0.0 TFLOPS, 30522, 768, padding_idx=0)
      (position_embeddings): Embedding(393.22 k, 0.36% Params, 0 MACs, 0.00% MACs, 104.19 us, 0.03% time, 0.0 TFLOPS, 512, 768)
      (token_type_embeddings): Embedding(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 80.59 us, 0.02% time, 0.0 TFLOPS, 2, 768)
      (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 309.71 us, 0.08% time, 0.0 TFLOPS, (768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 22.41 us, 0.01% time, 0.0 TFLOPS, p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      85.05 M, 77.69% Params, 43.49 GMACs, 100.00% MACs, 387.94 ms, 99.41% time, 0.22 TFLOPS,
      (layer): ModuleList(
        85.05 M, 77.69% Params, 43.49 GMACs, 100.00% MACs, 0, 0.00% time, 0.0 TFLOPS,
        (0): BertLayer(
          7.09 M, 6.47% Params, 3.62 GMACs, 8.33% MACs, 31.32 ms, 8.03% time, 0.23 TFLOPS,
          (attention): BertAttention(
            2.36 M, 2.16% Params, 1.21 GMACs, 2.78% MACs, 16.05 ms, 4.11% time, 0.15 TFLOPS,
            (self): BertSelfAttention(
              1.77 M, 1.62% Params, 905.97 MMACs, 2.08% MACs, 15.24 ms, 3.91% time, 0.12 TFLOPS,
              (query): Linear(590.59 k, 0.54% Params, 301.99 MMACs, 0.69% MACs, 3.49 ms, 0.89% time, 0.17 TFLOPS, in_features=768, out_features=768, bias=True)
              (key): Linear(590.59 k, 0.54% Params, 301.99 MMACs, 0.69% MACs, 4.12 ms, 1.05% time, 0.15 TFLOPS, in_features=768, out_features=768, bias=True)
              (value): Linear(590.59 k, 0.54% Params, 301.99 MMACs, 0.69% MACs, 4.19 ms, 1.07% time, 0.14 TFLOPS, in_features=768, out_features=768, bias=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 33.86 us, 0.01% time, 0.0 TFLOPS, p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              592.13 k, 0.54% Params, 301.99 MMACs, 0.69% MACs, 772.24 us, 0.20% time, 0.78 TFLOPS,
              (dense): Linear(590.59 k, 0.54% Params, 301.99 MMACs, 0.69% MACs, 514.98 us, 0.13% time, 1.2 TFLOPS, in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 119.45 us, 0.03% time, 0.0 TFLOPS, (768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 27.42 us, 0.01% time, 0.0 TFLOPS, p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            2.36 M, 2.16% Params, 1.21 GMACs, 2.78% MACs, 9.26 ms, 2.37% time, 0.26 TFLOPS,
            (dense): Linear(2.36 M, 2.16% Params, 1.21 GMACs, 2.78% MACs, 8.47 ms, 2.17% time, 0.29 TFLOPS, in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            2.36 M, 2.16% Params, 1.21 GMACs, 2.78% MACs, 5.81 ms, 1.49% time, 0.42 TFLOPS,
            (dense): Linear(2.36 M, 2.16% Params, 1.21 GMACs, 2.78% MACs, 5.04 ms, 1.29% time, 0.48 TFLOPS, in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 342.61 us, 0.09% time, 0.0 TFLOPS, (768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 36.95 us, 0.01% time, 0.0 TFLOPS, p=0.1, inplace=False)
          )
        )
        ...
        (11): BertLayer(...)
      )
    )
    (pooler): BertPooler(
      590.59 k, 0.54% Params, 1.18 MMACs, 0.00% MACs, 264.41 us, 0.07% time, 0.0089 TFLOPS,
      (dense): Linear(590.59 k, 0.54% Params, 1.18 MMACs, 0.00% MACs, 157.59 us, 0.04% time, 0.015 TFLOPS, in_features=768, out_features=768, bias=True)
      (activation): Tanh(0, 0.00% Params, 0 MACs, 0.00% MACs, 42.44 us, 0.01% time, 0.0 TFLOPS, )
    )
  )
  (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 19.79 us, 0.01% time, 0.0 TFLOPS, p=0.1, inplace=False)
  (classifier): Linear(1.54 k, 0.00% Params, 3.07 KMACs, 0.00% MACs, 55.55 us, 0.01% time, 0.00011 TFLOPS, in_features=768, out_features=2, bias=True)
)
Top 3 modules in flops at depth 7 are {'Linear': '14.5 GMACs', 'Dropout': '0 MACs', 'LayerNorm': '0 MACs'}
Top 3 modules in params at depth 7 are {'Linear': '28.35 M', 'LayerNorm': '18.43 k', 'Dropout': '0'}
Top 3 modules in time at depth 7 are {'Linear': '149.14 ms', 'LayerNorm': '1.4 ms', 'Dropout': '641.82 us'}
Number of multiply-adds:        43.49 GMACs
Number of parameters:           109.48 M
```

### In Model Training Workflow

To profile a model in a training workflow, use the `FlopsProfiler`class.
The `FlopsProfiler`class provides the follwing methods:
  * `start_profile()` - starts profiling
  * `get_total_flops(as_string=False)` - returns the total number of MACs in the model
  * `get_total_params(as_string=False)` - returns the total number of parameters in the model
  * `print_model_profile()` - prints the model graph with the measured profile attached to each module
  * `print_model_aggregated_profile(module_depth=-1, top_modules=3)` - prints the names of the top modules in terms of aggregated time, flops, and parameters at depth `module_depth`.
  * `end_profile()` - ends profiling and cleans up. This should be invoked at the end of the profiling and after any printing method.

#### Example Training Workflow

Below is an example of this usage in a typical training workflow.

```python
from deepspeed.profiling.flops_profiler import FlopsProfiler

model = Model()
prof = FlopsProfiler(model)

profile_step = 5

for step, batch in enumerate(data_loader):
  # start profiling at training step "profile_step"
  if step == profile_step:
    prof.start_profile()

  # forward() method
  loss = model(batch)

  # end profiling and print output
  if step == profile_step: # if using multi nodes, check global_rank == 0 as well
    flops = prof.get_total_flops(as_string=True)
    params = prof.get_total_params(as_string=True)
    if print_profile:
        prof.print_model_profile()
    if print_aggregated_profile:
        prof.print_model_aggregated_profile(module_depth=-1, top_modules=3)
    prof.end_profile()

  # runs backpropagation
  loss.backward()

  # weight update
  optimizer.step()

```
