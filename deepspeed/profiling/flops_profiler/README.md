# DeepSpeed Flops Profiler

> Measures the time, number of estimated flops and parameters of each module in a PyTorch Model.

## Overview

The DeepSpeed flops profiler profiles the forward pass of a PyTorch model and prints the model graph with the measured profile attached to each module.
It shows the latency, flops, and number of parameters of the modules within the model to identify potential bottlenecks.
It also outputs the names of the top `k` modules in terms of aggregated time, flops, and number of parameters at depth `l` with `k` and `l` specified by the user.
The DeepSpeed flops profiler can be used with the DeepSpeed runtime or as a standalone package.

The output profile is computed for each batch of input and printed to the `stdout`. If multiple forward passes are specified by the user to caputre (in the case where the model have different paths or for more accurate time measurement), the average profile of the multiple batches is taken. For each module, the measured profile is annotated after the name and is listed in the order of `number of parameters, percentage of total parameters, number of multiply-accumulate operations (MACs), percentage of total MACs, latency of the module, percentage of the totatal latency, floating point operations per second (FLOPS) of the module`. Note that the number of flops is estimated as `2 * MACs` in the profiler (each MAC operation is counted as 2 floating point operations).

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
    "start_step": 5,
    "end_step": 6,
    "module_depth": -1,
    "top_modules": 3
  }
}
```

An example output of 4-layer Megatron-LM model (`hidden_size = 512, num_attention_heads = 16, batch_size = 8, seq_length = 1024`) is shown below.

```shell
DistributedDataParallel(
  38.89 M, 100.00% Params, 207.23 GMACs, 100.00% MACs, 32.86 ms, 100.00% time, 1.3e+01 TFLOPS, 1,
  (module): FP16_Module(
    38.89 M, 100.00% Params, 207.23 GMACs, 100.00% MACs, 32.84 ms, 99.95% time, 1.3e+01 TFLOPS, 1,
    (module): GPT2Model(
      38.89 M, 100.00% Params, 207.23 GMACs, 100.00% MACs, 32.81 ms, 99.84% time, 1.3e+01 TFLOPS, 1,
      (language_model): TransformerLanguageModel(
        38.89 M, 100.00% Params, 207.23 GMACs, 100.00% MACs, 4.96 ms, 15.08% time, 8.4e+01 TFLOPS, 1,
        (embedding): Embedding(
          26.28 M, 67.57% Params, 0.0 MACs, 0.00% MACs, 288.96 us, 0.88% time, 0.0 TFLOPS, 1,
          (word_embeddings): VocabParallelEmbedding(25.76 M, 66.23% Params, 0.0 MACs, 0.00% MACs, 97.04 us, 0.30% time, 0.0 TFLOPS, 1, )
          (position_embeddings): Embedding(524.29 k, 1.35% Params, 0.0 MACs, 0.00% MACs, 67.95 us, 0.21% time, 0.0 TFLOPS, 1, 1024, 512)
          (embedding_dropout): Dropout(0, 0.00% Params, 0.0 MACs, 0.00% MACs, 71.53 us, 0.22% time, 0.0 TFLOPS, 1, p=0.1, inplace=False)
        )
        (transformer): ParallelTransformer(
          12.61 M, 32.43% Params, 207.23 GMACs, 100.00% MACs, 4.65 ms, 14.14% time, 8.9e+01 TFLOPS, 1,
          (layers): ModuleList(
            12.61 M, 32.42% Params, 207.23 GMACs, 100.00% MACs, 0.0, 0.00% time, 0.0 TFLOPS, 0,
            (0): ParallelTransformerLayer(
              3.15 M, 8.11% Params, 51.81 GMACs, 25.00% MACs, 2.13 ms, 6.49% time, 4.9e+01 TFLOPS, 2,
              (input_layernorm): FusedLayerNorm(1.02 k, 0.00% Params, 0.0 MACs, 0.00% MACs, 110.63 us, 0.34% time, 0.0 TFLOPS, 2, torch.Size([512]), eps=1e-05, elementwise_affine=True)
              (attention): ParallelSelfAttention(
                1.05 M, 2.70% Params, 17.45 GMACs, 8.42% MACs, 1.11 ms, 3.37% time, 3.2e+01 TFLOPS, 2,
                (query_key_value): ColumnParallelLinear(787.97 k, 2.03% Params, 12.88 GMACs, 6.22% MACs, 202.66 us, 0.62% time, 1.3e+02 TFLOPS, 2, )
                (scale_mask_softmax): FusedScaleMaskSoftmax(0, 0.00% Params, 268.44 MMACs, 0.13% MACs, 165.94 us, 0.51% time, 3.2 TFLOPS, 2, )
                (attention_dropout): Dropout(0, 0.00% Params, 0.0 MACs, 0.00% MACs, 74.63 us, 0.23% time, 0.0 TFLOPS, 2, p=0.1, inplace=False)
                (dense): RowParallelLinear(262.66 k, 0.68% Params, 4.29 GMACs, 2.07% MACs, 146.39 us, 0.45% time, 5.9e+01 TFLOPS, 2, )
              )
              (post_attention_layernorm): FusedLayerNorm(1.02 k, 0.00% Params, 0.0 MACs, 0.00% MACs, 101.33 us, 0.31% time, 0.0 TFLOPS, 2, torch.Size([512]), eps=1e-05, elementwise_affine=True)
              (mlp): ParallelMLP(
                2.1 M, 5.40% Params, 34.36 GMACs, 16.58% MACs, 411.03 us, 1.25% time, 1.7e+02 TFLOPS, 2,
                (dense_h_to_4h): ColumnParallelLinear(1.05 M, 2.70% Params, 17.18 GMACs, 8.29% MACs, 138.28 us, 0.42% time, 2.5e+02 TFLOPS, 2, )
                (dense_4h_to_h): RowParallelLinear(1.05 M, 2.70% Params, 17.18 GMACs, 8.29% MACs, 155.21 us, 0.47% time, 2.2e+02 TFLOPS, 2, )
              )
            )
            ...
            (3): ParallelTransformerLayer(...
            )
          )
          (final_layernorm): FusedLayerNorm(1.02 k, 0.00% Params, 0.0 MACs, 0.00% MACs, 55.31 us, 0.17% time, 0.0 TFLOPS, 1, torch.Size([512]), eps=1e-05, elementwise_affine=True)
        )
      )
    )
  )
)
Top 3 modules in flops at depth 8 are {'ColumnParallelLinear': '120.26 GMACs', 'RowParallelLinear': '85.9 GMACs', 'FusedScaleMaskSoftmax': '1.07 GMACs'}
Top 3 modules in params at depth 8 are {'ColumnParallelLinear': '7.35 M', 'RowParallelLinear': '5.25 M', 'FusedScaleMaskSoftmax': '0'}
Top 3 modules in time at depth 8 are {'ColumnParallelLinear': '1.25 ms', 'RowParallelLinear': '1.18 ms', 'FusedScaleMaskSoftmax': '640.39 us'}
Batch size:                     8
Number of MACs:        207.23 GMACs
Number of parameters:           38.89 M
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
    macs, params, steps = get_model_profile(model=model, # model
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
    print("{:<30}  {:<8}".format("Number of steps profiled: ", steps))
```

An example output:

```shell
AlexNet(
  61.1 M, 100.00% Params, 183.18 GMACs, 100.00% MACs, 1.07 ms, 100.00% time, 3.4e+02 TFLOPS, 10,
  (features): Sequential(
    2.47 M, 4.04% Params, 168.17 GMACs, 91.81% MACs, 713.94 us, 67.00% time, 4.7e+02 TFLOPS, 10,
    (0): Conv2d(23.3 k, 0.04% Params, 18.04 GMACs, 9.85% MACs, 110.96 us, 10.41% time, 3.3e+02 TFLOPS, 10, 3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(0, 0.00% Params, 49.56 MMACs, 0.03% MACs, 26.2 us, 2.46% time, 3.8 TFLOPS, 10, inplace=True)
    (2): MaxPool2d(0, 0.00% Params, 49.56 MMACs, 0.03% MACs, 34.62 us, 3.25% time, 2.9 TFLOPS, 10, kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(307.39 k, 0.50% Params, 57.37 GMACs, 31.32% MACs, 88.07 us, 8.27% time, 1.3e+03 TFLOPS, 10, 64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(0, 0.00% Params, 35.83 MMACs, 0.02% MACs, 20.98 us, 1.97% time, 3.4 TFLOPS, 10, inplace=True)
    (5): MaxPool2d(0, 0.00% Params, 35.83 MMACs, 0.02% MACs, 29.64 us, 2.78% time, 2.4 TFLOPS, 10, kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(663.94 k, 1.09% Params, 28.72 GMACs, 15.68% MACs, 83.37 us, 7.82% time, 6.9e+02 TFLOPS, 10, 192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(0, 0.00% Params, 16.61 MMACs, 0.01% MACs, 20.58 us, 1.93% time, 1.6 TFLOPS, 10, inplace=True)
    (8): Conv2d(884.99 k, 1.45% Params, 38.29 GMACs, 20.90% MACs, 78.56 us, 7.37% time, 9.7e+02 TFLOPS, 10, 384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(0, 0.00% Params, 11.08 MMACs, 0.01% MACs, 19.53 us, 1.83% time, 1.1 TFLOPS, 10, inplace=True)
    (10): Conv2d(590.08 k, 0.97% Params, 25.53 GMACs, 13.94% MACs, 77.7 us, 7.29% time, 6.6e+02 TFLOPS, 10, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(0, 0.00% Params, 11.08 MMACs, 0.01% MACs, 20.65 us, 1.94% time, 1.1 TFLOPS, 10, inplace=True)
    (12): MaxPool2d(0, 0.00% Params, 11.08 MMACs, 0.01% MACs, 33.21 us, 3.12% time, 0.67 TFLOPS, 10, kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(0, 0.00% Params, 2.36 MMACs, 0.00% MACs, 30.06 us, 2.82% time, 0.16 TFLOPS, 10, output_size=(6, 6))
  (classifier): Sequential(
    58.63 M, 95.96% Params, 15.01 GMACs, 8.19% MACs, 287.25 us, 26.96% time, 1e+02 TFLOPS, 10,
    (0): Dropout(0, 0.00% Params, 0.0 MACs, 0.00% MACs, 11.71 us, 1.10% time, 0.0 TFLOPS, 10, p=0.5, inplace=False)
    (1): Linear(37.75 M, 61.79% Params, 9.66 GMACs, 5.28% MACs, 72.31 us, 6.79% time, 2.7e+02 TFLOPS, 10, in_features=9216, out_features=4096, bias=True)
    (2): ReLU(0, 0.00% Params, 1.05 MMACs, 0.00% MACs, 22.2 us, 2.08% time, 0.094 TFLOPS, 10, inplace=True)
    (3): Dropout(0, 0.00% Params, 0.0 MACs, 0.00% MACs, 9.54 us, 0.90% time, 0.0 TFLOPS, 10, p=0.5, inplace=False)
    (4): Linear(16.78 M, 27.46% Params, 4.29 GMACs, 2.34% MACs, 59.44 us, 5.58% time, 1.4e+02 TFLOPS, 10, in_features=4096, out_features=4096, bias=True)
    (5): ReLU(0, 0.00% Params, 1.05 MMACs, 0.00% MACs, 20.19 us, 1.90% time, 0.1 TFLOPS, 10, inplace=True)
    (6): Linear(4.1 M, 6.71% Params, 1.05 GMACs, 0.57% MACs, 56.31 us, 5.29% time, 3.7e+01 TFLOPS, 10, in_features=4096, out_features=1000, bias=True)
  )
)
Top 3 modules in flops at depth 2 are {'Conv2d': '167.95 GMACs', 'Linear': '15.01 GMACs', 'ReLU': '126.26 MMACs'}
Top 3 modules in params at depth 2 are {'Linear': '58.63 M', 'Conv2d': '2.47 M', 'ReLU': '0'}
Top 3 modules in time at depth 2 are {'Conv2d': '438.67 us', 'Linear': '188.06 us', 'ReLU': '150.32 us'}
Batch size:                     256
Number of MACs:                 183.18 GMACs
Number of parameters:           61.1 M
Number of steps profiled:       10
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
    macs, params, steps = get_model_profile(
        model,
        (batch_size, seq_len),
        input_constructor=partial(bert_input_constructor,
                                  tokenizer=tokenizer),
        print_profile=True,
        print_aggregated_profile=True,
    )
    print("{:<30}  {:<8}".format("Number of MACs: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
    print("{:<30}  {:<8}".format("Number of steps profiled: ", steps))
```

An example output:

```
BertForSequenceClassification(
  109.48 M, 100.00% Params, 54.36 GMACs, 100.00% MACs, 152.37 ms, 100.00% time, 0.71 TFLOPS, 10,
  (bert): BertModel(
    109.48 M, 100.00% Params, 54.36 GMACs, 100.00% MACs, 152.26 ms, 99.93% time, 0.71 TFLOPS, 10,
    (embeddings): BertEmbeddings(
      23.84 M, 21.77% Params, 0.0 MACs, 0.00% MACs, 1.25 ms, 0.82% time, 0.0 TFLOPS, 10,
      (word_embeddings): Embedding(23.44 M, 21.41% Params, 0.0 MACs, 0.00% MACs, 118.52 us, 0.08% time, 0.0 TFLOPS, 10, 30522, 768, padding_idx=0)
      (position_embeddings): Embedding(393.22 k, 0.36% Params, 0.0 MACs, 0.00% MACs, 64.61 us, 0.04% time, 0.0 TFLOPS, 10, 512, 768)
      (token_type_embeddings): Embedding(1.54 k, 0.00% Params, 0.0 MACs, 0.00% MACs, 72.6 us, 0.05% time, 0.0 TFLOPS, 10, 2, 768)
      (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0.0 MACs, 0.00% MACs, 127.82 us, 0.08% time, 0.0 TFLOPS, 10, (768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(0, 0.00% Params, 0.0 MACs, 0.00% MACs, 15.26 us, 0.01% time, 0.0 TFLOPS, 10, p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      85.05 M, 77.69% Params, 54.36 GMACs, 99.99% MACs, 150.55 ms, 98.80% time, 0.72 TFLOPS, 10,
      (layer): ModuleList(
        85.05 M, 77.69% Params, 54.36 GMACs, 99.99% MACs, 0.0, 0.00% time, 0.0 TFLOPS, 0,
        (0): BertLayer(
          7.09 M, 6.47% Params, 4.53 GMACs, 8.33% MACs, 14.83 ms, 9.73% time, 0.61 TFLOPS, 10,
          (attention): BertAttention(
            2.36 M, 2.16% Params, 1.51 GMACs, 2.78% MACs, 3.89 ms, 2.55% time, 0.78 TFLOPS, 10,
            (self): BertSelfAttention(
              1.77 M, 1.62% Params, 1.13 GMACs, 2.08% MACs, 3.09 ms, 2.03% time, 0.73 TFLOPS, 10,
              (query): Linear(590.59 k, 0.54% Params, 377.49 MMACs, 0.69% MACs, 595.4 us, 0.39% time, 1.3 TFLOPS, 10, in_features=768, out_features=768, bias=True)
              (key): Linear(590.59 k, 0.54% Params, 377.49 MMACs, 0.69% MACs, 465.18 us, 0.31% time, 1.6 TFLOPS, 10, in_features=768, out_features=768, bias=True)
              (value): Linear(590.59 k, 0.54% Params, 377.49 MMACs, 0.69% MACs, 506.5 us, 0.33% time, 1.5 TFLOPS, 10, in_features=768, out_features=768, bias=True)
              (dropout): Dropout(0, 0.00% Params, 0.0 MACs, 0.00% MACs, 17.86 us, 0.01% time, 0.0 TFLOPS, 10, p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              592.13 k, 0.54% Params, 377.49 MMACs, 0.69% MACs, 759.86 us, 0.50% time, 0.99 TFLOPS, 10,
              (dense): Linear(590.59 k, 0.54% Params, 377.49 MMACs, 0.69% MACs, 521.06 us, 0.34% time, 1.4 TFLOPS, 10, in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0.0 MACs, 0.00% MACs, 112.2 us, 0.07% time, 0.0 TFLOPS, 10, (768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(0, 0.00% Params, 0.0 MACs, 0.00% MACs, 18.02 us, 0.01% time, 0.0 TFLOPS, 10, p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            2.36 M, 2.16% Params, 1.51 GMACs, 2.78% MACs, 1.88 ms, 1.23% time, 1.6 TFLOPS, 10,
            (dense): Linear(2.36 M, 2.16% Params, 1.51 GMACs, 2.78% MACs, 1.4 ms, 0.92% time, 2.2 TFLOPS, 10, in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            2.36 M, 2.16% Params, 1.51 GMACs, 2.78% MACs, 8.89 ms, 5.83% time, 0.34 TFLOPS, 10,
            (dense): Linear(2.36 M, 2.16% Params, 1.51 GMACs, 2.78% MACs, 8.61 ms, 5.65% time, 0.35 TFLOPS, 10, in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0.0 MACs, 0.00% MACs, 123.19 us, 0.08% time, 0.0 TFLOPS, 10, (768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(0, 0.00% Params, 0.0 MACs, 0.00% MACs, 27.51 us, 0.02% time, 0.0 TFLOPS, 10, p=0.1, inplace=False)
          )
        )
        ...
        (11): BertLayer(...
        )
      )
    )
    (pooler): BertPooler(
      590.59 k, 0.54% Params, 2.95 MMACs, 0.01% MACs, 229.86 us, 0.15% time, 0.026 TFLOPS, 10,
      (dense): Linear(590.59 k, 0.54% Params, 2.95 MMACs, 0.01% MACs, 135.42 us, 0.09% time, 0.044 TFLOPS, 10, in_features=768, out_features=768, bias=True)
      (activation): Tanh(0, 0.00% Params, 0.0 MACs, 0.00% MACs, 39.65 us, 0.03% time, 0.0 TFLOPS, 10, )
    )
  )
  (dropout): Dropout(0, 0.00% Params, 0.0 MACs, 0.00% MACs, 14.09 us, 0.01% time, 0.0 TFLOPS, 10, p=0.1, inplace=False)  (classifier): Linear(1.54 k, 0.00% Params, 7.68 KMACs, 0.00% MACs, 45.01 us, 0.03% time, 0.00034 TFLOPS, 10, in_features=768, out_features=2, bias=True)
)
Top 3 modules in flops at depth 7 are {'Linear': '18.12 GMACs', 'Dropout': '0.0 MACs', 'LayerNorm': '0.0 MACs'}
Top 3 modules in params at depth 7 are {'Linear': '28.35 M', 'LayerNorm': '18.43 k', 'Dropout': '0'}
Top 3 modules in time at depth 7 are {'Linear': '25.63 ms', 'LayerNorm': '1.42 ms', 'Dropout': '447.56 us'}
Number of multiply-adds:        54.36 GMACs
Number of parameters:           109.48 M
Number of steps profiled:       10
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

start_step = 5
end_step = 6
assert (end_step > start_step), "should end profiling after start profiling"

for step, batch in enumerate(data_loader):
  # start profiling at training step "start_step"
  if step == start_step:
    prof.start_profile()

  # end profiling and print output at training step "end_step"
  if step == end_step: # if using multi nodes, check global_rank == 0 as well
    flops = prof.get_total_flops(as_string=True)
    params = prof.get_total_params(as_string=True)
    if print_profile:
        prof.print_model_profile()
    if print_aggregated_profile:
        prof.print_model_aggregated_profile(module_depth=-1, top_modules=3)
    prof.end_profile()

  # forward() method
  loss = model(batch)

  # runs backpropagation
  loss.backward()

  # weight update
  optimizer.step()

```
