---
title: "Flops Profiler"
excerpt: "Measure the time, flops and parameters of your model"
---

In this tutorial, we introduce the flops profiler in DeepSpeed and provide examples on how to use it.

## Overview

The flops profiler profiles the forward pass of a PyTorch model and prints the model graph with the measured profile attached to each module. It shows how time, flops and parameters are spent in the model and which modules or layers could be the bottleneck. It also outputs the names of the top `k` modules in terms of aggregated time, flops, and parameters at depth `l` with `k` and `l` specified by the user. 
The flops profiler is an integral part of DeepSpeed and can be used within the DeepSpeed runtime or as a standalone package. 

The output profile is computed for each batch of input and printed to the stdout. If multiple forward passes are specified by the user to caputre (in the case where the model have different paths or for more accurate timing), the average profile of the multiple batches is taken. For each module, the measured profile is annotated after the name and is listed in the order of `number of parameters, percentage of total parameters, number of multiply-accumulate operations (MACs), percentage of total MACs, latency of the module, percentage of the totatal latency, floating point operations per second (FLOPS)`. Note that each MAC operation is counted as 2 floating point operations.

Below is an example output for LeNet5 with batch size 1024 on a V100 GPU:

```
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
Number of multiply-adds:        439.55 MMACs
Number of parameters:           61.71 k
Number of steps profiled:       10
```

## Supported Models

The flops estimation is partly inspired by [ptflops](https://github.com/sovrasov/flops-counter.pytorch) with the major difference being that the DeepSpeed flops profiler captures ```torch.nn.functional``` invoked in a module to estimate the flops. Thus the DeepSpeed flops profiler allows for customized modules in the model, e.g., ```ParallelTransformerLayerworks, ParallelSelfAttention, RowParallelLinear, etc.``` in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM). If using tools that profile at ```torch.nn.module``` level, such as ptflops,  The flops profiler also supports flops computation at module level (for RNNs).

## Multi-node, Multi-GPU Runs

For models running on multi-node or multi-gpu, only the model parallelism affects the number of flops and parameters (e.g. ```--model-parallel-size``` in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)), i.e., 
`model_parallel_size * flops = total_flops` and `model_parallel_size * parameters = total_parameters`. The number of gpus or nodes does not affect the output profile.


## Usage With the DeepSpeed Runtime

When using DeepSpeed for model training, the flops profiler can be configured in the `deepspeed_config` file. No explict API calls are needed to use the profiler. Refer to [flops profiler](https://www.deepspeed.ai/docs/config-json/#flops-profiler) for details.
 
```json
{
  "flops_profiler": {
    "enabled": false, # whether to enable the flops profile
    "start_step": 5,
    "end_step": 6,
    "module_depth": -1,
    "top_modules": 3
  }
}

```

### Example: Megatron-LM 

For information on running Megatron-LM with DeepSpeed, please refer to our tutorial [Megatron-LM](https://github.com/microsoft/DeepSpeedExamples/tree/master/Megatron-LM)

The flops profiler can be enabled by adding the following field to `deepspeed_config.json` file.

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
Number of multiply-adds:        207232172032.0
Number of parameters:           38890496
Number of steps profiled:       1       
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
            (1): ParallelTransformerLayer(...
            )
            (2): ParallelTransformerLayer(...
            )
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
```

##  Usage Without the DeepSpeed Runtime

The flops profiler can be used as a standalone package outside of the DeepSpeed runtime. 
One can simply install DeepSpeed and import the `flops_profiler` package to use the APIs directly.
Refer to [installaiton of DeepSpeed](https://www.deepspeed.ai/getting-started/#installation) for installing DeepSpeed.

### In Model Inference

To profile a trained model in inference, use `get_model_profile`.

#### Example: AlexNet

An example of AlexNet is given below.

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
                                     print_aggregated_profile=True, # print the aggregated profile for top modules
                                     module_depth=-1, # depth into the nested modules with -1 being the inner most modules
                                     top_modules=3, # the number of top modules to print aggregated profile
                                     warm_up=10, # the number of warm-ups before measuring the time of each module
                                     as_strings=True, # print raw numbers (e.g. 1000) or strings (e.g. 1k)
                                     ignore_modules=None) # the list of modules to ignore in the profiling
    print("{:<30}  {:<8}".format("Batch size: ", batch_size))
    print('{:<30}  {:<8}'.format('Number of MACs: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print("{:<30}  {:<8}".format("Number of steps profiled: ", steps))
```

### In Model Training Workflow

To profile a model in a training workflow, use ```FlopsProfiler()``` and its methods:
  * ```start_profile()``` - starts profiling
  * ```get_total_flops(in_str=False)``` - returns the total number of MACs in the model
  * ```get_total_params((in_str=False)``` - returns the total number of parameters in the model
  * ```print_model_profile()``` - prints the model graph with the measured profile attached to each module
  * ```print_model_aggregated_profile(module_depth=-1, top_modules=3)``` - Prints the names of the top `top_modules` modules in terms of aggregated time, flops, and parameters at depth `module_depth`.
  * ```end_profile()``` - ends profiling and cleans up, invoked at the end of the profiling and after any printing method.

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
    flops = prof.get_total_flops(in_str=True)
    params = prof.get_total_params(in_str=True)
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
