---
title: "flops-profiler: Measures the time, flops and parameters of a PyTorch Model"
---

In this tutorial, we introduce the flops-profiler in DeepSpeed and provide examples on how to use it. The flops-profiler profiles the forward pass of a PyTorch model and prints the model graph with the measured profile attached to each module. It shows how time, flops and parameters are spent in the model and which modules or layers could be the bottleneck. It also outputs the names of the top `k` modules in terms of aggregated time, flops, and parameters at depth `l` with `k` and `l` specified by the user. The output profile is computed for each batch of input. If multiple forward passes are specified by the user to caputre (in the case where the model have different paths or for more accurate timing), the average profile of the multiple batches is taken.

## 1. Overview

The flops estimation is partly inspired by [ptflops](https://github.com/sovrasov/flops-counter.pytorch) with the major difference being that flops-profiler captures ```torch.nn.functional``` invoked in a module to estimate the flops, thus allowing customized modules in the model (e.g. ```ParallelTransformerLayerworks, ParallelSelfAttention, RowParallelLinear, etc.``` in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)). The flops-profiler also supports flops computation at module level (for RNNs).

For models running on multi-node or multi-gpu, only the model parallelism affects the number of flops and parameters (e.g. ```--model-parallel-size``` in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)), i.e., model_parallel_size * flops = total_flops, model_parallel_size * parameters = total_parameters. The number of gpus or nodes does not affect the output profile.

Below is an example output for LeNet5 with batch size 1024 on a V100 GPU:
<!-- ![](header.png) -->

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

## 2. Installation

The profiler is an integral part of DeepSpeed and can be installed by

```
pip install deepspeed
```

Refer to the [installaiton of DeepSpeed](https://www.deepspeed.ai/getting-started/#installation) for more information.

## 3. Usage

### With the DeepSpeed runtime

If using DeepSpeed for model training, no explict API calls are needed to use the flops-profiler.

In DeepSpeed config file, specify:
* ```"flops_profiler": true``` to enable the flops-profiler.
* ```"profile_start_step": 5``` to start the profiler at step 5. Note that warm-up is necessary for getting accurate timing information.
* ```"profile_end_step": 6``` to end the profiler at step 6. Note that ```profile_end_step > profile_start_step```.
* ```"profile_depth": -1``` to print aggregated module information at the maximum depth (innermost modules). Can be set to any positive number, caped by the maximum depth of the model.
* ```"profile_top_num": 3```to set the number of top modules to print aggregated profile


###  Without the DeepSpeed runtime

The flops-profiler can be used as a standalone package outside of the deepspeed runtime.
#### Use the high level-API and run the model inference for profiling purpose

```python
import torchvision.models as models
import torch
from deepspeed.profiling.flops_profiler import get_model_profile

with torch.cuda.device(0):
    mod = models.alexnet()
    batch_size = 256
    macs, params, steps = get_model_profile(mod, # model
                                     (batch_size, 3, 224, 224), # input shape or input to the input_constructor
                                     input_constructor=None, # if specified, a constructor taking the the parameter before is used as input to the model
                                     print_profile=True, # print model graph with the profile annotated
                                     print_aggregated_profile=True, # print aggregated profile for top modules
                                     depth=-1, # depth into the nested modules with -1 being the inner most modules
                                     top_num=3, # the number of top modules to print aggregated profile
                                     warm_up=10, # the number of warm-ups before measuring the time of each module
                                     as_strings=True, # print raw numbers (e.g. 1000) or strings (e.g. 1k)
                                     ignore_modules=None) # the list of modules to ignore in the profiling
    print("{:<30}  {:<8}".format("Batch size: ", batch_size))
    print('{:<30}  {:<8}'.format('Number of multiply-adds: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print("{:<30}  {:<8}".format("Number of steps profiled: ", steps))
```

Examples of this usage is given in [examples](examples).

#### Use the low-level APIs to profile the forward pass in the existing model training workflow

```add_profile_methods```: adds the following methods to the model object:
  * ```start_profile``` - starts profiling
  * ```compute_total_flops``` - returns the total number of flops
  * ```compute_total_duration``` - returns the total duration
  * ```compute_total_params``` - returns the total number of params
  * ```compute_total_steps``` - returns the total number of steps (or input batches) profiled.
  * ```print_model_profile``` - prints the profile annotated
  * ```print_model_aggregated_profile``` - prints the aggregated profile for the top modules
  * ```end_profile``` - ends profiling and cleans up, invoked at the end of the profiling and before any printing method.

```flops_to_string```,  ```params_to_string```, ```duration_to_string``` are utility functions to convert the metric number to string.

Below is an example of this usage in a typical training workflow.

```python
import deepspeed.profiling.flops_profiler as prof

model = Model()
model = prof.add_profile_methods(model)

profile_start_setp = 5
profile_end_step = 10
assert (profile_end_step > profile_start_step), "should end profiling after start profiling"
print_profile = True
pring_aggregated_profile = True

for step, batch in enumerate(data_loader):
  # start profiling at training step "profile_step"
  if step == profile_start_step:
    model.start_profile()

  # end profiling and print output at training step "profile_step"
  if model == profile_end_step: # if using multi nodes, check global_rank == 0 as well
    flops = model.get_total_flops()
    params = model.get_total_params()
    if print_profile:
        model.print_model_profile()
    if print_aggregated_profile:
        model.print_model_aggregated_profile(depth=-1, top_num=3)
    model.end_profile()

  # forward() method
  loss = model(batch)

  # runs backpropagation
  loss.backward()

  # weight update
  optimizer.step()

```
## 4. Examples

### 4.1 Megatron-LM

If you don't already have a copy of the DeepSpeed repository, please clone in
now and checkout the DeepSpeedExamples submodule that contains the Megatron-LM example.

```shell
git clone https://github.com/microsoft/DeepSpeed
cd DeepSpeed
git submodule update --init --recursive
cd DeepSpeedExamples/
```

For more information on running Megatron-LM with DeepSpeed, please refer to our tutorial [Megatron-LM](https://github.com/microsoft/DeepSpeedExamples/tree/master/Megatron-LM)

The flops-profiler can be enabled using the json config file as shown below.

```json
{
  "flops_profiler": {
    "enabled": false,
    "start_step": 2,
    "end_step": 3,
    "module_depth": -1,
    "top_modules": 3
  },
}

``
### 3. Expected Results
