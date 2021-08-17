---
title: "Mixture of Experts"
---

DeepSpeed v0.5 introduces new support for training Mixture of Experts (MoE) models. MoE models are an emerging class of sparsely activated models that have sublinear compute costs with respect to their parameters. For example, the [Switch Transformer](https://arxiv.org/abs/2101.03961) consists of over 1.6 trillion parameters, while the compute required to train it is approximately equal to that of a 10 billion-parameter dense model. This increase in model size offers tremendous accuracy gains for a constant compute budget.

For more details on results and further discussion, please see our [paper](https://aka.ms/zcode_moe) and press
release: [DeepSpeed powers 8x larger MoE model training with high performance]({{ site.press_release_v5 }}).

## Getting started with a simple MoE example

**Note:** DeepSpeed MoE requires Pytorch 1.8 or above.
{: .notice--info}

As a simple starting point we will show how to apply DeepSpeed MoE to a cifar10 example. Please refer to
our [cifar10 example](https://github.com/microsoft/DeepSpeedExamples/tree/master/cifar) going forward.

If you are adding MoE to an existing model you can use the snippet below to help guide you:


### Expert group initialization

DeepSpeed MoE supports five different forms of parallelism, and it exploits both GPU and CPU memory. Its flexible design enables users to mix different types of prevalent parallelism techniques, as shown in the table below. 

| Short Name       | Flexible Parallelism Configurations | Benefit                                                                     |
| ---------------- | ------------------------------------| --------------------------------------------------------------------------- | 
| E                | Expert                              | Scales the model size by increasing the number of experts                   |
| E + D            | Expert + Data                       | Accelerates training throughput by scaling to multiple data parallel groups |
| E + Z            | Expert + ZeRO-powered data          | Partitions the nonexpert parameters to support larger base models           |
| E + D + M        | Expert + Data + Model               | Supports massive hidden sizes and even larger base models than E+Z          |
| E + D + Z        | Expert + Data + ZeRO-powered data   | Supports massive hidden sizes and even larger base models than E+Z          |
| E + Z-Off + M    | Expert + ZeRO-Offload + Model       | Leverages both GPU and CPU memory for large MoE models on limited # of GPUs |

To support different forms of parallelism, we create a notion of DeepSpeed process groups that resides in ```deepspeed.utils.groups.py```

For most cases, the model training code needs to initialize these groups by calling 
```python 
deepspeed.utils.groups.initialize(ep_size="desired expert-parallel world size")
```

The GPUs (or ranks) participating in an expert-parallel group will distribute the total number of experts specified by the model training code argument num_experts.

For example, we can set our total number of GPUs and number of GPUs in our expert-parallel world as follows. 

```python
WORLD_SIZE = 4
EP_WORLD_SIZE = 2
EXPERTS = 8
```

Given the setting above, the user code needs to initialize the groups as follows

```python
groups.initialize (ep_size=EP_WORLD_SIZE)
```
After that, the model code needs to use the deepspeed.moe.layer.MoE API as follows.

```python
self.experts = deepspeed.moe.layer.MoE(hidden_size=input_dim, expert=ExpertModule(), num_experts=EXPERTS)
```
With the above two commands, the DeepSpeed runtime will be set to train an MoE model with a total of 8 experts on 4 GPUs in 4 experts/GPU mode. We call this the E + D mode as described earlier in the table. 

For more advanced use case of the groups API including the inter-operability with Megatron style mpu object, watch this space! 


### MoE layer

The hidden_size is the input dimension of a particular layer and the output dimension is the same as that. This could lead to some changes to your model definition, especially for vision/convolutional models because the input/output dimensions don't match in certain cases. E.g. in the CIFAR-10 example, we modify the third fully connected layer to add the MoE layer. To cater for this, we need to add an additional fully-connected layer, whose input dimension is equal to the output dimension of the MoE layer.

Original model config

```python
    self.fc3 = nn.Linear(84, 10)
```

Updated with MoE Layers

```python
    self.fc3 = nn.Linear(84, 84)
    self.fc3 = deepspeed.moe.layer.MoE(hidden_size=84, expert=self.fc3, num_experts=args.num_experts, ...)
    self.fc4 = nn.Linear(84, 10)
```

```python
import deepspeed

deepspeed.utils.groups.initialize()

experts = deepspeed.moe.layer.MoE(hidden_size, expert=Expert(params..), num_experts=num_experts, k=2)
```

<!--
hidden_size (int): the hidden dimension of the model.
expert (torch.nn.Module): the torch module that defines the expert (e.g., MLP, torch.linear).
num_experts (int, optional): default=1, the total number of experts per layer.
k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.
output_dropout_prob (float, optional): default=0.5, output dropout probability.
capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.
eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.
min_capacity (int, optional): default=4, min number of tokens per expert.
noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample' or 'None'.
-->



## Advanced MoE usage

Watch this space! We plan to add more interesting and detailed examples of using DeepSpeed MoE in the coming weeks.
