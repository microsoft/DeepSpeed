---
title: "Mixture of Experts"
---

DeepSpeed v0.4.6 introduces new support for training Mixture of Experts (MoE) models.

TODO: add short description of MoE background (can copy from press release).

For more details on results and further discussion, please see our press
release: [DeepSpeed powers 8x larger MoE model training with high performance]({{ site.press_release_v3 }})

## Getting started with a simple MoE example

**Note:** DeepSpeed MoE requires Pytorch 1.8 or above.
{: .notice--info}

As a simple starting point we will show how to apply DeepSpeed MoE to a toy cifar10 example. Please refer to
our [cifar10 example](https://github.com/microsoft/DeepSpeedExamples/tree/master/cifar) going forward.

If you are adding MoE to an existing model you can use the snippet below to help guide you:

```python
import deepspeed

deepspeed.utils.groups.initialize()

experts = deepspeed.moe.layer.MoE(hidden_size, expert=Expert(params..), num_experts=num_experts, k=2)
```

### Expert group initialization

TODO: add description of why this is needed and what it does

### MoE layer

TODO: add details about input/output dimension assumptions


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
