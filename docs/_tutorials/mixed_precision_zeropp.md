---
title: "Mixed Precision ZeRO++"
tags: training ZeRO communication-efficiency large-model
---

Mixed Precision ZeRO++ (MixZ++) is optimization strategies based on [ZeRO](/tutorials/zero/) and [ZeRO++](/tutorials/zeropp/) to improve the efficiency and reduce memory usage for large model training and inference when users use [Low-Rank Adaptation (LoRA)]([/tutorials/zero/](https://arxiv.org/abs/2106.09685)) training. Read our [ZeRO++ blog](https://www.microsoft.com/en-us/research/blog/deepspeed-zero-a-leap-in-speed-for-llm-and-chat-model-training-with-4x-less-communication/) and [paper](https://arxiv.org/pdf/2306.10209.pdf) to learn more!

We recommend that you read the tutorials on [Getting Started](/getting-started/), [ZeRO](/tutorials/zero/)  and [Megatron-DeepSpeed](/tutorials/megatron/) before stepping through this tutorial.

## Key Designs
Mixed Precision ZeRO++ (MixZ++) inherits key designs from [ZeRO++](/tutorials/zeropp/), namely quantized weights (*qwZ*), hierarchical partitioning ZeRO (*hpZ*) but has different applicability:
 - *qwZ* applies block-based quantization on frozen weights to reduce memory usage and all-gather communication volume. Compared with ZeRO++, *qwZ* in Mixed Precision ZeRO++ keeps the frozen weights quantized so there is no quantization overhead during runtime and memory usage is reduced.
 - *hpZ* eliminates inter-node parameter all-gather communication through data remapping and recomputation. Compared with ZeRO++, *hpZ* in Mixed Precision ZeRO++ applies to both backward and generation passes.

Collectively, the optimizations bring better scalability and efficiency to LoRA training. Each of the components can be enabled independent of each other and collectively as a group.

## Enabling Mixed Precision ZeRO++

A ready to go Mixed Precision ZeRO++ example has been prepared at https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/training_scripts/llama2/run_llama2_7b_mixz.sh. If you prefer to manually enable Mixed Precision ZeRO++ in your pipeline, please refer to the instructions below.

### DeepSpeed Configuration Changes
An example snippet of deepspeed configurations with all Mixed Precision ZeRO++ optimization enabled is shown below:
```json
{
    "zero_optimization": {
        "stage": 3,
        "..."
        "zero_quantized_nontrainable_weights": true,
        "zero_hpz_partition_size": 16,
        "..."
    }
}
```
Note that the `"zero_hpz_partition_size"` should be set to the number of GPUs per node. For example, if you have 8 GPUs per node, then `"zero_hpz_partition_size"` should be set to 8.

### Training Script Changes
DeepSpeed engine will identify the LoRA frozen parameters if the LoRA model is passed when DeepSpeed initializes. However, the popular implementation is to initialize a base model and then convert to LoRA model later. In such cases, users need to explicitly call DeepSpeed engine after LoRA model is converted. This is only a 1-line effort. An example snippet of training script is shown below:

```python
model, optimizer, _, lr_scheduler = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    args=args,
    config=ds_config,
    lr_scheduler=lr_scheduler,
    dist_init_required=True)
# ...
# (the custom code to convert base model to LoRA model)
# ...
# call DeepSpeed engine again to identify LoRA frozen parameters
model.optimizer.quantize_nontrainable_params()
# ...
```

Congratulations! You have completed the Mixed Precision ZeRO++ tutorial.
