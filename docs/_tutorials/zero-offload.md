---
title: "ZeRO-Offload"
---
We recommend that you read the tutorials on [Getting Started](/getting-started/)  and [ZeRO](/zero/) before stepping through this tutorial.

ZeRO-Offload is a ZeRO optimization that offloads the optimizer memory and computation from the GPU to the host CPU. ZeRO-Offload enables large models with up to 13 billion parameters to be efficiently trained on a single GPU. In this tutorial we will leverage ZeRO-Offload to train a 10-billion parameter GPT-2 model in DeepSpeed. 

### ZeRO-Offload Overview
For large model training, optimizers such as [Adam](https://arxiv.org/abs/1412.6980), can consume a significant amount of GPU compute and memory. ZeRO-Offload reduces the GPU compute and memory requirements of such models by leveraging the host CPU compute and memory resources to execute the optimizer.  