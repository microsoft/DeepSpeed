---
title: How-to DeepSpeed Accelerator Abstraction Interface
---

DeepSpeed Accelerator Interface is introduced to allow user to run large language model seamlessly on different Deep Learning acceleration hardware seamlessly with DeepSpeed.   It provides a set of accelerator runtime and accelerator op builder interface which can be implemented for different hardware.  It also allows user to use the interface to write large language model code that does not has hardware specific code.  With DeepSpeed Accelerator Interface, user can run the same large language model on different hareware platform, without the need to rewrite model code for different hardware.  This makes running large language model on different hardware easier.

This document cover two topics related to DeepSpeed Accelerator Abstraction Interface:
1. How to write accelerator agnostic models with DeepSpeed Accelerator Abstraction Interface
2. How to make a new accelerator implementation for DeepSpeed Accelerator Abstraction Interface

# How to write accelerator agnostic models with DeepSpeed Accelerator Abstraction Interface
In this part, you will learn how to write a model that does not contain HW specific code, or how to port a model that run on a specific HW only to be device agnostic.  To do this, we first import `get_accelerator` from `deepspeed.accelerator`

```
from deepspeed.accelerator import get_accelerator
```

`get_accelerator()` is the single entrance to DeepSpeed Accelerator Abstraction Interface

<code that use accelerator functionality> get_accelerator().<interface name>(...)
For existing torch.cuda.<interface name> runtime call, we convert it like the following example:

if torch.cuda.is_available():
    ...
-->

if get_accelerator().is_available():
    ...
For CUDA specific device name such as 'cuda' or 'cuda:0', or 'cuda:1', we convert them to get_accelerator().device_name(), get_accelerator().device_name(0), and get_accelerator().device_name(1).

It is a little bit trick when we convert places where torch.cuda.current_device() are called. Current device return device index, but if we supply device index in Pytorch code where a device is needed, Pytorch will explain it as a CUDA device. To get current device that can be used as a device name, we need to call get_accelerator().current_device_name():

my_tensor = torch.empty(3, 4, device=get_accelerator().current_device_name())
Only when an integer number is expected we use get_accelerator().current_device():

idx = get_accelerator().current_device()
default_generator = get_accelerator().default_generator(idx)
Tensor operations
When we convert a torch tensor to accelerator device such as my_tensor.cuda(), we use my_tensor.to(get_accelerator().deivce_name())

When we check whether a torch tensor is on accelerator device such as my_tensor.is_cuda, we use get_accelerator().on_accelerator(my_tensor)

When pin a tensor to GPU memory such as my_tensor.pin_memory(), we use get_accelerator().pin_memory(my_tensor)

Communication backend
When a communication backend string is used, the interface get_accelerator().communication_backend_name() is used get get communication backend name. So instead of torch.distributed.init_process_group('nccl'), we use torch.distributed.init_process_group(get_accelerator().communication_backend_name())

Op builder abstraction
Op builders are abstracted through get_accelerator().create_op_builder(<op builder name>), if the op builder is implemented in the accelerator, an object of OpBuilder subclass will be returned. If the op builder is not implemented, None will be returned.

A typical implementation can be referred to from the CUDA implementation, or from an XPU implementation which will be released later. Typical call such as CPUAdamBuilder().load() can be convert to get_accelerator().create_op_builder("CPUAdamBuilder").load().
