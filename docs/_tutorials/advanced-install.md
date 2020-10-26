---
title: "Installation Details"
date: 2020-10-28
---

# Installation

The quickist way to get started with DeepSpeed is via pip, this will install
the latest release of DeepSpeed which is not tied to specific PyTorch or CUDA
versions. By default, all of DeepSpeed's C++/CUDA ops will be built
just-in-time (JIT) using [torch's JIT C++ extension loader that relies on
ninja](https://pytorch.org/docs/stable/cpp_extension.html).

```bash
pip install deepspeed
```

After installation you can validate your install and see which ops your machine
is compatible with via the DeepSpeed environment report.

```bash
ds_report
```

## Pre-install DeepSpeed Ops

Sometimes we have found it useful to pre-install either some or call DeepSpeed
C++/CUDA ops instead of using the JIT compiled path. In order to support
pre-installation we introduce build environment flags to turn on/off building
specific ops.
