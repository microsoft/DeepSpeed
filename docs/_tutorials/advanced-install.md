---
title: "Installation Details"
date: 2020-10-28
---

The quickist way to get started with DeepSpeed is via pip, this will install
the latest release of DeepSpeed which is not tied to specific PyTorch or CUDA
versions. By default, all of DeepSpeed's C++/CUDA ops will be built
just-in-time (JIT) using [torch's JIT C++ extension loader that relies on
ninja](https://pytorch.org/docs/stable/cpp_extension.html).

```bash
pip install deepspeed
```

After installation you can validate your install and see which ops your machine
is compatible with via the DeepSpeed environment report with `ds_report` or
`python -m deepspeed.env_report`. We've found this report useful when debugging
DeepSpeed install or compatability issues.

```bash
ds_report
```

## Install DeepSpeed from source

After cloning the DeepSpeed repo from github you can install deepspeed in the
JIT mode (described above) via `pip install .`. This install should complete
quickly since it is not compiling any C++/CUDA source files.

For installs spanning multiple nodes we find it useful to install DeepSpeed
using the
[install.sh](https://github.com/microsoft/DeepSpeed/blob/master/install.sh)
script in the repo. This will build a python wheel locally and copy it to all
the nodes listed in your hostfile (either given via --hostfile, or defaults to
/job/hostfile).

## Pre-install DeepSpeed Ops

Sometimes we have found it useful to pre-install either some or all DeepSpeed
C++/CUDA ops instead of using the JIT compiled path. In order to support
pre-installation we introduce build environment flags to turn on/off building
specific ops.

You can indicate to our installer (either install.sh or pip install) that you
want to attempt to install all of our ops by setting the `DS_BUILD_OPS`
environment variable to 1, for example: `DS_BUILD_OPS=1 pip install .`.

If you want to install only a specific op (e.g., FusedLamb) you can view the op
specific build environment variable (set as `BUILD_VAR`) in the corresponding
op builder class in the
[https://github.com/microsoft/DeepSpeed/tree/master/op_builder](op\_builder)
directory.

## Pre-compiled DeepSpeed builds from PyPI

Coming soon
