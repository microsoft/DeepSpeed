---
title: DeepSpeed Accelerator SetupGuides
tags: getting-started
---

# Contents
- [Contents](#contents)
- [Introduction](#introduction)
- [Intel Architecture (IA) CPU](#intel-architecture-ia-cpu)
- [Intel XPU](#intel-xpu)

# Introduction
DeepSpeed supports different accelerators from different companies.   Setup steps to run DeepSpeed on certain accelerators might be different.  This guide allows user to lookup the accelerator family they are using and setup environment for the hardware they are using.

# Intel Architecture (IA) CPU
DeepSpeed supports CPU with Intel Architecture instruction set.  It is recommended to have the CPU support at least AVX2 instruction set and preferrably AVX512 instruction set.

DeepSpeed has been verified on the following CPU processors:
* Intel Gen 4th Xeon Processors
* Intel Gen 5th Xeon Processors

## Installation steps for Intel Architecture CPU
To install DeepSpeed on Intel Architecture CPU, use the following steps:
1. Install gcc compiler
DeepSpeed requires gcc-9 or above to build kernels on Intel Architecture CPU, install gcc-9 or above.
2. Install numactl
DeepSpeed use numactl for fine grain CPU core allocation for load-balancing, install numactl on your system.
3. Install PyTorch
`pip install torch`
4. Install DeepSpeed
`pip install deepspeed`

## How to launch DeepSpeed on Intel Architecture CPU
DeepSpeed can launch on Intel Architecture CPU with default deepspeed command.  However, for compute intensive workloads, Intel Architecture CPU works best when each worker process runs on different set of physical CPU cores, so worker process does not compete CPU cores with each other.  To bind cores to each worker (rank), use the following command line switch for better performance.
```
deepspeed --bind_cores_to_rank <deepspeed-model-script>
```
This switch would automatically detect the number of CPU NUMA node on the host, then launch as many worker as number of NUMA nodes, and each worker to cores/memory of each NUMA node.  This ensures workers does not interfere with each other and all memory allocation is local memory which improves performance.

When user wish to get more control on the number of workers and which cores can be used by the workload, user can use the following command line switches.
```
deepspeed --num_accelerators <number-of-workers> --bind_cores_to_rank --bind_core_list <comma-seperated-dash-range> <deepspeed-model-script>
```
For example:
```
deepspeed --num_accelerators 4 --bind_cores_to_rank --bind_core_list <0-27,32-59> inference.py
```
This would start 4 workers for the workload.  The core list range will be divided evenly between 4 workers, with worker 0 take 0-13, worker 1, take 14-27, worker 2 take 32-45, and worker 3 take 46-59.  Core 28-31,60-63 are left out because there might be some background process running on the system, leaving some idle cores will reduce performance jitting and straggler effect.

Launching DeepSpeed model on multiple CPU nodes is similar to other accelerators.  We need to specify `impi` as launcher and specify `--bind_cores_to_rank` for better core binding.  Also specify `slots` number according to number of CPU sockets in   host file.

```
# hostfile content should follow the format
# worker-1-hostname slots=<#sockets>
# worker-2-hostname slots=<#sockets>
# ...

deepspeed --hostfile=<hostfile> --bind_cores_to_rank --launcher impi --master_addr <master-ip> <deepspeed-model-script>
```

## Install with Intel Extension for PyTorch and oneCCL
Although not mandatory, Intel Extension for PyTorch and Intel oneCCL provide better optimizations for LLM models.  Intel oneCCL also provide optimization when running LLM model on multi-node.  To use DeepSpeed with Intel Extension for PyTorch and oneCCL, use the following steps:
1. Install Intel Extension for PyTorch.  This is suggested if you want to get better LLM inference performance on CPU.
`pip install intel-extension-for-pytorch`

The following steps are to install oneCCL binding for PyTorch.  This is suggested if you are running DeepSpeed on multiple CPU node, for better communication performance.   On single node with multiple CPU socket, these steps are not needed.
2. Install oneCCL binding for PyTorch
`python -m pip install oneccl_bind_pt -f https://developer.intel.com/ipex-whl-stable-cpu`
3. Install Intel oneCCL, this will be used to build direct oneCCL kernels (CCLBackend kernels)
```
pip install oneccl-devel
pip install impi-devel
```
Then set the environment variables for Intel oneCCL (assuming using conda environment).
```
export CPATH=${CONDA_PREFIX}/include:$CPATH
export CCL_ROOT=${CONDA_PREFIX}
export I_MPI_ROOT=${CONDA_PREFIX}
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/ccl/cpu:${CONDA_PREFIX}/lib/libfabric:${CONDA_PREFIX}/lib
```

##Optimize LLM inference with Intel Extension for PyTorch
Intel Extension for PyTorch compatible w]th DeepSpeed AutoTP tensor parallel inference.  It allows CPU inference benefit from both DeepSpeed Automatic Tensor Parallelism and LLM optimization from Intel Extension for PyTorch.  To use Intel Extension for PyTorch, after call deepspeed.init_inference, call
```
ipex_model = ipex.llm.optimize(deepspeed_model)
```
to get model optimzied by Intel Extension for PyTorch.

Refer to https://github.com/intel/intel-extension-for-pytorch/tree/main/examples/cpu/inference/python/llm for more extensive guide.

# Intel XPU
DeepSpeed XPU accelerator supports Intel® Data Center GPU Max Series.

DeepSpeed has been verified on the following GPU products:
* Intel® Data Center GPU Max 1100
* Intel® Data Center GPU Max 1550

## Installation steps for Intel XPU
To install DeepSpeed on Intel XPU, use the following steps:
1. Install oneAPI base toolkit \
The Intel® oneAPI Base Toolkit (Base Kit) is a core set of tools and libraries, including an DPC++/C++ Compiler for building Deepspeed XPU kernels like fusedAdam and CPUAdam, high performance computation libraries demanded by IPEX, etc.
For easy download, usage and more details, check [Intel oneAPI base-toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html).
2. Install PyTorch \
`pip install torch`
3. Install Intel extension for pytorch, for torch functionality and performance on Intel platform \
`pip install intel-extension-for-pytorch`
4. Install oneccl_bindings_for_pytorch, which is the default communication backend cross XPU devices \
`pip install oneccl_bind_pt`
5. Install DeepSpeed
`pip install deepspeed`

**_NOTE:_** Should keep the software stack latest for the kernels of XPU in DeepSpeed will always be compatible with the latest released oneAPI basekit and IPEX(Intel extension for pytorch). Also you can add `-f https://developer.intel.com/ipex-whl-stable-xpu` flag for better experience of pip install intel packages.

## How to use DeepSpeed on Intel XPU
DeepSpeed can launch on Intel XPU with common deepspeed command. Before that, user needs activate the oneAPI environment by: \
`source <oneAPI installed path>/setvars.sh`

To validate the XPU availability and if the XPU accelerator is correctly chosen, here is an example:
```
$ python
>>> import torch; print('torch:', torch.__version__)
torch: 2.3.0
>>> import intel_extension_for_pytorch; print('XPU available:', torch.xpu.is_available())
XPU available: True
>>> from deepspeed.accelerator import get_accelerator; print('accelerator:', get_accelerator()._name)
accelerator: xpu
```
