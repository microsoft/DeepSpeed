---
title: "Communication Logging"
excerpt: "Log all DeepSpeed communication calls"
tags: profiling performance-tuning
---

In this tutorial, we introduce DeepSpeed communication logging and provide examples of its usage.

  - [Overview](#overview)
  - [Usage](#usage)

## Overview

Logging communication calls is vital to ensure networking resources are fully utilized. The DeepSpeed communication logger enables the detection and logging of all communication operations launched under `deepspeed.comm`. Each communication operation can all be directly printed to the console immediately after completion (via the `verbose` config option), or a summary may be printed with a call to `deepspeed.comm.log_summary()` in the client code at the completion of training, an epoch, after N training iterations, etc.

## Usage

Communication logging in DeepSpeed is configured within the deepspeed [configuration file](/docs/config-json/#communication-logging). DeepSpeed will automatically log communication either all operations (`prof_all`), or user-specified operations (`prof_ops`).

  - [Configuration Setup](#configuration-setup)
  - [Log Summaries](#log-summaries)

### Configuration Setup

Communication logging can be configured in the DeepSpeed [configuration file](/docs/config-json/#communication-logging). Communication logging can be enabled by adding the following field to DeepSpeed's configuration json file. Refer to [Communication Logging](/docs/config-json/#communication-logging) for details.

```json
"comms_logger": {
  "enabled": true,
  "verbose": false,
  "prof_all": true,
  "debug": false
}
```

There are currently two ways to view communication log records:

1. Print all communication operations with `verbose` config option
2. (Recommended) Print log summary with `deepspeed.comm.log_summary()` function call. See [Log Summaries](#log-summaries)

The following is an example snippet of `verbose` output:

```
[2022-06-26 01:39:55,722] [INFO] [logging.py:69:log_dist] [Rank 0] rank=0 | comm op: reduce_scatter_base | time (ms): 9.46 | msg size: 678.86 MB | algbw (Gbps): 1204.52  | busbw (Gbps): 1129.23
[2022-06-26 01:39:56,470] [INFO] [logging.py:69:log_dist] [Rank 0] rank=0 | comm op: all_gather_base | time (ms): 0.11 | msg size: 6.0 MB | algbw (Gbps): 954.41  | busbw (Gbps): 894.76
[2022-06-26 01:39:56,471] [INFO] [logging.py:69:log_dist] [Rank 0] rank=0 | comm op: all_gather_base | time (ms): 0.08 | msg size: 6.0 MB | algbw (Gbps): 1293.47  | busbw (Gbps): 1212.63
```

For advanced users, the `debug` option will append the calling function of each communication operation to that operation's `log_name`. See [Log Summaries](#log-summaries) for an example of a `deepspeed.comm.log_summary()` call with `debug` enabled.


### Log Summaries

It's recommended that users add a call to `deepspeed.comm.log_summary()` at training milestones (e.g. every epoch or N iterations). This enables high-level communication logging without having to sift through logs from `verbose`.

The following is an example output of `deepspeed.comm.log_summary()` at the end of 10 iterations of Megatron-DeepSpeed with ZeRO-3:

```
Comm. Op            Message Size        Count               Total Latency(ms)   Avg Latency(ms)     tput_avg (Gbps)     busbw_avg (Gbps)
broadcast
                    0B                  2                   0.19                0.10                0.00                0.00
                    2.0 KB              146                 11.12               0.08                0.43                0.41
                    6.0 KB              24                  1.84                0.08                1.29                1.21
                    6.31 KB             1                   0.12                0.12                0.87                0.82
                    8.0 KB              24                  1.85                0.08                1.73                1.62
                    2.0 MB              24                  2.05                0.08                397.06              372.24
                    4.0 MB              1                   0.15                0.15                434.01              406.89
                    6.0 MB              24                  2.78                0.12                872.00              817.50
                    8.0 MB              48                  6.36                0.13                1020.36             956.59
                    98.25 MB            1                   8317.12             8317.12             0.20                0.19
barrier
                    0B                  3                   237.96              79.32               0.00                0.00
all_gather
                    128.0 B             146                 70.81               0.38                0.01                0.01
                    384.0 B             24                  11.45               0.37                0.02                0.02
                    512.0 B             24                  10.72               0.38                0.02                0.02
                    128.0 KB            24                  12.44               0.52                4.20                3.94
                    256.0 KB            1                   0.45                0.45                9.29                8.71
                    384.0 KB            24                  18.01               0.57                11.50               10.78
                    512.0 KB            48                  43.42               0.65                13.22               12.40
                    6.14 MB             2                   17.76               8.88                40.96               38.40
all_gather_base
                    128.0 B             1460                87.36               0.06                0.03                0.03
                    256.0 B             147                 3.18                0.02                0.67                0.63
                    384.0 B             240                 14.28               0.06                0.10                0.10
                    512.0 B             240                 14.43               0.06                0.14                0.13
                    128.0 KB            48                  3.38                0.07                29.79               27.93
                    128.12 KB           24                  0.24                0.00                471.78              442.30
                    256.0 KB            2                   0.18                0.09                45.52               42.67
                    384.38 KB           72                  3.69                0.05                353.74              331.63
                    512.0 KB            96                  5.22                0.06                412.72              386.92
                    512.12 KB           24                  1.16                0.05                275.55              258.33
                    512.5 KB            24                  1.70                0.07                120.35              112.83
                    6.0 MB              219                 16.63               0.07                1401.24             1313.66
                    6.0 MB              6                   0.51                0.08                1234.17             1157.03
                    6.14 MB             11                  1.00                0.08                1308.19             1226.43
                    6.25 MB             9                   0.31                0.03                15263.03            14309.09
reduce_scatter_base
                    678.86 MB           40                  602.29              9.69                1468.06             1376.31
all_reduce
                    1.0 B               20                  5572.57             6.37                0.00                0.00
                    8.0 B               40                  100.00              0.58                0.00                0.00
log_summary_barrier
                    0B                  1                   0.11                0.11                0.00                0.00
```


And the following is a call to `deepspeed.comm.log_summary` under the same configuration with `debug` enabled:

```
Comm. Op            Message Size        Count               Total Latency(ms)   Avg Latency(ms)     tput_avg (Gbps)     busbw_avg (Gbps)
broadcast | [Caller Func: _broadcast_model]
                    2.0 KB              146                 9.39                0.06                0.52                0.48
                    6.0 KB              24                  1.55                0.06                1.53                1.43
                    8.0 KB              24                  1.55                0.06                2.04                1.91
                    2.0 MB              24                  1.79                0.07                451.85              423.61
                    4.0 MB              1                   0.15                0.15                455.11              426.67
                    6.0 MB              24                  2.53                0.11                955.90              896.16
                    8.0 MB              48                  5.82                0.12                1108.38             1039.11
                    98.25 MB            1                   8540.60             8540.60             0.19                0.18
barrier | [Caller Func: _create_fp16_partitions_with_defragmentation]
                    0B                  1                   0.26                0.26                0.00                0.00
barrier | [Caller Func: _setup_for_real_optimizer]
                    0B                  2                   224.58              112.29              0.00                0.00
all_gather | [Caller Func: _allgather_params]
                    128.0 B             146                 80.27               0.45                0.00                0.00
                    384.0 B             24                  11.94               0.44                0.01                0.01
                    512.0 B             24                  11.67               0.43                0.02                0.02
                    128.0 KB            24                  13.97               0.39                5.42                5.08
                    256.0 KB            1                   1.90                1.90                2.21                2.07
                    384.0 KB            24                  11.80               0.42                14.93               13.99
                    512.0 KB            48                  28.41               0.49                17.25               16.18
                    6.14 MB             2                   18.58               9.29                20.79               19.49
all_gather_base | [Caller Func: allgather_fn]
                    256.0 B             147                 2.28                0.01                0.80                0.75
                    128.0 KB            48                  3.23                0.07                28.47               26.69
                    128.12 KB           24                  0.53                0.02                364.75              341.96
                    256.0 KB            2                   0.16                0.08                52.83               49.52
                    384.38 KB           72                  3.25                0.05                501.23              469.90
                    512.0 KB            96                  5.76                0.06                332.60              311.81
                    512.12 KB           24                  1.19                0.05                543.23              509.28
                    512.5 KB            24                  1.74                0.07                117.44              110.10
                    6.0 MB              449                 34.53               0.07                1390.20             1303.31
                    6.0 MB              6                   0.50                0.08                1230.25             1153.36
                    6.14 MB             21                  1.85                0.08                1255.51             1177.04
                    6.25 MB             19                  0.32                0.01                23649.18            22171.11
broadcast | [Caller Func: get_lst_from_rank0]
                    0B                  2                   0.16                0.08                0.00                0.00
                    6.31 KB             1                   0.13                0.13                0.79                0.74
reduce_scatter_base | [Caller Func: reduce_scatter_fn]
                    678.86 MB           80                  1527.17             13.94               1211.75             1136.01
all_reduce | [Caller Func: has_overflow]
                    1.0 B               20                  156.43              6.76                0.00                0.00
all_reduce | [Caller Func: _model_parallel_all_reduce]
                    1.0 B               20                  5895.90             0.08                0.00                0.00
                    8.0 B               40                  2.59                0.06                0.00                0.00
all_reduce | [Caller Func: get_grad_norm_direct]
                    8.0 B               40                  406.06              1.08                0.00                0.00
all_gather_base | [Caller Func: _allgather_params_coalesced]
                    128.0 B             2920                150.09              0.05                0.04                0.04
                    384.0 B             480                 24.51               0.05                0.12                0.12
                    512.0 B             480                 24.54               0.05                0.16                0.15
log_summary_barrier | [Caller Func: log_summary]
                    0B                  3                   1.96                0.65                0.00                0.00
```

The steps to add DeepSpeed communication log summaries are as follows:

1. Modify configuration file with desired settings
2. (Optional) Import `deepspeed.comm` package and modify `torch.distributed` calls to use `deepspeed.comm` if you wish to log them (Note: The `deepspeed.comm` API exactly matches `torch.distributed`)
3. Call `deepspeed.comm.log_summary`

For example usage, see the following modified [DeepSpeedExamples/cifar](https://github.com/microsoft/DeepSpeedExamples/tree/master/cifar) example:

```python
# Step 2: (Optional) Import deepspeed.comm
import deepspeed.comm as dist

# Note that any communication operations using `import torch.distributed as dist` calls can remain unchanged, and will be automatically logged under deepspeed.comm!
dist.all_reduce(tensor)

for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(trainloader):
        pre = time.time()
        inputs, labels = data[0].to(model_engine.local_rank), data[1].to(
            model_engine.local_rank)
        if fp16:
            inputs = inputs.half()
        outputs = model_engine(inputs)
        loss = criterion(outputs, labels)

        model_engine.backward(loss)
        model_engine.step()
        post = time.time()
    # Step 3: Call `deepspeed.comm.log_summary()`
    dist.log_summary()
```
