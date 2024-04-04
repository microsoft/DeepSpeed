# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

#########################################
# Timers
#########################################
''' Elasticity Utility in DeepSpeed can be used to create highly elastic jobs compatible
with a large number of GPUs. For elastic jobs, DeepSpeed will provide a batch size that
can support a large number of GPUs based on the user specified parameters
'''
TIMERS_FORMAT = '''
Timers should be enabled as:
"timers": {
  "throughput": {
    "enabled": true,
    "synchronized": true
  }
}
'''

TIMERS = "timers"
TIMERS_THROUGHPUT = "throughput"

TIMERS_THROUGHPUT_ENABLED = "enabled"
TIMERS_THROUGHPUT_ENABLED_DEFAULT = True

# Synchronizing a device is required to produce the most accurate timer measurements.
# However, this comes at the expense of performance degradation. The CPU timer provides
# sufficient accuracy in many cases.
TIMERS_THROUGHPUT_SYNCHRONIZED = "synchronized"
TIMERS_THROUGHPUT_SYNCHRONIZED_DEFAULT = True
