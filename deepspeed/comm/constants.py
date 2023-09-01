# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

NCCL_BACKEND = 'nccl'
CCL_BACKEND = 'ccl'
MPI_BACKEND = 'mpi'
GLOO_BACKEND = 'gloo'
SCCL_BACKEND = 'sccl'
HCCL_BACKEND = 'hccl'

DEFAULT_AML_MASTER_PORT = "54965"
DEFAULT_AML_NCCL_SOCKET_IFNAME = "^docker0,lo"
