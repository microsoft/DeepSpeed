# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Helpers for optional parallel process groups.

Code for an optional parallel dimension (tensor, sequence, expert, ...) commonly receives ``group=None``
when that dimension is not used. Here ``None`` therefore means *no parallelism*, unlike the collectives in
``deepspeed.comm`` where ``group=None`` means the world group. To keep the two meanings apart, these helpers
are not re-exported from ``deepspeed.comm``; import them as ``from deepspeed.comm import pg_utils``.
"""

import deepspeed.comm as dist
from deepspeed.comm.reduce_op import ReduceOp


def is_parallel(group):
    """
    Return whether ``group`` spans more than one rank, i.e. whether a collective over it has any effect.

    ``None`` marks an absent parallel dimension and returns ``False`` without touching the process group,
    so it also works before ``torch.distributed`` is initialized. It never means the world group; pass
    the world group explicitly to test it.
    """
    return group is not None and dist.get_world_size(group) > 1


def all_reduce_if_parallel(tensor, op=ReduceOp.SUM, group=None):
    """
    All-reduce ``tensor`` in place over ``group`` and return it, skipping the collective unless
    :func:`is_parallel` holds for ``group``.

    Only synchronous reductions are supported, since a skipped call has no work handle to return; guard
    asynchronous calls with :func:`is_parallel` instead.
    """
    if is_parallel(group):
        dist.all_reduce(tensor, op=op, group=group)
    return tensor
