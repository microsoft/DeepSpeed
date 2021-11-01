"""
    Copyright 2021 The Microsoft DeepSpeed Team

    DeepSpeed Communication Package: deepspeed.comm

    deepspeed.comm
        -- import and use deepspeeed.ops.comm
        -- use torch.distributed directly if both this package and torch.distributed use the same NCCL version
        -- use custom collectives
            -- can either use torch.dist or ds.ops.comm?

        Note: the old 1-bit compressed allreduce variants that resided in deepspeed.runtime.comm will be moved here as well.

    deepspeed.comm API
        -- must be kept fully compatible (same signatures) as torch.dist API to ensure backward/cross-framework compatibility.
        -- e.g. if a client code used
            import deepspeed.comm as dist

            instead of
            import torch.distributed as dist

            The code should work without breaking any of the public torch.distributed functionality

    Future:
        -- deepspeed groups API should be brought into ds.comm
"""

import os
import sys
from enum import Enum
import torch

from deepspeed.comm.backend import Backend
from deepspeed.comm.nccl import NcclBackend
from deepspeed.comm.torch import TorchBackend

from deepspeed.utils import logger, log_dist
from datetime import timedelta

# Current deepspeed.comm backend (cdb) global object for simple access by client code
use_ds_backend = False
cdb = None

# Maintain objects of all initialized ds backends and assign them using the API functions in this file
nccl_backend = None
mpi_backend = None

# This should be set here so all rank/size information from the launcher can be propagated
from deepspeed.comm.utils import *

ds_world_rank = get_world_rank_from_launcher()
ds_world_size = get_world_size_from_launcher()

# For compatibility with torch distributed's init_process_group, we shall retain the signature from PyTorch code.
# DeepSpeed NCCL/MPI backend may not need all these params as we will have our own implementation.
# Please read full torch.distributed API docs from https://pytorch.org/docs/stable/distributed.html


def init_process_group(backend,
                       use_deepspeed=False,
                       init_method=None,
                       timeout=timedelta(0,
                                         1800),
                       world_size=-1,
                       rank=-1,
                       store=None,
                       group_name='',
                       pg_options=None):
    '''
        Initialize a default communication backend in deepspeed and create the world group

        Args:
            backend (str, required): The backend to use. Currently supported options: 'nccl', 'gloo', 'mpi'

            backend (str): The backend to use. Depending on
            build-time configurations, valid values include ``mpi``, ``gloo``,
            and ``nccl``. This field should be given as a lowercase string
            (e.g., ``"gloo"``). If using multiple processes per machine with ``nccl``
            backend, each process must have exclusive access to every GPU it uses, as
            sharing GPUs between processes can result in deadlocks.

            use_deepspeed (boolean, optional): Set to true to use the DeepSpeed communication backend. The param
            'backend' will be used as the deepspeed.comm.* internal backend. Special case for 'gloo' as we do not plan
            to write a custom implementation. We will just use gloo via torch.distributed backend only.

            init_method (str, optional): URL specifying how to initialize the
                                            process group. Default is "env://" if no
                                            ``init_method`` or ``store`` is specified.
                                            Mutually exclusive with ``store``.
            world_size (int, optional): Number of processes participating in
                                        the job. Required if ``store`` is specified.
            rank (int, optional): Rank of the current process (it should be a
                                    number between 0 and ``world_size``-1).
                                    Required if ``store`` is specified.
            store(Store, optional): Key/value store accessible to all workers, used
                                    to exchange connection/address information.
                                    Mutually exclusive with ``init_method``.
            timeout (timedelta, optional): Timeout for operations executed against
                the process group. Default value equals 30 minutes.
                This is applicable for the ``gloo`` backend. For ``nccl``, this is
                applicable only if the environment variable ``NCCL_BLOCKING_WAIT``
                or ``NCCL_ASYNC_ERROR_HANDLING`` is set to 1. When
                ``NCCL_BLOCKING_WAIT`` is set, this is the duration for which the
                process will block and wait for collectives to complete before
                throwing an exception. When ``NCCL_ASYNC_ERROR_HANDLING`` is set,
                this is the duration after which collectives will be aborted
                asynchronously and the process will crash. ``NCCL_BLOCKING_WAIT``
                will provide errors to the user which can be caught and handled,
                but due to its blocking nature, it has a performance overhead. On
                the other hand, ``NCCL_ASYNC_ERROR_HANDLING`` has very little
                performance overhead, but crashes the process on errors. This is
                done since CUDA execution is async and it is no longer safe to
                continue executing user code since failed async NCCL operations
                might result in subsequent CUDA operations running on corrupted
                data. Only one of these two environment variables should be set.
            group_name (str, optional, deprecated): Group name.
            pg_options (ProcessGroupOptions, optional): process group options
                specifying what additional options need to be passed in during
                the construction of specific process groups. As of now, the only
                options we support is ``ProcessGroupNCCL.Options`` for the ``nccl``
                backend, ``is_high_priority_stream`` can be specified so that
                the nccl backend can pick up high priority cuda streams when
                there're compute kernels waiting.

        .. note:: To enable ``backend == Backend.MPI``, PyTorch needs to be built from source
            on a system that supports MPI.
    '''
    global cdb
    global nccl_backend
    global mpi_backend
    global use_ds_backend
    global ds_world_rank
    global ds_world_size

    if use_deepspeed:
        if backend == 'nccl':
            if nccl_backend is not None:
                if nccl_backend.is_initialized():
                    cdb = nccl_backend
                else:
                    nccl_backend.initialize()
            else:
                nccl_backend = NcclBackend(rank=ds_world_rank, size=ds_world_size)
                cdb = nccl_backend
            use_ds_backend = True
        elif backend == 'mpi':
            logger.warn("MPI backend in DeepSpeed not yet implemented")
        elif backend == 'gloo':
            logger.warn(
                "Gloo backed is supported in DeepSpeed via torch.distributed only")
            init_torch_backend(backend)
        else:
            logger.warn(f"DeepSpeed does not support {backend} backend")
    else:
        if ds_world_rank == 0:
            logger.info(
                'Using torch.distributed as the communication backend in DeepSpeed')
        init_torch_backend(backend)
        use_ds_backend = False


def is_initialized():
    assert cdb is not None, 'DeepSpeed backend not set, please initialize it using init_process_group()'
    return cdb.is_initialized()


def init_torch_backend(backend):
    global cdb
    global ds_world_rank
    global ds_world_size

    if cdb is not None and cdb.is_initialized():
        if ds_world_rank == 0:
            logger.info('torch.distributed already initialized')
    else:
        if ds_world_rank == 0:
            logger.info('Initializing TorchBackend in DeepSpeed')
        cdb = TorchBackend(rank=ds_world_rank, size=ds_world_size, dist_backend=backend)


def destroy_process_group(group=None):
    global cdb
    return cdb.destroy_process_group(group=group)


def new_group(ranks):
    global cdb
    assert cdb is not None and cdb.is_initialized(), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    return cdb.new_group(ranks)


def is_available() -> bool:
    """
    Returns ``True`` if the deepspeed comm package is available.
    """
    # TODO: load other ops. Clients including deepspeed itself should use deepspeed.comm to import
    # any communication related primitives from this package.
    # use hasattr(deepspeed.csrc.ops, "_comm") or something
    return True


def set_backend(backend):
    if not use_ds_backend:
        logger.warn(
            "DeepSpeed communication backend is required. Please use deepspeed.comm.init_process_group(backend, use_deepspeed=True) to use this functionality"
        )
        return

    global cdb
    global nccl_backend
    global mpi_backend

    try:
        if backend_name == 'nccl':
            if nccl_backend is not None and nccl_backend.is_initialized():
                cdb = nccl_backend
        elif backend_name == 'mpi':
            if mpi_backend is not None and mpi_backend.is_initialized():
                cdb = mpi_backend
    except Exception as inst:
        print(inst)


def broadcast(tensor, src, group=None, async_op=False):
    global cdb
    return cdb.broadcast(tensor=tensor, src=src, group=group, async_op=async_op)


def all_gather(tensor_list, tensor, group=None, async_op=False):
    global cdb
    return cdb.all_gather(tensor_list=tensor_list,
                          tensor=tensor,
                          group=group,
                          async_op=async_op)


def all_to_all_single(
    output,
    input,
    output_split_sizes=None,
    input_split_sizes=None,
    group=None,
    async_op=False,
):
    global cdb
    return cdb.all_to_all_single(output=output,
                                 input=input,
                                 output_split_sizes=output_split_sizes,
                                 input_split_sizes=input_split_sizes,
                                 group=group,
                                 async_op=async_op)


def send(tensor, dst, group=None, tag=0):
    global cdb
    return cdb.send(tensor=tensor, dst=dst, group=group, tag=tag)


def recv(tensor, src=None, group=None, tag=0):
    global cdb
    return cdb.recv(tensor=tensor, src=src, group=group, tag=tag)


def gather(tensor, gather_list=None, dst=0, group=None, async_op=False):
    global cdb
    return cdb.gather(tensor=tensor,
                      gather_list=gather_list,
                      dst=dst,
                      group=group,
                      async_op=async_op)


def scatter(tensor, scatter_list=None, src=0, group=None, async_op=False):
    global cdb
    return cdb.scatter(tensor=tensor,
                       scatter_list=scatter_list,
                       src=src,
                       group=group,
                       async_op=async_op)


def barrier(group=None):
    global cdb
    return cdb.barrier()


# Local enum for Reduction operators
from .utils import ReduceOp


def reduce(tensor, dst, op=ReduceOp.SUM, group=None, async_op=False):
    global cdb
    return cdb.reduce(tensor=tensor, dst=dst, op=op, group=group, async_op=async_op)


def reduce_scatter(output, input_list, op=ReduceOp.SUM, group=None, async_op=False):
    global cdb
    return cdb.reduce_scatter(output=output,
                              input_list=input_list,
                              op=op,
                              group=group,
                              async_op=async_op)


def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    #if profile_comm:
    # context of the timers?
    # timers.start()
    # TensorBoard logging for comm calls.?
    global cdb
    print(f'op = {op}, cdb= {cdb.name}')
    return cdb.all_reduce(tensor, op, group, async_op)


def get_world_size(group=None) -> int:
    """
    Returns the number of processes in the current process group
    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
    Returns:
        The world size of the process group
        -1, if not part of the group
    """
    global cdb

    assert cdb is not None and cdb.is_initialized(), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    return cdb.get_world_size(group)


def get_rank(group=None):
    """
    Returns the rank of the current process in the provided ``group`` or the
    default group if none was provided.
    Rank is a unique identifier assigned to each process within a distributed
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.
    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
    Returns:
        The rank of the process group
        -1, if not part of the group
    """
    global cdb
    assert cdb is not None and cdb.is_initialized(), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    return cdb.get_rank(group)


def get_local_rank():
    """
        Helper function to get local rank after a backend has been set and initialized
        Args:
            None
        Returns:
            local rank (= GPU device ID)
    """
    assert cdb is not None and cdb.is_initialized(), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    return get_local_rank_from_launcher()
