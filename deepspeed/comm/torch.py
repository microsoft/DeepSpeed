'''
Copyright 2021 The Microsoft DeepSpeed Team
'''
import os
import torch
from datetime import timedelta

from ..utils.logging import logger
from ..constants import TORCH_DISTRIBUTED_DEFAULT_PORT, default_pg_timeout

import torch.distributed as dist

from .utils import *
from .backend import *


class TorchBackend(Backend):
    """
        DeepSpeed wrapper class for torch.distributed functionality.
        Only a subset of functions are wrapped. Once the init_process_group
        is initialized, standard torch.distributed.func*() can be used diretly
        so no need to wrap all the functions.
    """
    def __init__(self, name='torch', rank=0, size=1, dist_backend="nccl"):
        super(TorchBackend, self).__init__()
        self.init_process_group(name, rank, size, dist_backend)

    def init_process_group(self, name='torch', rank=0, size=1, dist_backend="nccl"):
        if size <= -1:
            # Do not initialize torch distributed but only yourself
            self.initialized = True
            # Future functionality to support ds.initialize() on a single GPU
            self.single_gpu_mode = True
        else:
            init_distributed(dist_backend)
            if torch.distributed.is_initialized():
                self.initalized = True
                self.single_gpu_mode = False

    def _reduce_op(self, op):
        '''
            Helper function. If the op provided is not a torch.dist.ReduceOp, convert it and return
        '''
        if not isinstance(op, torch.distributed.ReduceOp):
            if op == ReduceOp.SUM:
                op = torch.distributed.ReduceOp.SUM
            elif op == ReduceOp.PROD:
                op = torch.distributed.ReduceOp.PROD
            elif op == ReduceOp.AVG:
                op = torch.distributed.ReduceOp.AVG
            elif op == ReduceOp.MIN:
                op = torch.distributed.ReduceOp.MIN
            elif op == ReduceOp.MAX:
                op = torch.distributed.ReduceOp.MAX
            elif op == ReduceOp.BAND:
                op = torch.distributed.ReduceOp.BAND
            elif op == ReduceOp.BOR:
                op = torch.distributed.ReduceOp.BOR
            elif op == ReduceOp.BXOR:
                op = torch.distributed.ReduceOp.BXOR
        return op

    def all_reduce(self,
                   tensor,
                   op=torch.distributed.ReduceOp.SUM,
                   group=None,
                   async_op=False):
        op = self._reduce_op(op)
        print('op = {op}')
        return torch.distributed.all_reduce(tensor=tensor,
                                            op=op,
                                            group=group,
                                            async_op=async_op)

    def reduce(self, tensor, dst, op=ReduceOp.SUM, group=None, async_op=False):
        return torch.distributed.reduce(tensor=tensor,
                                        dst=dst,
                                        op=self._reduce_op(op),
                                        group=group,
                                        async_op=async_op)

    def reduce_scatter(self,
                       output,
                       input_list,
                       op=ReduceOp.SUM,
                       group=None,
                       async_op=False):
        return torch.distributed.reduce_scatter(output=output,
                                                input_list=input_list,
                                                op=self._reduce_op(op),
                                                group=group,
                                                async_op=async_op)

    def broadcast(self, tensor, src, group=None, async_op=False):
        return torch.distributed.broadcast(tensor=tensor,
                                           src=src,
                                           group=group,
                                           async_op=async_op)

    def all_gather(self, tensor_list, tensor, group=None, async_op=False):
        return torch.distributed.all_gather(tensor_list=tensor_list,
                                            tensor=tensor,
                                            group=group,
                                            async_op=async_op)

    def all_to_all_single(self,
                          output,
                          input,
                          output_split_sizes=None,
                          input_split_sizes=None,
                          group=None,
                          async_op=False):
        return torch.distributed.all_to_all_single(output=output,
                                                   input=input,
                                                   output_split_sizes=output_split_sizes,
                                                   input_split_sizes=input_split_sizes,
                                                   group=group,
                                                   async_op=async_op)

    def send(self, tensor, dst, group=None, tag=0):
        return torch.distributed.send(tensor=tensor, dst=dst, group=group, tag=tag)

    def recv(self, tensor, src=None, group=None, tag=0):
        return torch.distributed.recv(tensor=tensor, src=src, group=group, tag=tag)

    def gather(self, tensor, gather_list=None, dst=0, group=None, async_op=False):
        return torch.distributed.gather(tensor=tensor,
                                        gather_list=gather_list,
                                        dst=dst,
                                        group=group,
                                        async_op=async_op)

    def scatter(self, tensor, scatter_list=None, src=0, group=None, async_op=False):
        return torch.distributed.scatter(tensor=tensor,
                                         scatter_list=scatter_list,
                                         src=src,
                                         group=group,
                                         async_op=async_op)

    def barrier(self):
        return torch.distributed.barrier()

    def get_rank(self, group=None):
        return torch.distributed.get_rank(group=group)

    def get_world_size(self, group=None):
        return torch.distributed.get_world_size(group=group)

    def is_initialized(self):
        return torch.distributed.is_initialized()

    def get_backend(self, group=None):
        return torch.distributed.get_backend(group=group)

    def new_group(self, ranks):
        logger.info(f"new group called with {ranks}")
        return torch.distributed.new_group(ranks)

    def destroy_process_group(self, group=None):
        return torch.distributed.destroy_process_group(group=group)


# The functions below are kept global so they can be used without creating a TorchBackend object
# Kept for legacy reasons and can be deprecated/made-class-members in future
def init_distributed(dist_backend="nccl",
                     auto_mpi_discovery=True,
                     distributed_port=TORCH_DISTRIBUTED_DEFAULT_PORT,
                     verbose=True,
                     timeout=default_pg_timeout,
                     init_method=None):
    """Initialize torch.distributed backend, potentially performing MPI discovery if needed

    Arguments:
        dist_backend: Optional (str). torch distributed backend, e.g., nccl, mpi, gloo

        auto_mpi_discovery Optional (bool). if distributed environment variables are not set, attempt to discover them from MPI

        distributed_port: Optional (int). torch distributed backend port

        verbose: Optional (bool). verbose logging

        timeout: Optional (timedelta). Timeout for operations executed against the process group. Default value equals 30 minutes.

        init_method: Optional (string). Torch distributed, URL specifying how to initialize the process group. Default is “env://” if no init_method or store is specified.
    """
    required_env = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    if auto_mpi_discovery and not all(map(lambda v: v in os.environ, required_env)):
        if verbose:
            if get_world_rank_from_launcher() == 0:
                logger.info(
                    "Not using the DeepSpeed or torch.distributed launchers, attempting to detect MPI environment..."
                )
        if in_aml() and not in_dlts():
            patch_aml_env_for_torch_nccl_backend(verbose=verbose)
        else:
            mpi_discovery(distributed_port=distributed_port, verbose=verbose)

    if not torch.distributed.is_initialized():
        if verbose:
            if get_world_rank_from_launcher() == 0:
                logger.info("Initializing torch.distributed with backend: {}".format(
                    dist_backend))
        assert isinstance(timeout, timedelta)
        torch.distributed.init_process_group(backend=dist_backend,
                                             timeout=timeout,
                                             init_method=init_method)


def mpi_discovery(distributed_port=TORCH_DISTRIBUTED_DEFAULT_PORT, verbose=True):
    """
    Discovery MPI environment via mpi4py and map to relevant torch.distributed state
    """
    from mpi4py import MPI
    import subprocess
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    master_addr = None
    if rank == 0:
        hostname_cmd = ["hostname -I"]
        result = subprocess.check_output(hostname_cmd, shell=True)
        master_addr = result.decode('utf-8').split()[0]
    master_addr = comm.bcast(master_addr, root=0)

    # Determine local rank by assuming hostnames are unique
    proc_name = MPI.Get_processor_name()
    all_procs = comm.allgather(proc_name)
    local_rank = sum([i == proc_name for i in all_procs[:rank]])

    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(distributed_port)

    if verbose:
        if get_world_rank_from_launcher() == 0:
            logger.info(
                "Discovered MPI settings of world_rank={}, local_rank={}, world_size={}, master_addr={}, master_port={}"
                .format(os.environ['RANK'],
                        os.environ['LOCAL_RANK'],
                        os.environ['WORLD_SIZE'],
                        os.environ['MASTER_ADDR'],
                        os.environ['MASTER_PORT']))

    if torch.distributed.is_initialized():
        assert torch.distributed.get_rank() == rank, "MPI rank {} does not match torch rank {}".format(
            rank, torch.distributed.get_rank())
        assert torch.distributed.get_world_size() == world_size, "MPI world size {} does not match torch world size {}".format(
            world_size, torch.distributed.get_world_size())


def in_aml():
    # Are we running inside an Azure Machine Learning (AML) environment?
    return 'AZUREML_EXPERIMENT_ID' in os.environ


def in_dlts():
    # Are we running on a DLTS cluster?
    return 'DLTS_JOB_ID' in os.environ


def patch_aml_env_for_torch_nccl_backend(master_port=6105, verbose=True):
    """Helper routine to get and set environment variables.
    This is adapted from Azure ML's documentation available from:
    https://azure.github.io/azureml-web/docs/cheatsheet/distributed-training/#environment-variables-from-openmpi
    """
    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
    single_node = int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"]) == int(
        os.environ["WORLD_SIZE"])

    if not single_node:
        master_node_params = os.environ["AZ_BATCH_MASTER_NODE"].split(":")
        os.environ["MASTER_ADDR"] = master_node_params[0]
        # Do not overwrite master port with that defined in AZ_BATCH_MASTER_NODE
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(master_port)
    else:
        os.environ["MASTER_ADDR"] = os.environ["AZ_BATCHAI_MPI_MASTER_NODE"]
        os.environ["MASTER_PORT"] = "54965"

    if verbose:
        logger.info("NCCL_SOCKET_IFNAME original value = {}".format(
            os.environ["NCCL_SOCKET_IFNAME"]))

    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
    os.environ['LOCAL_RANK'] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]

    if verbose:
        if get_world_rank_from_launcher() == 0:
            logger.info(
                "Discovered AzureML settings of world_rank={}, local_rank={}, world_size={}, master_addr={}, master_port={}"
                .format(os.environ['RANK'],
                        os.environ['LOCAL_RANK'],
                        os.environ['WORLD_SIZE'],
                        os.environ['MASTER_ADDR'],
                        os.environ['MASTER_PORT']))


# This will become a light-weight wrapper around torch.distributed functions
# TODO: create some example to show how this wrapper can help profile communication
# TODO: make sure there is no performance regression with this approach
# TODO: explore monkey-patching if this does not work
