'''
Copyright 2021 The Microsoft DeepSpeed Team
'''

import torch.distributed as dist

from deepspeed.ops.comm.ccl import build_ccl_op

from .utils import *
from .backend import *
from .reduce_op import ReduceOp
from deepspeed.utils import logger
import os


class CCLBackend(Backend):
    def __init__(self, name='ccl', rank=-1, size=-1, mpu=None):
        super(CCLBackend, self).__init__()
        # has_allgather_base is needed for torch. Included here for compatibility with ds comms
        self.has_allgather_base = True
        self.name = 'ccl'
        self.ccl_comm_op = build_ccl_op()
        if rank == -1:
            self.rank = self.ccl_comm_op.get_rank(0)
            self.size = self.ccl_comm_op.get_world_size(0)
        else:
            self.rank = int(rank)
            self.size = int(size)
        os.environ['CCL_LOCAL_SIZE'] = f"{self.size}"
        os.environ['CCL_LOCAL_RANK'] = f"{self.rank}"
        os.environ['LOCAL_SIZE'] = f"{self.size}"
        os.environ['LOCAL_RANK'] = f"{self.rank}"
        os.environ['WORLD_SIZE'] = f"{self.size}"
        os.environ['SIZE'] = f"{self.size}"
        os.environ['RANK'] = f"{self.rank}"
        self.enable_onebit = False
        self.init_process_group()

        if mpu is not None:
            # handle the mpu case later
            #self.world_group = dist.new_group(ranks=range(dist.get_world_size()))
            self.mpu = mpu
            #self.world_group = self.mpu.get_data_parallel_group()

    def init_process_group(self):
        logger.info(
            f"Initializing DeepSpeed's {self.name} Communication Backend with rank = {self.rank} and size = {self.size}"
        )
        if self.size <= 0:
            # Do not initialize torch distributed but only yourself
            self.initialized = True
            # Future functionality to support ds.initialize() on a single GPU
            self.single_gpu_mode = True
        else:
            self.ccl_comm_op.initialize(self.rank, self.size)
            self.initialized = True
            self.single_gpu_mode = False
        self.using_mpi = False

    def destroy_process_group(self, group=None):
        pass

    def new_group(self, ranks):
        return self.ccl_comm_op.new_group(ranks)
        # TODO: Change this to use comm_op.new_group when the impl. is ready.
        #if not torch.distributed.is_initialized():
        #    from deepspeed.comm.torch_backend import TorchBackend
        #    d = TorchBackend(rank=self.rank, size=self.size)
        #logger.info(f"new group called with {ranks}")
        #return torch.distributed.new_group(ranks)

    def test_set(self):
        self.ccl_comm_op.test_set()

    def get_rank(self, group=None):
        return self.ccl_comm_op.get_rank(0)

    def get_world_size(self, group=None):
        return self.ccl_comm_op.get_world_size(0)

    def is_initialized(self):
        return self.initialized

    def get_world_group(self):
        return self.ccl_comm_op.get_world_group()

    def barrier(self, group=None, async_op=False):
        self.ccl_comm_op.barrier(group, async_op)

    def broadcast(self, tensor, src, group=None, async_op=False):
        # TODO: Fix calls to op. Fix op to support groups and async
        self.ccl_comm_op.broadcast(tensor, src, group, async_op)

    def send(self, tensor, dst, group=None, tag=0, async_op=False):
        self.ccl_comm_op.send(tensor, dst, tag, group, async_op)

    def recv(self, tensor, src=None, group=None, tag=0, async_op=False):
        self.ccl_comm_op.recv(tensor, src, tag, group, async_op)

    def all_reduce(self, tensor, op=ReduceOp.SUM, group=None, async_op=False):
        self.ccl_comm_op.all_reduce(tensor, op, group, async_op)

    def reduce(self, tensor, dst, op=ReduceOp.SUM, group=None, async_op=False):
        self.ccl_comm_op.reduce(tensor, dst, op, group, async_op)

    def reduce_scatter(self,
                       output,
                       input_list,
                       op=ReduceOp.SUM,
                       group=None,
                       async_op=False):
        self.ccl_comm_op.reduce_scatter(tensor, op, group, async_op)

    def all_gather(self, tensor_list, tensor, group=None, async_op=False):
        self.ccl_comm_op.all_gather([tensor_list], [tensor], group, async_op)

    def all_gather_base(self,
                        output_tensor,
                        input_tensor,
                        group=None,
                        async_op=False,
                        comm_id=0):
        self.ccl_comm_op.all_gather_base(output_tensor, input_tensor, group, async_op)

    def all_to_all_single(self,
                          output,
                          input,
                          output_split_sizes=None,
                          input_split_sizes=None,
                          group=None,
                          async_op=False):
        self.ccl_comm_op.all_to_all_single(output, input, group, async_op)

    def all_to_all(self,
                   output_tensor_list,
                   input_tensor_list,
                   group=None,
                   async_op=False):
        self.ccl_comm_op.all_to_all(output, input, group, async_op)

    def synchronize():
        self.ccl_comm_op.synchronize()

    def create_comm_group(self, comm_ranks, rank, comm_id, color):
        self.ccl_comm_op.create_comm_group(comm_ranks, rank, comm_id, color)

    def my_igather(self, rank, size, group, sendbuf, recvbuf, root):
        req = []
        if rank == root:
            for idx in range(size):
                if idx != rank:
                    req.append(dist.irecv(recvbuf[idx], src=idx, group=group))
                else:
                    recvbuf[rank] = sendbuf
        else:
            req.append(dist.isend(sendbuf, group=group, dst=root))
        return req

    def my_gather(self, rank, size, group, sendbuf, recvbuf, root):
        if rank == root:
            for idx in range(size):
                if idx != rank:
                    dist.recv(recvbuf[idx], src=idx, group=group)
                else:
                    recvbuf[rank] = sendbuf
        else:
            dist.send(sendbuf, group=group, dst=root)
