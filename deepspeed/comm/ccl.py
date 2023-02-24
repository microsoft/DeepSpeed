'''
Copyright 2021 The Microsoft DeepSpeed Team
'''

#import torch
import torch.distributed as dist
#import numpy as np

from deepspeed.ops.comm.ccl import build_ccl_op
#from deepspeed.ops.comm.mpi import build_mpi_op
#from deepspeed.ops.comm.common import build_op
#from deepspeed.ops.comm import ReduceOp

from .utils import *
from .backend import *
#from deepspeed.ops.comm.common import ReduceOp
#from deepspeed.ops.comm.nccl import ReduceOp
#from .comm import ReduceOp
#from .utils import ReduceOp
#from deepspeed.ops.comm.nccl import ReduceOp
from .comm import ReduceOp
from deepspeed.utils import logger

cupy = None


class CclBackend(Backend):
    def __init__(self, name='ccl', rank=0, size=1, mpu=None):
        super(CclBackend, self).__init__()
        # has_allgather_base is needed for torch. Included here for compatibility with ds comms
        self.has_allgather_base = True
        self.name = 'ccl'
        self.ccl_comm_op = build_ccl_op()
        #self.mpi_comm_op = build_mpi_op()
        #self.reduce_op = build_op().ReduceOp
        #self.mpi_comm_op.initialize()
        #self.rank = get_local_rank_from_launcher()
        #self.size = get_world_size_from_launcher()
        self.rank = self.ccl_comm_op.get_rank(0)
        self.size = self.ccl_comm_op.get_world_size(0)
        self.enable_onebit = False
        self.init_process_group()

        if mpu is not None:
            # handle the mpu case later
            #self.world_group = dist.new_group(ranks=range(dist.get_world_size()))
            self.mpu = mpu
            #self.world_group = self.mpu.get_data_parallel_group()

        #if self.enable_onebit:
        #    global cupy
        #    import cupy
        #    from deepspeed.runtime.compression.cupy import CupyBackend
        #    self.compression_backend = CupyBackend()

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

    def barrier(self):
        self.ccl_comm_op.barrier()

    def broadcast(self, tensor, src, group=None, async_op=False, block=False):
        # TODO: Fix calls to op. Fix op to support groups and async
        self.ccl_comm_op.broadcast(tensor,
                                   src,
                                   block,
                                   group,
                                   async_op)  #, group=group, async_op=async_op)

    def send(self, tensor, dst, group=None, tag=0, block=False, async_op=False):
        self.ccl_comm_op.send(tensor, dst, tag, block, group, async_op)

    def recv(self, tensor, src=None, group=None, tag=0, block=False, async_op=False):
        self.ccl_comm_op.recv(tensor, src, tag, block, group, async_op)

    def all_reduce(self,
                   tensor,
                   op=ReduceOp.SUM,
                   group=None,
                   async_op=False,
                   block=False):
        self.ccl_comm_op.all_reduce(tensor, op, block, group, async_op)

    def reduce(self,
               tensor,
               dst,
               op=ReduceOp.SUM,
               group=None,
               async_op=False,
               block=False):
        self.ccl_comm_op.reduce(tensor, dst, op, block, group, async_op)

    def reduce_scatter(self,
                       output,
                       input_list,
                       op=ReduceOp.SUM,
                       group=None,
                       async_op=False,
                       block=False):
        self.ccl_comm_op.reduce_scatter(tensor, op, block, group, async_op)

    def all_gather(self, tensor_list, tensor, group=None, async_op=False, block=False):
        self.ccl_comm_op.all_gather([tensor_list], [tensor], block, group, async_op)

    def all_gather_base(self,
                        output_tensor,
                        input_tensor,
                        group=None,
                        async_op=False,
                        block=False,
                        comm_id=0):
        self.ccl_comm_op.all_gather_base(output_tensor,
                                         input_tensor,
                                         block,
                                         group,
                                         async_op)

    def all_to_all_single(self,
                          output,
                          input,
                          output_split_sizes=None,
                          input_split_sizes=None,
                          group=None,
                          async_op=False,
                          block=False):
        self.ccl_comm_op.all_to_all_single(output, input, block, group, async_op)

    def all_to_all(self,
                   output_tensor_list,
                   input_tensor_list,
                   group=None,
                   async_op=False,
                   block=False):
        self.ccl_comm_op.all_to_all(output, input, block, group, async_op)

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

    '''
    def compressed_allreduce(self,
                             buffer_m: torch.tensor,
                             worker_error,
                             server_error,
                             local_rank):

        # all_start_time = time.time()
        original_shape = buffer_m.size()
        if len(original_shape) > 1:
            buffer_m = torch.flatten(buffer_m)
        original_size = buffer_m.numel()
        worker_error_size = worker_error.numel()
        cupy.cuda.Device(local_rank).use()

        if original_size != worker_error_size:
            empty_tensor = torch.zeros(worker_error_size - original_size,
                                       device=buffer_m.device)
            buffer_m = torch.cat([buffer_m, empty_tensor])

        buffer_m.add_(worker_error)
        worker_scale = torch.norm(buffer_m) / np.sqrt(torch.numel(buffer_m))
        worker_error.set_(buffer_m - worker_scale *
                          buffer_m.sign().add_(1).bool().float().add_(-0.5).mul_(2.0))

        cupy_sign_list_packed = self.compression_backend.compress_by_chunk(
            self.compression_backend.torch2cupy(buffer_m.sign_().add_(1).bool()),
            self.size)
        cupy_worker_scale = self.compression_backend.torch2cupy(worker_scale)

        cupy_recvbuf_sign = cupy.zeros(
            [self.size,
             cupy_sign_list_packed[self.rank].size],
            dtype=cupy_sign_list_packed[0].dtype)
        # cupy_recvbuf_scale = cupy.zeros([self.size, 1], dtype=cupy_worker_scale.dtype)

        sign_list_packed = [
            self.compression_backend.cupy2torch(cupy_sign_list_packed[idx])
            for idx in range(self.size)
        ]

        # worker_scale = self.compression_backend.cupy2torch(cupy_worker_scale)
        recvbuf_sign = self.compression_backend.cupy2torch(cupy_recvbuf_sign)
        #recvbuf_scale = self.compression_backend.cupy2torch(cupy_recvbuf_scale)
        recvbuf_scale = [
            torch.zeros(1,
                        dtype=worker_scale.dtype,
                        device=torch.device(local_rank)) for i in range(self.size)
        ]

        # communication phase 1
        # gather_start = time.time()
        # Alltoall for sign
        dist.all_to_all_single(recvbuf_sign,
                               torch.stack(sign_list_packed),
                               group=self.world_group)
        # Allgather for scale
        dist.all_gather(recvbuf_scale, worker_scale, group=self.world_group)

        # gather_end = time.time()

        # cupy_sign_list_packed, sign_list_packed, cupy_worker_scale, worker_scale = None, None, None, None
        cupy_sign_list_packed = None

        cupy_recvbuf_sign = self.compression_backend.torch2cupy(recvbuf_sign)
        #cupy_recvbuf_scale = self.compression_backend.torch2cupy(torch.stack(recvbuf_scale))

        compensated_server_m = self.compression_backend.cupy2torch(
            (cupy.unpackbits(cupy_recvbuf_sign.flatten())).reshape(
                self.size,
                -1)).float().add_(-0.5).mul_(2.0).mul_(
                    torch.stack(recvbuf_scale).mul_(1 / self.size)).sum(0)
        compensated_server_m.add_(server_error)
        server_scale = torch.norm(compensated_server_m) / np.sqrt(
            compensated_server_m.numel())
        server_error.set_(
            compensated_server_m - server_scale *
            compensated_server_m.sign().add_(1).bool().float().add_(-0.5).mul_(2.0))

        # cupy_server_scale = self.compression_backend.torch2cupy(server_scale)

        cupy_server_sign_packed = self.compression_backend.compress_by_chunk(
            self.compression_backend.torch2cupy(
                compensated_server_m.sign_().add_(1).bool()),
            1)
        compensated_server_m = None

        cupy_recvbuf_sign_server = cupy.zeros(
            [self.size,
             cupy_server_sign_packed[0].size],
            dtype=cupy_recvbuf_sign.dtype)
        # cupy_recvbuf_sign, recvbuf_sign = None, None
        cupy_recvbuf_sign = None

        server_sign_packed = [
            self.compression_backend.cupy2torch(cupy_server_sign_packed[0])
        ]
        recvbuf_sign_server = [
            self.compression_backend.cupy2torch(cupy_recvbuf_sign_server[idx])
            for idx in range(self.size)
        ]

        # server_scale = self.compression_backend.cupy2torch(cupy_server_scale)
        cupy_recvbuf_scale_server = cupy.zeros([self.size,
                                                1],
                                               dtype=cupy_worker_scale.dtype)
        # cupy_recvbuf_scale, recvbuf_scale = None, None

        recvbuf_scale_server = [
            self.compression_backend.cupy2torch(cupy_recvbuf_scale_server[idx])
            for idx in range(self.size)
        ]

        # Communication Phase 2
        dist.all_gather(recvbuf_sign_server,
                        server_sign_packed[0],
                        group=self.world_group)
        dist.all_gather(recvbuf_scale_server, server_scale, group=self.world_group)

        cupy_server_sign_packed = None

        # need to convert from a tensor list to a single tensor
        # dist.all_gather only provides a tensor list as the recv/output buffer
        recvbuf_sign_server = torch.stack(recvbuf_sign_server)

        cupy_recvbuf_sign_server = self.compression_backend.torch2cupy(
            recvbuf_sign_server)

        buffer_m.data.copy_(
            self.compression_backend.cupy2torch(
                (cupy.unpackbits(cupy_recvbuf_sign_server.flatten())).reshape(
                    self.size,
                    -1)).float().add_(-0.5).mul_(2.0).mul_(
                        self.compression_backend.cupy2torch(
                            cupy_recvbuf_scale_server)).flatten().data)
        if original_size != worker_error_size:
            buffer_m = buffer_m[0:original_size]
        if len(original_shape) > 1:
            buffer_m = buffer_m.reshape(original_shape)

        return buffer_m
    '''
