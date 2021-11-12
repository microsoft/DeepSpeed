import torch
import os

ds_inference = None


class InferenceComm:
    current_comm = None

    def __init__(self, ranks, set_device=True):
        if ranks is None:
            self.ranks = range(int(os.getenv('WORLD_SIZE', '0')))
        else:
            self.ranks = sorted(ranks)

        self.mp_size = len(self.ranks)
        local_rank = int(os.getenv('LOCAL_RANK', '0'))

        global ds_inference
        if ds_inference is None:
            from ... import op_builder
            builder = op_builder.InferenceBuilder()
            ds_inference = builder.load()
        if set_device:
            torch.cuda.set_device(f"cuda:{local_rank}")
        ds_inference.init_comm_group(self.ranks, local_rank)

    def all_reduce(self, val, inplace=True, async_op=False):
        val_sum = val if inplace else torch.empty_like(val)
        op = communicate_op(val, val_sum, async_op, op_type="all_reduce")
        return val_sum, op

    def all_gather(self, val, inplace=True, async_op=False):
        val_gather = torch.empty((self.mp_size * val.size(0),
                                  *val.shape[1:]),
                                 device=val.device)
        op = communicate_op(val, val_gather, async_op, op_type="all_gather")
        return val_gather, op

    def broadcast(self, val, inplace=True, async_op=False):
        val_bcst = torch.empty_like(val)
        op = communicate_op(val, val_bcst, async_op, op_type="broadcast")
        return val_bcst, op

    def barrier(self):
        ds_inference.wait_comm()
        ds_inference.barrier()

    @classmethod
    def get_current_comm(cls):
        if cls.current_comm is None:
            cls.current_comm = cls(ranks=None)
        return cls.current_comm


class communicate_op:
    def __init__(self, val, result, async_op, op_type="all_reduce"):
        if op_type == "all_reduce":
            ds_inference.allReduce(val, result, val.numel(), async_op)
        elif op_type == "all_gather":
            ds_inference.allGather(val, result, val.numel(), async_op)
        elif op_type == "broadcast":
            ds_inference.broadcast(val, result, val.numel(), async_op)

    def wait(self):
        ds_inference.wait_comm()


def create_comm(ranks=None, set_device=True):
    return InferenceComm(ranks=ranks, set_device=set_device)


def get_default_comm():
    return InferenceComm.get_current_comm()
