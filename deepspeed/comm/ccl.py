# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
'''
Copyright 2021 The Microsoft DeepSpeed Team
'''

import torch
from deepspeed.accelerator import get_accelerator
from .reduce_op import ReduceOp
from .torch import TorchBackend


def build_ccl_op():
    builder = get_accelerator().create_op_builder("CCLCommBuilder")
    if builder is None:
        return None
    ccl_cpp_module = builder.load()
    print(f'DeepSpeed {builder.absolute_name()} built successfully')
    return ccl_cpp_module

class CCLHandler():
    def __init__(self, ccl_comm_op=None):
        self.ccl_comm_op = ccl_comm_op
    
    def wait(self):
        # backend covered it
        pass

class CCLBackend(TorchBackend):

    def __init__(self, name='ccl', rank=-1, world_size=-1, mpu=None, timeout=None, init_method=None):
        self.ccl_comm_op = build_ccl_op()
        if self.ccl_comm_op is None:
            # set CCLBackend to uninitialized state if CCLCommBuilder cannot be loaded
            self.initialized = False
            return
        super(CCLBackend, self).__init__(backend='ccl',
                                         name='torch',
                                         rank=rank,
                                         world_size=world_size,
                                         timeout=timeout,
                                         init_method=init_method)
        self.name = 'ccl'
        size = self.get_world_size()
        rank = self.get_rank()
        main_kvs = self.ccl_comm_op.get_kvs_addr(rank)
        main_kvs = torch.tensor(main_kvs).to(torch.uint8).to(get_accelerator().device_name(rank))
        super(CCLBackend, self).broadcast(main_kvs, 0)
        self.ccl_comm_op.initialize(size, rank, main_kvs)
        self.initialized = True

    def is_initialized(self):
        return self.initialized

    def all_reduce(self, tensor, op=ReduceOp.SUM, group=None, async_op=False):
        use_caching = False
        group_ranks = self.get_all_ranks_from_group(group)
        if use_caching:
            match_id = f"{tensor.size()}-{op}"
            self.ccl_comm_op.all_reduce_caching(tensor, op, match_id, group_ranks, async_op)
        else:
            self.ccl_comm_op.all_reduce(tensor, op, group_ranks, async_op)

    def inference_all_reduce(self, tensor, op=ReduceOp.SUM, group=None, async_op=False):
        self.ccl_comm_op.inference_all_reduce(tensor, op, group, async_op)

    def broadcast(self, tensor, src, group=None, async_op=False):
        group_ranks = self.get_all_ranks_from_group(group)
        self.ccl_comm_op.broadcast(tensor, src, group_ranks, async_op)
        return CCLHandler(self.ccl_comm_op)         

    def all_gather(self, tensor_list, tensor, group=None, async_op=False):    
        group_ranks = self.get_all_ranks_from_group(group)
        self.ccl_comm_op.all_gather(tensor_list, tensor, group_ranks, async_op)
        return CCLHandler(self.ccl_comm_op)

    def reduce_scatter_tensor(self, output_tensor,input_tensor, op, group=None, async_op=False):
        #todo: ccl version
        super(CCLBackend, self).reduce_scatter_tensor(output_tensor,input_tensor, op, group)

    def all_gather_into_tensor(self, output_tensor, input_tensor, group=None, async_op=False):
        #todo: ccl version
        super(CCLBackend, self).all_gather_into_tensor(output_tensor, input_tensor, group)

    def all_to_all_single(self, output, input, output_split_sizes, input_split_sizes, group=None, async_op=False):
        #todo: ccl version
        super(CCLBackend, self).all_to_all_single(output, input, output_split_sizes, input_split_sizes, group)

    def send(self, tensor, dst, group=None, async_op=False):
        group_ranks = self.get_all_ranks_from_group(group)
        self.ccl_comm_op.send(tensor, dst, group_ranks, async_op)
        return CCLHandler(self.ccl_comm_op)
    
    def recv(self, tensor, src, group=None, async_op=False):
        group_ranks = self.get_all_ranks_from_group(group)
        self.ccl_comm_op.recv(tensor, src, group_ranks, async_op)
        return CCLHandler(self.ccl_comm_op)

    def gather(self, tensor, gather_list, dst, group=None, async_op=False):
        #todo: ccl version
        super(CCLBackend, self).gather(tensor, gather_list, dst, group)

    def scatter(self, tensor, gather_list, dst, group=None, async_op=False):
        #todo: ccl version
        super(CCLBackend, self).scatter(tensor, gather_list, dst, group)

    def barrier(self, group=None, async_op=False):      
        group_ranks = self.get_all_ranks_from_group(group)
        self.ccl_comm_op.barrier(group_ranks, async_op)
        return CCLHandler(self.ccl_comm_op)
        
    def monitored_barrier(self, group=None, timeout=None, wait_all_ranks=False):
        #todo: ccl version
        super(CCLBackend, self).monitored_barrier(group)

    def reduce_scatter(self, output, input_list, op=ReduceOp.SUM, group=None, async_op=False):
        group_ranks = self.get_all_ranks_from_group(group)
        self.ccl_comm_op.reduce_scatter(output, input_list, op, group_ranks, async_op)
        return CCLHandler(self.ccl_comm_op)

    def reduce(self, tensor, dst, op=ReduceOp.SUM, group=None, async_op=False): 
        group_ranks = self.get_all_ranks_from_group(group)
        self.ccl_comm_op.reduce(tensor, dst, op, group_ranks, async_op)
        return CCLHandler(self.ccl_comm_op)
    
    def new_group(self, ranks):
        size = len(ranks)
        rank = self.get_rank()
        if tuple(ranks) in self.groups or rank not in ranks:
            return
        sub_main_kvs = self.ccl_comm_op.get_sub_kvs_addr(rank == ranks[0])
        sub_main_kvs = torch.tensor(sub_main_kvs).to(torch.uint8).to("xpu:"+str(rank))
        torch_new_group = super(CCLBackend, self).new_group(ranks)
        super(CCLBackend, self).broadcast(sub_main_kvs, ranks[0], torch_new_group, False)
        self.ccl_comm_op.initialize_sub_comm(size, ranks.index(rank), sub_main_kvs, ranks)
        self.groups.append(tuple(ranks))
        return torch_new_group
    
    def get_all_ranks_from_group(self, group):
        if group is None:
            return list(range(self.get_world_size()))
        rank=0
        results=[]
        try:
            while True:
                results.append(torch.distributed.distributed_c10d._get_global_rank(group, rank))
                rank+=1
        except RuntimeError:
            pass

        if tuple(results) not in self.groups:
            self.new_group(results)
        return results
