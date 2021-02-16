'''
Copyright 2020 The Microsoft DeepSpeed Team
'''

import torch.nn.init as init
import torch
import torch.distributed as dist

# import for sanity testing only
#from .basic_moe import BasicMoE

from .sharded_moe import MOELayer, TopKGate
from .experts import Experts
import copy 

class MoE(torch.nn.Module):
    '''
        DeepSpeed MOE API: This defines a simple API that can be used from client-side code.
        E.g. See more details of usage from Megatron-LM code in https://github.com/microsoft/DeepSpeedExamples/tree/amawa/moe
    '''
    def __init__(self, hidden_size, output_dropout_prob, expert, num_experts = 1, k = 1): 
        super(MoE, self).__init__()

        world_size = dist.get_world_size()
        num_local_experts = num_experts // world_size

        self.num_experts = num_experts
        experts = Experts(expert, num_local_experts)
        # TODO Capacity factor needs to be configurable
        # TODO add top-k gate
        self.deepspeed_moe = MOELayer(TopKGate(hidden_size, num_experts), experts, num_local_experts)

        self.dropout = torch.nn.Dropout(output_dropout_prob)

    def forward(self, hidden_states):
        output = self.deepspeed_moe(hidden_states)
        output = self.dropout(output)
        return output, self.deepspeed_moe.l_aux

