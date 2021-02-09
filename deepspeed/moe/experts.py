'''
Copyright 2020 The Microsoft DeepSpeed Team
'''

import torch
import copy

class Experts(torch.nn.Module):
    def __init__(self, expert, num_local_experts=1, world_size=1):
        super(Experts, self).__init__()

        self.experts = torch.nn.ModuleList([copy.deepcopy(expert) for i in range(num_local_experts)])
        self.world_size = 1

        for expert in self.experts:
            # TODO: Create param groups to handle expert + data case (e.g. param.group = moe_group)
            for name, param in expert.named_parameters():
                param.allreduce = False
                    
    def forward(self, inputs):
        chunks = inputs.chunk(self.experts, dim=1)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.experts):
            expert_outputs += [expert(chunk)]
        expert_output = torch.cat(expert_outputs, dim=1)
        return expert_output
