'''
Copyright 2020 The Microsoft DeepSpeed Team
'''

import torch
import copy

class DeepSpeedExperts(torch.nn.Module):
    def __init__(self, expert, num_experts=1):
        super(DeepSpeedExperts, self).__init__()

        self.experts = torch.nn.ModuleList([copy.deepcopy(expert) for i in range(num_experts)])
        
        print("------------ creating experts in deepspeed = ", self.experts)
        for expert in self.experts:
            # TODO: Create param groups to handle expert + data case (e.g. param.group = moe_group)
            for name, param in expert.named_parameters():
                param.allreduce = False
                    
    def forward(self, inputs):
        print("forward in expert")
        for expert in self.experts:
            intermediate, extra_output = expert(inputs)
        return intermediate
