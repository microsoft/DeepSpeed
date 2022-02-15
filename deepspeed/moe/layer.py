'''
Copyright 2020 The Microsoft DeepSpeed Team
'''

import torch.nn.init as init
import torch
import torch.distributed as dist

from deepspeed.utils import logger, log_dist

import deepspeed.utils.groups as groups
from .sharded_moe import MOELayer, TopKGate
from .experts import Experts
import copy
import typing


class MoE(torch.nn.Module):
    def __init__(self,
                 hidden_size,
                 expert,
                 num_experts=1,
                 ep_size=1,
                 mpu=None,
                 k=1,
                 capacity_factor=1.,
                 eval_capacity_factor=1.,
                 min_capacity=4,
                 noisy_gate_policy: typing.Optional[str] = None,
                 drop_tokens: bool = True,
                 use_rts=True,
                 use_tutel: bool = False):
        """Initialize an MoE layer.

        Arguments:
            hidden_size (int): the hidden dimension of the model, importantly this is also the input and output dimension.
            expert (torch.nn.Module): the torch module that defines the expert (e.g., MLP, torch.linear).
            num_experts (int, optional): default=1, the total number of experts per layer.
            ep_size (int, optional): default=1, number of ranks in the expert parallel world or group.
            mpu (mpu, optional): default=None, a Megatron style mpu object to support tensor parallelism in addition to expert parallelism.
            k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.
            capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.
            eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.
            min_capacity (int, optional): default=4, the minimum capacity per expert regardless of the capacity_factor.
            noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample' or 'None'.
            drop_tokens (bool, optional): default=True, whether to drop tokens - (setting to False is equivalent to infinite capacity).
            use_rts (bool, optional): default=True, whether to use Random Token Selection.
            use_tutel (bool, optional): default=False, whether to use Tutel optimizations (if installed).
        """

        super(MoE, self).__init__()

        self.ep_size = ep_size
        self.mpu = mpu
        self.expert_group_name = f"ep_size_{self.ep_size}"
        self.num_experts = num_experts
        self.num_local_experts = 1 if num_experts < ep_size else num_experts // ep_size

        log_dist(
            f'Creating MoE layer with num_experts: {num_experts} | num_local_experts: {self.num_local_experts} | expert_parallel_size: {ep_size}',
            [0])

        # Create the required process group if not already created
        self.create_process_groups()

        assert noisy_gate_policy is None or noisy_gate_policy in ['None', 'Jitter', 'RSample'], \
            'Unsupported noisy_gate_policy: ' + noisy_gate_policy

        experts = Experts(expert, self.num_local_experts, self.expert_group_name)
        self.deepspeed_moe = MOELayer(TopKGate(hidden_size,
                                               num_experts,
                                               k,
                                               capacity_factor,
                                               eval_capacity_factor,
                                               min_capacity,
                                               noisy_gate_policy,
                                               drop_tokens,
                                               use_rts),
                                      experts,
                                      self.num_local_experts,
                                      group=groups.get_expert_parallel_group(
                                          self.expert_group_name),
                                      use_tutel=use_tutel)

    def create_process_groups(self):
        if self.expert_group_name not in groups.get_expert_parallel_group_dict():
            print(
                f"MoE Layer found no existing expert parallel process group, creating a new group named: {self.expert_group_name}"
            )
            if self.mpu is None:
                groups.create_expert_and_data_parallel(self.ep_size)
            else:
                groups.create_expert_data_and_model_parallel(ep_size=self.ep_size,
                                                             mpu=self.mpu)

    def forward(self, hidden_states, used_token=None):
        """ MoE forward

        Arguments:
            hidden_states (Tensor): input to the layer
            used_token (Tensor, optional): default: None, mask only used tokens

        Returns:
            A tuple including output, gate loss, and expert count.

            * output (Tensor): output of the model

            * l_aux (Tensor): gate loss value

            * exp_counts (int): expert count
        """
        output = self.deepspeed_moe(hidden_states, used_token)
        return output, self.deepspeed_moe.l_aux, self.deepspeed_moe.exp_counts
