import torch
from .base import BaseTransformerContainer

class BaseTransformerMoEContainer(BaseTransformerContainer):
    def __init__(self, policy):
        self.policy = policy

        # Call the init function of the parent class to initialize the tensors and configs from parent class
        super.__init__(self, policy)
        
        self.num_experts = 1
        
        # MoE models will have a list of mlp related tensors
        self._4hh_w = []
        self._4hh_b = []
        self._h4h_w = []
        self._h4h_b = []
        
        # Residual MoE needs extra parameters
        self._res_h4h_b = None
        self._res_4hh_b = None
        self._res_h4h_w = None
        self._res_4hh_w = None
        self._res_coef = None
            
    def initialize_tensors(self):
        self.num_experts = self.policy.get_num_experts()
        # todo: refactor this to become part of config instead of tensor list
        self.set_hidden_heads(*self.policy.get_hidden_heads())
        assert self.num_attention_heads % self.mp_size == 0,\
                "To run the model parallel across the GPUs, the attention_heads require to be divisible by the world_size!" +\
                "This is because the attention computation is partitioned evenly among the parallel GPUs."
        # Set the tensors from policy (user module) to container (DS module)
        self.set_attention(*self.policy.attention())
        self.set_mlp(*self.policy.mlp())
        self.set_layernorm(*self.policy.layernorm())

    def set_mlp(self, _h4h_w, _h4h_b, _4hh_w, _4hh_b):
        self._h4h_w = _h4h_w
        self._h4h_b = _h4h_b
        self._4hh_w = _4hh_w
        self._4hh_b = _4hh_b
