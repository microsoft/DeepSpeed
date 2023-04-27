# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from abc import ABC, abstractmethod
from typing import List, Tuple

import torch


class HybridEngineContainer(ABC):
    """
    This container identifies which methods need to be overridden in addition to
    the base container to enable use in the RLHF pipeline. These methods are not
    necessary for inference alone.
    """

    def initalize_tensors(self, enable_training=False):
        """
        Same purposes as the base container, but also grabs the hooks for any LoRA
        parameters.
        """
        super().initialize_tensors(enable_training=enable_training)
        self.set_lora_params()

    @abstractmethod
    def set_lora_params(self, lora_params):
        """
        If available, set the LoRA parameters for the module. It is unlikely this needs to be
        modified for different models.
        """
        raise NotImplementedError("A set_lora_params() function must be defined for the relevant parameters.")

    def apply_tensor_parallelism(self, mp_replace, reversed_dim=False):
        """
        Add support for reversed dim in tensor parallelism. If necessary, override
        the called methods to handle partitioned weights (i.e. if qkv is split)
        """
        # Setup the new Attention module
        self.attention_qkv_mp(mp_replace, reversed_dim=reversed_dim)
        self.attention_out_mp(mp_replace, reversed_dim=reversed_dim)

        # Setup the new MLP module
        self.mlp_inter_mp(mp_replace, reversed_dim=reversed_dim)
        self.mlp_output_mp(mp_replace, reversed_dim=reversed_dim)

        # Apply weight quantization
        # TODO(cmikeh2): Re-enable this once verified
        #self.apply_weight_quantization()

    def _release_params(self, param_pairs: List[Tuple[torch.Tensor, torch.Tensor]]):
        """
        Helper for `release_[component]` methods. Accepts a list of tuples where the first
        element is the module param that needs to be deleted, and the second is the reassignment
        from the container.
        """
        for module_param, container_param in param_pairs:
            if module_param is not None:
                del module_param
            module_param = container_param

    def release_memory(self):
        """
        Delete module parameters if they exist and point them back to the container. This
        should cover all populated params in the container, even those that may alias with
        each other.
        """
        # Release the memory for parameters that should be universally
        # releaseable.
        general_params = [
            (self.module.attention.attn_ow, self.dense_w),
            (self.module.attention.attn_ob, self.dense_b),
            (self.module.attn_nw, self.attn_nw),
            (self.module.attn_nb, self.attn_nb),
            (self.module.norm_w, self.input_nw),
            (self.module.norm_b, self.input_nb),
        ]

        self._release_params(general_params)

        self.release_qkv()
        self.release_mlp()

    def release_qkv(self):
        """
        Release for QKV parameters (as well as any aliases).
        """
        qkv_params = [
            (self.module.attention.attn_qkvw, self.qkvw),
            (self.module.attention.attn_qkvb, self.qkvb),
        ]

        self._release_params(qkv_params)

    def release_mlp(self):
        """
        Release for MLP parameters (as well as any aliases).
        """
        mlp_params = [
            (self.module.mlp.inter_w, self._h4h_w),
            (self.module.mlp.inter_b, self._h4h_b),
            (self.module.mlp.output_w, self._4hh_w),
            (self.module.mlp.output_b, self._4hh_b),
        ]

        self._release_params(mlp_params)

    def reset_params(self):
        """
        The purpose of reset params is to get the weights from the FP16 training
        copy of the model and copy to them to contiguous inference view. This only needs
        to be performed when the container parameters cannot be used directly for inference.
        """
        self.reset_qkv()
        self.reset_mlp()

    def reset_qkv(self):
        """
        Perform any necessary resets of the model parameters for the QKV components.
        """
        pass

    def reset_mlp(self):
        """
        Perform any necessary resets of the model parameters for the MLP components.
        """
        pass

    def get_lora_params(self):
        """
        Return a list of all parameters that would have LoRA for the module. This does not
        refer to the actual LoRA weights themselves, but the parameters that would be fine-tuned
        with LoRA.
        """
        return self.lora_params

    def set_params_wo_copy(self, Z3_enabled=False):
        """
        Rather than copying into, set the parameters directly. This is necessary to provide
        an inexpensive (low-memory-overhead) view onto the FP16 forward weights.
        """
        self.module.mlp.attn_nw = self.attn_nw
        self.module.mlp.attn_nb = self.attn_nb
        self.module.norm_w = self.input_nw
        self.module.norm_b = self.input_nb
        self.set_attn_params_wo_copy(Z3_enabled=Z3_enabled)
        self.set_mlp_params_wo_copy(Z3_enabled=Z3_enabled)

    def set_attn_params_wo_copy(self, Z3_enabled=False):
        """
        Narrower sub-method for finer grained overriding.
        """
        self.module.attention.attn_ow = self.dense_w
        self.module.attention.attn_ob = self.dense_b
        self.module.attention.attn_qkvw = self.qkvw
        self.module.attention.attn_qkvb = self.qkvb

    def set_mlp_params_wo_copy(self):
        """
        Narrower sub-method for finer grained overriding.
        """
        self.module.mlp.inter_w = self._h4h_w
        self.module.mlp.inter_b = self._h4h_b
        self.module.mlp.output_w = self._4hh_w
        self.module.mlp.output_b = self._4hh_b
