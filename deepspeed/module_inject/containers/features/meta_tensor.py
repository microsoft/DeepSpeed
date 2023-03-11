'''Copyright The Microsoft DeepSpeed Team'''

from abc import ABC, abstractmethod


class MetaTensorContainer(ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_meta = False
        self.ckpt_load_enabled = True

    def initialize_tensors(self):
        super().initialize_tensors()
        self.is_meta = self.qkvw.is_meta

    def apply_tensor_parallelism(self, mp_replace):
        if self.is_meta:
            if self.qkvb is None:
                self.module.attention.attn_qkvb = None
            if self.dense_b is None:
                self.module.attention.attn_ob = None
        else:
            super().apply_tensor_parallelism(mp_replace)

    def copy_data_to_new_module(self):
        if self.is_meta:
            if self.attn_nw is None:
                self.module.mlp.attn_nw = self.attn_nw
                self.module.mlp.attn_nb = self.attn_nb
        else:
            super().copy_data_to_new_module()

    def transpose(self):
        if not self.is_meta:
            super().transpose()

    @abstractmethod
    def load_params(self, module, sd, weight_quantizer, mp_replace, prefix):
        """
        Load all the transformer parameter from the checkpoint file (sd).
        In addition to the parameter names, we require two
        more parameters to help read the the data correctly
        from the checkpoint and split the qkv heads in the
        right order:
            1. `use_load_prefix` (Default: False): this specifies
                whether we need to use the name of first abstraction
                layer of the model for searching the parameter's name
                in a checkpoint file. For more information of how this
                is used please see
                https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/module_inject/load_checkpoint.py
            2. `split_qkv` (Default: True): we use this flag when splitting
                the qkv parameter into heads. If it is False, it means the heads
                of q, k, and v are stored together and needs to split in the
                DeepSpeed-Inference API.
        """
        raise NotImplementedError(
            "A load_params() function must be defined in the model container \
                                  when inheriting the MetaTensorContainer feature")
