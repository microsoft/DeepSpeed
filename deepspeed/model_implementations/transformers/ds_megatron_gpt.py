'''
Copyright 2022 The Microsoft DeepSpeed Team
'''

from deepspeed.model_implementations.transformers.ds_transformer import DeepSpeedTransformerInference


class DeepSpeedMegatronGPTInference(DeepSpeedTransformerInference):
    """Initialize the DeepSpeed Megatron GPT Transformer Layer.
    """

    def __init__(self,
                 config,
                 mp_group=None,
                 quantize_scales=None,
                 quantize_groups=1,
                 merge_count=1,
                 mlp_extra_grouping=False):
        super().__init__(config, mp_group, quantize_scales, quantize_groups, merge_count, mlp_extra_grouping)
