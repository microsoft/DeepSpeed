# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed import comm as dist
from deepspeed.model_implementations.transformers.ds_transformer import DeepSpeedTransformerInference

inference_module = None


class DeepSpeedOPTInference(DeepSpeedTransformerInference):
    """
        Initialize the DeepSpeed OPT Transformer Layer.
    """

    def __init__(self,
                 config,
                 mp_group=None,
                 quantize_scales=None,
                 quantize_groups=1,
                 merge_count=1,
                 mlp_extra_grouping=False):

        super(DeepSpeedOPTInference, self).__init__(config, mp_group, quantize_scales, quantize_groups, merge_count,
                                                    mlp_extra_grouping)

    @classmethod
    def reset_cache(cls):
        if inference_module is not None:
            inference_module.reset_cache()

    def forward(
            self,
            input=None,
            input_mask=None,
            attention_mask=None,
            attn_mask=None,
            head_mask=None,
            layer_past=None,
            get_key_value=False,
            get_present=False,
            encoder_output=None,
            enc_dec_attn_mask=None,
            x=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            alibi=None,
            output_attentions=False,
            # TODO(arashb): 'layer_head_mask' and 'past_key_value' are only added to satisfy the OPT models API.
            # This needs to be redesigned later!
            layer_head_mask=None,
            past_key_value=None,
            **kwargs):

        if x is not None:
            input = x
        if "hidden_states" in kwargs:
            input = kwargs["hidden_states"]

        input_mask = (input_mask if attn_mask is None else attn_mask) if attention_mask is None else attention_mask

        debug = False
        if debug: print(f'ds b4 attn: input = {torch.norm(input)}')
        if debug: print(f'ds b4 attn: input_mask = {torch.norm(input_mask)}')

        # Allocate memory only on first layer forward
        if self.config.layer_id == 0 and self._alloc_workspace:
            self.allocate_workspace(self.config.hidden_size, self.config.heads,
                                    input.size()[1],
                                    input.size()[0], DeepSpeedOPTInference.layer_id, self.config.mp_size,
                                    self.config.bigscience_bloom,
                                    dist.get_rank() if dist.is_initialized() else 0, self.config.max_out_tokens,
                                    self.config.min_out_tokens)
            self._alloc_workspace = False

        get_present = (get_present or get_key_value or use_cache)
        input_mask = input_mask if attention_mask is None else attention_mask

        # We set the prev key/value to None when there is a prompt
        if input.shape[1] > 1:
            self.layer_past = None
        layer_past = layer_past if layer_past is not None else self.layer_past
        head_mask = layer_head_mask if layer_head_mask is not None else head_mask

        attn_mask = None
        if isinstance(input, tuple):
            attn_mask = input[1]
            input = input[0]
        input_type = input.dtype

        if (self.config.dtype in [torch.float16, torch.bfloat16, torch.int8]) \
            and input.dtype == torch.float:
            target_dtype = torch.half if self.dtype == torch.int8 else self.dtype
            input = input.to(target_dtype)

        if debug: print(f'ds b4 attn: norm = {torch.norm(input)}, tensor = {input}')

        # set this to True to use pytorch based attention and mlp
        # (base=False => 2 seconds vs. base=True => 12 seconds on A6000)
        # base = True ==> matches the output of HF model output but base=False does not
        base = True

        with torch.no_grad():
            attention_output, key, value, context_outputtn_ctx, inp_norm = \
                                     self.attention(input,
                                              input_mask,
                                              head_mask,
                                              layer_past,
                                              get_present,
                                              encoder_hidden_states,
                                              encoder_attention_mask,
                                              output_attentions,
                                              self.norm_w,
                                              self.norm_b,
                                              alibi,
                                              attn_base=base)
            if debug: print(f'ds a4 attn + ln: norm = {torch.norm(attention_output)}, tensor = {attention_output}')

            presents = (key, value)
            # Bug? Setting layer past to presents every pass fixes key states issue
            self.layer_past = presents  #if layer_past is None else None

            output = self.mlp(attention_output,
                              input,
                              inp_norm,
                              self.attention.attn_ob,
                              self.attention.attn_ow,
                              mlp_base=base)

            if debug: print(f"after mlp: {torch.norm(output)}")
            #exit(0)
            if not self.config.pre_layer_norm:
                output = inference_module.layer_norm(output, self.norm_w, self.norm_b, self.config.epsilon)
            if debug: print(f"after layernorm: {torch.norm(output)}")
            # if self.config.layer_id == 1:
            #   exit(0)
            #import pdb; pdb.set_trace()
            output = output.to(input_type)
        if get_present:
            output = (output, presents)

        if self.config.return_single_tuple:
            return (output, )
        elif self.config.return_tuple:
            return output if type(output) is tuple else (output, attn_mask)
        else:
            return output
