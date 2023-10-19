# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.model_implementations.transformers.ds_transformer import DeepSpeedTransformerInference


class DeepSpeedBigCodeInference(DeepSpeedTransformerInference):

    def __init__(self,
                 config,
                 mp_group=None,
                 quantize_scales=None,
                 quantize_groups=1,
                 merge_count=1,
                 mlp_extra_grouping=False):
        super().__init__(config, mp_group, quantize_scales, quantize_groups, merge_count, mlp_extra_grouping)

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

        get_present = (get_present or get_key_value or use_cache)
        is_mqa = self.config.num_kv != self.config.heads and self.config.num_kv == 1

        attention_mask = attention_mask.to(torch.int64) if attention_mask.dtype != torch.int64 else attention_mask

        # if Multi-Query Attention
        if is_mqa:
            attention_mask = attention_mask.squeeze(2).unsqueeze(1)

        outputs = super().forward(input=input,
                                  input_mask=input_mask,
                                  attention_mask=attention_mask,
                                  attn_mask=attn_mask,
                                  head_mask=head_mask,
                                  layer_past=layer_past,
                                  get_key_value=get_key_value,
                                  get_present=get_present,
                                  encoder_output=encoder_output,
                                  enc_dec_attn_mask=enc_dec_attn_mask,
                                  x=x,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=encoder_attention_mask,
                                  use_cache=use_cache,
                                  alibi=alibi,
                                  output_attentions=output_attentions,
                                  **kwargs)

        if get_present:
            outputs = list(outputs)
            (key, value) = outputs[1]
            if is_mqa:
                key = key[:, 0, :, :]
                value = value[:, 0, :, :]
            presents = torch.cat((key, value), dim=-1)
            outputs[1] = presents

        return tuple(outputs)
