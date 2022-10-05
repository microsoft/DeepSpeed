'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import torch
import diffusers


class DSUNet(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        # SD pipeline accesses this attribute
        self.in_channels = unet.in_channels
        self._traced_unet = None
        self._enabled = True
        self.device = self.unet.device
        self.unet.requires_grad_(requires_grad=False)

    def forward(self, sample, timestamp, encoder_hidden_states, return_dict=True):
        if self._enabled:
            if self._traced_unet is None:
                # boosts perf ~10%
                self.unet.to(memory_format=torch.channels_last)

                # force return tuple instead of dict
                self._traced_unet = torch.jit.trace(
                    lambda _sample,
                    _timestamp,
                    _encoder_hidden_states: self.unet(_sample,
                                                      _timestamp,
                                                      _encoder_hidden_states,
                                                      return_dict=False),
                    (sample,
                     timestamp,
                     encoder_hidden_states))
                return self.unet(sample, timestamp, encoder_hidden_states)
            else:
                # convert return type to UNet2DConditionOutput
                out_sample, *_ = self._traced_unet(sample, timestamp, encoder_hidden_states)
                return diffusers.models.unet_2d_condition.UNet2DConditionOutput(
                    out_sample)
        else:
            return self.unet(sample, timestamp, encoder_hidden_states, return_dict)
