# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch


class DSVAE(torch.nn.Module):

    def __init__(self, vae, enable_cuda_graph=True):
        super().__init__()
        self.vae = vae
        self.config = vae.config
        self.device = self.vae.device
        self.dtype = self.vae.dtype
        with torch.no_grad():
            self.vae.eval()
        self.enable_cuda_graph = enable_cuda_graph
        self._decoder_cuda_graph = None
        self._encoder_cuda_graph = None
        self._all_cuda_graph = None
        self.static_decoder_inputs = None
        self.static_decoder_kwargs = None
        self.static_encoder_inputs = None
        self.static_encoder_kwargs = None
        self.static_inputs = None
        self.static_kwargs = None
        self.static_decoder_output = None
        self.static_encoder_output = None
        self.static_output = None

    @torch.no_grad()
    def _graph_replay_decoder(self, *inputs, **kwargs):
        for i in range(len(inputs)):
            if torch.is_tensor(inputs[i]):
                self.static_decoder_inputs[i].copy_(inputs[i])
        for k in kwargs:
            if torch.is_tensor(kwargs[k]):
                self.static_decoder_kwargs[k].copy_(kwargs[k])
        self._decoder_cuda_graph.replay()
        return self.static_decoder_output

    def _decode(self, x, return_dict=True, generator=None):
        return self.vae.decode(x, return_dict=return_dict)

    def _create_cuda_graph_decoder(self, *inputs, **kwargs):
        self._decoder_cuda_graph = torch.cuda.CUDAGraph()

        cuda_stream = torch.cuda.current_stream()
        with torch.cuda.stream(cuda_stream):
            for i in range(3):
                ret = self._decode(*inputs, **kwargs)

        self.static_decoder_inputs = inputs
        self.static_decoder_kwargs = kwargs

        with torch.cuda.graph(self._decoder_cuda_graph):
            self.static_decoder_output = self._decode(*self.static_decoder_inputs, **self.static_decoder_kwargs)

    def decode(self, *inputs, **kwargs):
        if self.enable_cuda_graph:
            if self._decoder_cuda_graph is None:
                self._create_cuda_graph_decoder(*inputs, **kwargs)
            return self._graph_replay_decoder(*inputs, **kwargs)
        else:
            return self._decode(*inputs, **kwargs)

    @torch.no_grad()
    def _graph_replay_encoder(self, *inputs, **kwargs):
        for i in range(len(inputs)):
            if torch.is_tensor(inputs[i]):
                self.static_encoder_inputs[i].copy_(inputs[i])
        for k in kwargs:
            if torch.is_tensor(kwargs[k]):
                self.static_encoder_kwargs[k].copy_(kwargs[k])
        self._encoder_cuda_graph.replay()
        return self.static_encoder_output

    def _encode(self, x, return_dict=True):
        return self.vae.encode(x, return_dict=return_dict)

    def _create_cuda_graph_encoder(self, *inputs, **kwargs):
        self._encoder_cuda_graph = torch.cuda.CUDAGraph()

        cuda_stream = torch.cuda.current_stream()
        with torch.cuda.stream(cuda_stream):
            for i in range(3):
                ret = self._encode(*inputs, **kwargs)

        self.static_encoder_inputs = inputs
        self.static_encoder_kwargs = kwargs

        with torch.cuda.graph(self._encoder_cuda_graph):
            self.static_encoder_output = self._encode(*self.static_encoder_inputs, **self.static_encoder_kwargs)

    def encode(self, *inputs, **kwargs):
        if self.enable_cuda_graph:
            if self._encoder_cuda_graph is None:
                self._create_cuda_graph_encoder(*inputs, **kwargs)
            return self._graph_replay_encoder(*inputs, **kwargs)
        else:
            return self._encode(*inputs, **kwargs)

    @torch.no_grad()
    def _graph_replay(self, *inputs, **kwargs):
        for i in range(len(inputs)):
            if torch.is_tensor(inputs[i]):
                self.static_inputs[i].copy_(inputs[i])
        for k in kwargs:
            if torch.is_tensor(kwargs[k]):
                self.static_kwargs[k].copy_(kwargs[k])
        self._all_cuda_graph.replay()
        return self.static_output

    def forward(self, *inputs, **kwargs):
        if self.enable_cuda_graph:
            if self._all_cuda_graph is None:
                self._create_cuda_graph(*inputs, **kwargs)
            return self._graph_replay(*inputs, **kwargs)
        else:
            return self._forward(*inputs, **kwargs)

    def _create_cuda_graph(self, *inputs, **kwargs):
        self._all_cuda_graph = torch.cuda.CUDAGraph()

        cuda_stream = torch.cuda.current_stream()
        with torch.cuda.stream(cuda_stream):
            for i in range(3):
                ret = self._forward(*inputs, **kwargs)

        self.static_inputs = inputs
        self.static_kwargs = kwargs

        with torch.cuda.graph(self._all_cuda_graph):
            self.static_output = self._forward(*self.static_inputs, **self.static_kwargs)

    def _forward(self, sample, timestamp, encoder_hidden_states, return_dict=True):
        return self.vae(sample, timestamp, encoder_hidden_states, return_dict)
