# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from abc import abstractmethod
import torch

from .hybrid_engine import HybridEngineContainer


class HybridSplitQKVContainer(HybridEngineContainer):

    def set_attention(self, qkvw, qkvb, dense_w, dense_b):
        super().set_attention(qkvw, qkvb, dense_w, dense_b)
        self.set_q_k_v()

    @abstractmethod
    def set_q_k_v(self):
        """
        In `set_q_k_v`, it is necessary to populate the following variables (where appropriate)
        for the given model:
            self.qw: q weight
            self.qb: q bias
            self.kw: k weight
            self.kb: k bias
            self.vw: v weight
            self.vb: v bias
        """
        raise NotImplementedError("A set_q_k_v() function must be defined in the model container \
                                    in order to set the unfused q, k, and v tensors.")

    def attention_qkv_mp(self, mp_replace, reversed_dim=False):
        if self.module.attention.attn_qkvw is None:
            params = [
                (self.module.attention.attn_qw.self.qw),
                (self.module.attention.attn_qb.self.qb),
                (self.module.attention.attn_kw.self.kw),
                (self.module.attention.attn_kb.self.kb),
                (self.module.attention.attn_vw.self.vw),
                (self.module.attention.attn_vb.self.vb),
            ]
            for dst, src in params:
                dst = mp_replace.copy(dst[:self.qw.shape[0] // mp_replace.mp_size],
                                      src,
                                      int8=reversed_dim,
                                      allocat_tensor=reversed_dim)
        else:
            self.module.attention.attn_qkvw = mp_replace.strided_copy(self.module.attention.attn_qkvw,
                                                                      self.qkvw,
                                                                      num_splits=3,
                                                                      int8=reversed_dim)
            self.module.attention.attn_qkvb = mp_replace.strided_copy(self.module.attention.attn_qkvb,
                                                                      self.qkvb,
                                                                      num_splits=3,
                                                                      int8=reversed_dim)

    def release_qkv(self):
        super().release_qkv()
        split_qkv_params = [
            (self.module.attention.attn_qw, self.qw),
            (self.module.attention.attn_qb, self.qb),
            (self.module.attention.attn_kw, self.kw),
            (self.module.attention.attn_kb, self.kb),
            (self.module.attention.attn_vw, self.vw),
            (self.module.attention.attn_vb, self.vb),
        ]

        self._release_params(split_qkv_params)

    def reset_qkv(self):
        self.qkvw.data[:self.qw.shape[0]] = self.qw.data
        self.qkvb.data[:self.qw.shape[0]] = self.qb.data
        self.qkvw.data[self.qw.shape[0]:2 * self.qw.shape[0]] = self.kw.data
        self.qkvb.data[self.qw.shape[0]:2 * self.qw.shape[0]] = self.kb.data
        self.qkvw.data[2 * self.qw.shape[0]:] = self.vw.data
        self.qkvb.data[2 * self.qw.shape[0]:] = self.vb.data

        qkv_data = [self.qw.data, \
                    self.qb.data, \
                    self.kw.data, \
                    self.kb.data, \
                    self.vw.data, \
                    self.vb.data]

        self.qw.data = self.qkvw.data[:self.qw.shape[0]]
        self.qb.data = self.qkvb.data[:self.qw.shape[0]]
        self.kw.data = self.qkvw.data[self.qw.shape[0]:2 * self.qw.shape[0]]
        self.kb.data = self.qkvb.data[self.qw.shape[0]:2 * self.qw.shape[0]]
        self.vw.data = self.qkvw.data[2 * self.qw.shape[0]:]
        self.vb.data = self.qkvb.data[2 * self.qw.shape[0]:]

        for data in qkv_data:
            del data

    def reset_qkv_experimental(self):
        """
        WIP - experimental and likely to be changed/improved.
        Unused by keeping for now.
        """
        if self.module.attention.attn_qkvw is None:
            self.module.attention.attn_qkvw = torch.empty(self.qw.shape[0] * 3,
                                                          self.qw.shape[0],
                                                          dtype=self.qw.dtype,
                                                          device=self.qw.device)
            self.module.attention.attn_qkvb = torch.empty(self.qw.shape[0] * 3,
                                                          dtype=self.qw.dtype,
                                                          device=self.qw.device)
        self.module.attention.attn_qkvw.data[:self.qw.shape[0]] = self.qw.data
        self.module.attention.attn_qkvb.data[:self.qw.shape[0]] = self.qb.data
        self.module.attention.attn_qkvw.data[self.qw.shape[0]:2 * self.qw.shape[0]] = self.kw.data
        self.module.attention.attn_qkvb.data[self.qw.shape[0]:2 * self.qw.shape[0]] = self.kb.data
        self.module.attention.attn_qkvw.data[2 * self.qw.shape[0]:] = self.vw.data
        self.module.attention.attn_qkvb.data[2 * self.qw.shape[0]:] = self.vb.data

        qkv_data = [self.qw.data, \
                    self.qb.data, \
                    self.kw.data, \
                    self.kb.data, \
                    self.vw.data, \
                    self.vb.data]

        self.qw.data = self.module.attention.attn_qkvw.data[:self.qw.shape[0]]
        self.qb.data = self.module.attention.attn_qkvb.data[:self.qw.shape[0]]
        self.kw.data = self.module.attention.attn_qkvw.data[self.qw.shape[0]:2 * self.qw.shape[0]]
        self.kb.data = self.module.attention.attn_qkvb.data[self.qw.shape[0]:2 * self.qw.shape[0]]
        self.vw.data = self.module.attention.attn_qkvw.data[2 * self.qw.shape[0]:]
        self.vb.data = self.module.attention.attn_qkvb.data[2 * self.qw.shape[0]:]

        for data in qkv_data:
            del data

    def set_attn_parameters_wo_copy(self, Z3_enabled=False):
        if not Z3_enabled:
            self.module.attn_qkvw = self.qkvw
            self.module.attn_qkvb = self.qkvb
            self.qw.data = self.qkvw[:self.qw.shape[0], :]
            self.qb.data = self.qkvb[:self.qw.shape[0]]
            self.kw.data = self.qkvw[self.qw.shape[0]:2 * self.qw.shape[0], :]
            self.kb.data = self.qkvb[self.qw.shape[0]:2 * self.qw.shape[0]]
            self.vw.data = self.qkvw[self.qw.shape[0] * 2:, :]
            self.vb.data = self.qkvb[self.qw.shape[0] * 2:]
        else:
            self.module.attention.attn_qw = self.qw
            self.module.attention.attn_qb = self.qb
            self.module.attention.attn_kw = self.kw
            self.module.attention.attn_kb = self.kb
            self.module.attention.attn_vw = self.vw
            self.module.attention.attn_vb = self.vb

    def get_attn_params(self):
        params = super().get_attn_params()
        params.extend([self.qw, self.qb, self.kw, self.kb, self.vw, self.vb])
        return params
