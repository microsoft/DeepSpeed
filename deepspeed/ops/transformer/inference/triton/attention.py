# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math
import torch
import torch.nn as nn
import triton
import triton.language as tl
from deepspeed.accelerator import get_accelerator
from deepspeed import comm as dist
from deepspeed.ops.transformer.inference.op_binding import LinearOp, VectorMatMulOp, SoftmaxContextOp, QKVGemmOp
from deepspeed.ops.transformer.inference.triton import (
    softmax,
    score_4d_matmul,
    context_4d_matmul,
)

minus_inf = -10000.0


class TritonSelfAttention(nn.Module):
    num_layers = 0

    def __init__(self, config, mp_group=None, q_scales=None, q_groups=1, merge_count=1, qkv_merging=False):
        super(TritonSelfAttention, self).__init__()
        self.config = config
        data_type = self.config.dtype
        data_type_fp = torch.half if self.config.dtype == torch.int8 else self.config.dtype
        assert data_type_fp == torch.half, "triton supports fp16 data_type_fp"

        self.config.layer_id = TritonSelfAttention.num_layers
        TritonSelfAttention.num_layers = TritonSelfAttention.num_layers + 1
        device = get_accelerator().current_device_name()  #if config.bigscience_bloom else 'cpu'

        assert config.mp_size == 1, "mp_size has to be 1 with triton attention yet"
        if self.config.set_empty_params:
            self.attn_qw = None
            self.attn_qb = None
            self.attn_kw = None
            self.attn_kb = None
            self.attn_vw = None
            self.attn_vb = None
            self.attn_qkvw = None
            self.attn_qkvb = None
            self.attn_ow = None
            self.attn_ob = None
        else:
            qkv_size_per_partition = (self.config.hidden_size // self.config.mp_size) * 3
            self.attn_qkvw = nn.Parameter(torch.empty(self.config.hidden_size,
                                                      qkv_size_per_partition,
                                                      dtype=data_type,
                                                      device=device),
                                          requires_grad=False)
            self.attn_qkvb = nn.Parameter(torch.empty(qkv_size_per_partition, dtype=data_type_fp, device=device),
                                          requires_grad=False)
            # self-ouput weights
            out_size_per_partition = self.config.hidden_size // self.config.mp_size
            self.attn_ow = nn.Parameter(torch.empty(out_size_per_partition,
                                                    self.config.hidden_size,
                                                    dtype=data_type,
                                                    device=device),
                                        requires_grad=False)

            self.attn_ob = nn.Parameter(torch.empty(self.config.hidden_size, dtype=data_type_fp, device=device),
                                        requires_grad=False)

        self.num_attention_heads_per_partition = self.config.heads // self.config.mp_size
        self.hidden_size_per_partition = self.config.hidden_size // self.config.mp_size
        self.hidden_size_per_attention_head = self.config.hidden_size // self.config.heads

        self.mp_group = mp_group
        self.use_flash = False

        # used for quantization
        self.q_scales = q_scales
        self.q_groups = q_groups
        self.merge_count = int(math.log2(merge_count))

        self.norm_factor = math.sqrt(self.config.hidden_size // self.config.heads)
        if not config.use_mup:
            self.norm_factor = math.sqrt(self.norm_factor)

        if self.config.scale_attn_by_inverse_layer_idx is True:
            self.norm_factor *= math.sqrt(self.config.layer_id + 1)
            # https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/models/gpt2/modeling_gpt2.py#L191

        triton_autotune = self.config.triton_autotune and self.config.layer_id == 0
        self.qkv_func = QKVGemmOp(config)
        self.score_context_func = SoftmaxContextOp(config)
        self.linear_func = LinearOp(config)
        self.vector_matmul_func = VectorMatMulOp(config)

        self.hidden_size = config.hidden_size
        self.head_size = config.hidden_size // config.heads
        self.scale = (1 / self.norm_factor / self.norm_factor if self.config.scale_attention else 1.0
                      )  # making it back to 1/sqrt(head_size)
        self.triangular_masking = self.config.triangular_masking

        # triton autotune table update for score/context matmul
        if triton_autotune:
            print(f"running triton autotune for attention")
            __class__._triton_autotune(2, self.config.max_out_tokens, self.head_size, self.config.hidden_size,
                                       self.triangular_masking, self.scale)

    @staticmethod
    def _triton_autotune(min_seqlen,
                         max_seqlen,
                         head_size,
                         hidden_size,
                         triangular_masking,
                         scale,
                         dtype=torch.float16):
        from deepspeed.ops.transformer.inference.triton.matmul_ext import Fp16Matmul, score_4d_matmul, context_4d_matmul
        seqlen = [(min_seqlen + i)
                  for i in range(0, max_seqlen - min_seqlen + Fp16Matmul._cache_stride + 1, Fp16Matmul._cache_stride)]
        Fp16Matmul._read_autotune_table()
        for N in seqlen:
            qkv = torch.randn((1, N, 3 * hidden_size), dtype=dtype, device='cuda')
            output = score_4d_matmul(qkv, head_size, triangular_masking, scale)
            context_4d_matmul(output, qkv, head_size)
        Fp16Matmul._update_autotune_table()

    def ds_compute_attention(self, qkv_out, input_mask, layer_past, alibi):
        if isinstance(qkv_out, list):
            qkv_out = qkv_out[0]

        no_masking = input_mask is None

        if no_masking:
            input_mask = torch.empty(1)

        attn_key_value = self.score_context_func(
            query_key_value=qkv_out,
            attn_mask=((1 - input_mask).to(qkv_out.dtype) *
                       minus_inf) if input_mask.dtype == torch.int64 else input_mask,
            heads=self.num_attention_heads_per_partition,
            norm_factor=(1 / self.norm_factor if self.config.scale_attention else 1.0),
            no_masking=no_masking,
            layer_id=self.config.layer_id,
            num_layers=TritonSelfAttention.num_layers,
            alibi=alibi)

        context_layer, key_layer, value_layer = attn_key_value
        return context_layer, key_layer, value_layer

    def forward(
            self,
            input,
            input_mask,
            head_mask=None,
            layer_past=None,
            get_present=False,  # not used
            encoder_hidden_states=None,  # not used
            encoder_attention_mask=None,  # not used
            triangularutput_attentions=False,  # not used
            norm_w=None,
            norm_b=None,
            alibi=None,
            use_triton_attention=True):

        if not self.config.pre_layer_norm:
            qkv_out = self.linear_func(input=input,
                                       weight=self.attn_qkvw,
                                       bias=self.attn_qkvb,
                                       add_bias=self.attn_qkvb is not None,
                                       do_flash_attn=False,
                                       num_heads=self.num_attention_heads_per_partition,
                                       num_layers=TritonSelfAttention.num_layers)
            qkv = qkv_out
        else:
            qkv_out = self.qkv_func(input=input,
                                    weight=self.attn_qkvw,
                                    bias=(self.attn_qkvb if self.attn_qkvb is not None else norm_b),
                                    gamma=norm_w,
                                    beta=norm_b)
            qkv = qkv_out[0]

        if use_triton_attention and (alibi is None):
            context_layer = compute_attention(qkv=qkv,
                                              input_mask=input_mask,
                                              scale=self.scale,
                                              layer_past=layer_past,
                                              alibi=alibi,
                                              head_size=self.head_size,
                                              use_triton_flash=self.use_flash,
                                              use_cuda_flash=False,
                                              triangular=self.triangular_masking)
            key_layer, value_layer = qkv[:, :, self.hidden_size:2 * self.hidden_size], qkv[:, :, 2 * self.hidden_size:]
        else:
            context_layer, key_layer, value_layer = self.ds_compute_attention(qkv_out=qkv_out,
                                                                              input_mask=input_mask,
                                                                              layer_past=layer_past,
                                                                              alibi=alibi)
        output = self.vector_matmul_func(input=context_layer, weight=self.attn_ow)

        inp_norm = qkv_out[-1]

        if self.config.mlp_after_attn and self.mp_group is not None and dist.get_world_size(group=self.mp_group) > 1:
            dist.all_reduce(output, group=self.mp_group)

        return (output, key_layer, value_layer, context_layer, inp_norm)


global inference_module

def compute_attention(qkv,
                      input_mask,
                      layer_past,
                      alibi,
                      scale,
                      head_size,
                      triangular=False,
                      use_cuda_flash=False,
                      use_triton_flash=False,
                      use_ds_attention=False):
    if isinstance(qkv, list):
        qkv = qkv[0]

    #assert layer_past is None, "layer_past not supported in triton yet"
    assert alibi is None, "layer_past not supported in alibi yet"
    output = score_4d_matmul(qkv, head_size, triangular, scale)
    if triangular:
        output = softmax(output)
    else:
        output = softmax(output, input_mask)
    output = context_4d_matmul(output, qkv, head_size)

    return output


@triton.jit
def _flash_unpacked_kernel(
    Q, K, V,
        mask,
        ADD_MASK: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    sm_scale,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
        stride_mh,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    P_SEQ,
        hidden_size,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    batch = off_hz // H
    head = off_hz % H

    q_offset = off_hz * stride_qh
    kv_offset = off_hz * stride_kh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(BLOCK_DMODEL, N_CTX + P_SEQ),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N_CTX + P_SEQ, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # mask
    if ADD_MASK:
        off_mask = batch * stride_mh + offs_n[None, :]
        mask_ptrs = mask + off_mask

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(tl.float16)
    # loop over k, v and update accumulator
    lo = 0
    hi = P_SEQ + (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX + P_SEQ
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float16)

        if ADD_MASK:
            mask_val = tl.load(mask_ptrs)
            mask_ptrs += BLOCK_N
            qk = qk + mask_val.to(tl.float32)

        if IS_CAUSAL:
            qk = tl.where(P_SEQ + offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

        qk += tl.dot(q, k, out_dtype=tl.float16)
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(tl.float16), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    # write back l and m
    acc = acc / l_i[:, None]
    # l_ptrs = L + off_hz * N_CTX + offs_m
    # tl.store(l_ptrs, m_i + tl.math.log2(l_i))
    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(tl.float16))


def _triton_unpacked_forward(
            q, k, v,
            heads,
            mask,
            sm_scale,
            causal=False, add_mask=True):
    BLOCK = 64
    head_size = q.shape[3]

    BLOCK_M = 128
    BLOCK_N = 64 if head_size <= 64 else 32

    # o = torch.empty((qkv.shape[0],
    #                  qkv.shape[1],
    #                  heads * head_size),
    #                 device=qkv.device,
    #                 dtype=torch.int8 if self.int8_output else torch.half)
    o = torch.empty_like(q)

    # grid = (triton.cdiv(qkv.shape[1], BLOCK), qkv.shape[0] * heads, 1)
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    # tmp = torch.empty((qkv.shape[0] * heads,
    #                    qkv.shape[1]),
    #                   device=qkv.device,
    #                   dtype=torch.float32)
    tmp = torch.empty(0)
    # num_warps = 4 if head_size <= 64 else 8
    num_stages = 4 if head_size <= 64 else 3
    num_warps = 4
    P_SEQ = 0 if q.shape[-2] == k.shape[-2] else k.shape[-2] - q.shape[-2]

    _flash_unpacked_kernel[grid](q,k,v,
                    mask,
                    add_mask,
                    causal,
                    sm_scale,
                    o,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    0 if mask is None else mask.stride(1),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    o.stride(0), o.stride(1), o.stride(2), o.stride(3),

                    q.shape[0], q.shape[1], q.shape[2],
                    P_SEQ,

                    q.shape[1] * q.shape[3],

                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_DMODEL=head_size,
                    num_warps=num_warps,
                    num_stages=num_stages)

    return o


def _triton_packed_forward(
            qkv,
            heads,
            mask,
            sm_scale,
            causal=False, add_mask=True):
    head_size = qkv.shape[-1] // 3 // heads
    hidden_size = qkv.shape[-1] // 3

    BLOCK_M = 128
    BLOCK_N = 64 if head_size <= 64 else 32

    # o = torch.empty((qkv.shape[0], heads, qkv.shape[1], head_size),
    #                 device=qkv.device,
    #                 dtype=torch.int8 if self.int8_output else torch.half)

    o = torch.empty((qkv.shape[0], qkv.shape[1], hidden_size),
                    device=qkv.device,
                    dtype=torch.half)
                    # dtype=torch.half)
    if mask is None:
        mask = torch.empty(0)
        add_mask = False

    # grid = (triton.cdiv(qkv.shape[1], BLOCK), qkv.shape[0] * heads, 1)
    grid = (triton.cdiv(qkv.shape[1], BLOCK_M), qkv.shape[0] * heads, 1)
    # tmp = torch.empty((qkv.shape[0] * heads,
    #                    qkv.shape[1]),
    #                   device=qkv.device,
    #                   dtype=torch.float32)
    tmp = torch.empty(0)
    # num_warps = 4 if head_size <= 64 else 8
    num_stages = 4 if head_size <= 64 else 3
    num_warps = 4
    P_SEQ = 0

    _flash_packed_kernel[grid](qkv,
                    mask,
                    add_mask,
                    causal,
                    sm_scale,
                    o,
                    qkv.stride(0), qkv.stride(1), qkv.stride(2),
                    mask.stride(1) if add_mask else 0,
                    qkv.stride(0), qkv.stride(1), qkv.stride(2),
                    qkv.stride(0), qkv.stride(1), qkv.stride(2),
                    o.stride(0), o.stride(1), o.stride(2),

                    qkv.shape[0], heads, qkv.shape[1], P_SEQ,
                    hidden_size,

                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_DMODEL=head_size,
                    num_warps=num_warps,
                    num_stages=num_stages)

    return o


@triton.jit
def _flash_packed_kernel(
    QKV,
        mask,
        ADD_MASK: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    sm_scale,
    # L,
    Out,
    stride_qz, stride_qh, stride_qm,
        stride_mh,
    stride_kz, stride_kh, stride_kn,
    stride_vz, stride_vh, stride_vk,
    stride_oz, stride_oh, stride_om,
    Z, H, N_CTX, P_SEQ,
        hidden_size,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    batch = off_hz // H
    head = off_hz % H

    q_offset = batch * stride_qz + head * BLOCK_DMODEL
    k_offset = q_offset + hidden_size
    v_offset = k_offset + hidden_size
    Q_block_ptr = tl.make_block_ptr(
        base=QKV + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qh, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=QKV + k_offset,
        shape=(BLOCK_DMODEL, N_CTX + P_SEQ),
        strides=(1, stride_qh),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=QKV + v_offset,
        shape=(N_CTX + P_SEQ, BLOCK_DMODEL),
        strides=(stride_qh, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # mask
    off_mask = batch * stride_mh + offs_n[None, :]
    mask_ptrs = mask + off_mask

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    # q = tl.load(Q_block_ptr)
    # q = (q * qk_scale).to(tl.float16)
    q = tl.load(Q_block_ptr)
    # loop over k, v and update accumulator
    lo = 0
    hi = P_SEQ + (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX + P_SEQ
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float16)

        if ADD_MASK:
            mask_val = tl.load(mask_ptrs)
            mask_ptrs += BLOCK_N
            qk = qk + mask_val.to(tl.float32)

        if IS_CAUSAL:
            qk = tl.where(P_SEQ + offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

        qk += tl.dot(q, k, out_dtype=tl.float16) * qk_scale
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(tl.float16), v.to(tl.float16)) # loading q,k and v in int8 gives incorrect results, looks like triton bug
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    # write back l and m
    acc = acc / l_i[:, None]
    # acc = attn_scale * acc
    # l_ptrs = L + off_hz * N_CTX + offs_m
    # tl.store(l_ptrs, m_i + tl.math.log2(l_i))
    # write back O
    # o_offset = off_hz * stride_oh
    o_offset = batch * stride_oz + head * BLOCK_DMODEL
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_oh, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(tl.float16))



def assert_almost_equal(x, y, decimal=2, err_msg=''):
    import numpy.testing as npt
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            x = x.float()
        x = x.cpu().detach().numpy()
    if isinstance(y, torch.Tensor):
        if y.dtype == torch.bfloat16:
            y = y.float()
        y = y.cpu().detach().numpy()
    npt.assert_array_almost_equal(x, y, err_msg=err_msg, decimal=decimal)


def max_diff(a, b):
    a = a.to(torch.float32).flatten()
    b = b.to(torch.float32).flatten()
    diff = torch.abs(a - b)
    max_diff_indices = torch.argsort(diff)[-1]
    print("Max difference indices:", max_diff_indices)
    print("Max difference values:", diff[max_diff_indices])
    print(f"{a[max_diff_indices]} vs {b[max_diff_indices]}")
    return max_diff_indices

# reference implementation
def ref_torch_attention(q, k, v, mask, sm_scale, verbose=True):
    if verbose:
        print(f"ref_torch_attention:q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}, mask.shape={mask.shape if mask is not None else 0}")
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale

    if verbose:
        print(f"ref_torch_attention:qk ={q.shape} x {k.transpose(2, 3).shape}")
    p = p.float()
    if mask is not None:
        p = p.float() + mask
    p = torch.softmax(p, dim=-1).type(q.dtype)

    ref_out = torch.matmul(p, v)
    if verbose:
        print(f"ref_torch_attention:context = {ref_out.shape} = {p.shape} x {v.shape}")
    return ref_out


# test attention operator
def test_attention(Z, H, N_CTX, D_HEAD, dtype=torch.float16):
    print(f"Z={Z}, H={H}, N_CTX={N_CTX}, D_HEAD={D_HEAD}")
    # skip autotune in testing
    from deepspeed.ops.transformer.inference.triton.matmul_ext import fp16_matmul
    fp16_matmul.skip_autotune()

    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0, std=.5)
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0, std=.5)
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0, std=.5)
    sm_scale = 0.3

    # reference implementation
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    score = p
    mask = torch.zeros((Z, H, N_CTX, N_CTX), dtype=dtype, device="cuda")
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    for z in range(Z):
        for h in range(H):
            mask[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float() + mask, dim=-1).half()
    softmax_out = p
    ref_out = torch.matmul(p, v)
    context = ref_out

    # adjust it to expected tensor format and run test
    qkv = torch.randn((Z, N_CTX, 3 * H * D_HEAD), dtype=dtype, device='cuda', requires_grad=False)
    qkv[:, :, :H * D_HEAD] = q.permute(0, 2, 1, 3).contiguous().reshape((Z, N_CTX, H * D_HEAD))
    qkv[:, :, 1 * H * D_HEAD:2 * H * D_HEAD] = k.permute(0, 2, 1, 3).contiguous().reshape((Z, N_CTX, H * D_HEAD))
    qkv[:, :, 2 * H * D_HEAD:] = v.permute(0, 2, 1, 3).contiguous().reshape((Z, N_CTX, H * D_HEAD))
    tri_out = compute_attention(qkv,
                                input_mask=mask,
                                layer_past=None,
                                alibi=None,
                                scale=sm_scale,
                                head_size=D_HEAD,
                                triangular=False,
                                use_cuda_flash=False,
                                use_triton_flash=False,
                                use_ds_attention=False)
    tri_out = tri_out.reshape((Z, N_CTX, H, D_HEAD)).permute(0, 2, 1, 3)
    assert_almost_equal(ref_out, tri_out)


    if True:
        ##############
        # triton 2.0 flash attn check in float16, ref attention from torch
        BATCH = Z
        q = torch.empty((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0, std=.5)
        k = torch.empty((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0, std=.5)
        v = torch.empty((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0, std=.5)
        sm_scale = 1 / math.sqrt(D_HEAD)

        qkv = torch.randn((BATCH, N_CTX, 3 * H * D_HEAD), dtype=torch.float16, device='cuda', requires_grad=False)
        qkv[:,:,:H * D_HEAD] = q.permute(0,2,1,3).contiguous().reshape((BATCH,N_CTX,H*D_HEAD))
        qkv[:,:,1 * H * D_HEAD: 2 * H * D_HEAD] = k.permute(0,2,1,3).contiguous().reshape((BATCH,N_CTX,H*D_HEAD))
        qkv[:,:,2 * H * D_HEAD:] = v.permute(0,2,1,3).contiguous().reshape((BATCH,N_CTX,H*D_HEAD))

        batch_size = BATCH
        nheads = H
        seqlen = N_CTX
        d = D_HEAD
        lengths = torch.randint(seqlen - 8, seqlen, (batch_size, 1), device='cuda')
        triton_mask = torch.zeros((BATCH, 1, 1, N_CTX), dtype=dtype, device="cuda")
        for i, l in enumerate(lengths):
            triton_mask[i,...,l:] = minus_inf
        mask = torch.zeros((BATCH, H, N_CTX, N_CTX), dtype=dtype, device="cuda")
        for b in range(batch_size):
            mask[b,:,:,lengths[b]:] = minus_inf

        causal_mask = torch.zeros((BATCH, H, N_CTX, N_CTX), dtype=dtype, device="cuda")
        M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
        for z in range(BATCH):
            for h in range(H):
                causal_mask[:, :, M == 0] = float("-inf")

        # ref_out = ref_torch_attention(q, k, v, causal_mask, sm_scale, verbose=False).reshape(BATCH, H, N_CTX, D_HEAD)
        # from triton.ops.flash_attention import attention
        # tri_out = attention(q, k, v, True, sm_scale) # always causal
        # print(f"ref_out={ref_out}")
        # print(f"tri_out={tri_out}")
        # assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
        # print(f"PASSED: triton2.0-flash, fp16, causal")

        # print(f"triton_mask={triton_mask.shape}, {triton_mask.stride()}, {triton_mask}")
        # ref_out = ref_torch_attention(q, k, v, causal_mask, sm_scale, verbose=False).permute(0,2,1,3).reshape(BATCH, N_CTX, H * D_HEAD)
        # ref_out = ref_torch_attention(q, k, v, mask, sm_scale, verbose=False).permute(0,2,1,3).reshape(BATCH, N_CTX, H * D_HEAD)
        # tri_out = _triton_flash_fwd(qkv, H, mask=None, sm_scale=sm_scale, causal=True, add_mask=False)

        ref_out = ref_torch_attention(q, k, v, causal_mask, sm_scale, verbose=False).reshape(BATCH, H, N_CTX, D_HEAD)
        tri_out = _triton_unpacked_forward(q, k, v, H, mask=None, sm_scale=sm_scale, causal=True, add_mask=False)

        # print(f"triton_mask={triton_mask}, mask={mask}, {mask[:,:,:,-6:]}")
        # print(f"ref_out={ref_out[0,0,:4,:4]}, tri_out={tri_out[0,0,:4,:4]}")
        # print(f"triton_mask={triton_mask}, mask={mask}, {mask[:,:,:,-6:]}")
        # print(f"ref_out={ref_out.shape}, tri_out={tri_out.shape}, mask={mask.shape}")
        assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
        print(f"PASSED: triton2.0-flash, unpacked, causal")

        ref_out = ref_torch_attention(q, k, v, causal_mask, sm_scale, verbose=False).permute(0,2,1,3).reshape(BATCH, N_CTX, H * D_HEAD)
        tri_out = _triton_packed_forward(qkv, H, mask=None, sm_scale=sm_scale, causal=True, add_mask=False)
        assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
        print(f"PASSED: triton2.0-flash, packed, causal")

        ref_out = ref_torch_attention(q, k, v, mask, sm_scale, verbose=False).permute(0,2,1,3).reshape(BATCH, N_CTX, H * D_HEAD)
        tri_out = _triton_packed_forward(qkv, H, mask=triton_mask, sm_scale=sm_scale, causal=False, add_mask=True)
        # print(f"triton_mask={triton_mask}, mask={mask}, lengths={lengths}")
        max_diff(ref_out, tri_out)
        # print(f"ref_out={ref_out}")
        # print(f"tri_out={tri_out}")
        assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
        print(f"PASSED: triton2.0-flash, packed, masked")


    # flash_tri_out = _triton_flash_fwd(qkv, H, mask, sm_scale, causal=False, add_mask=(not mask is None))
    # print(f"ref_out={ref_out}")
    # print(f"flash_tri_out={flash_tri_out}")
    # assert_almost_equal(ref_out, flash_tri_out)

test_attention(Z=1, H=2, N_CTX=128, D_HEAD=128, dtype=torch.float16)
test_attention(Z=4, H=12, N_CTX=128, D_HEAD=64, dtype=torch.float16)
seqlen = [32, 55, 71, 128]
for s in seqlen:
    test_attention(Z=4, H=12, N_CTX=128, D_HEAD=64, dtype=torch.float16)