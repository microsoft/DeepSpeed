# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Triton kernels for the vocabulary-parallel log-probability, entropy and KL ops.

Each kernel handles one row of a ``[rows, local_vocab]`` logits shard and loops over the local
vocabulary in blocks, so no ``[rows, local_vocab]`` fp32 temporary is materialized. The kernels
only see the local shard; ``deepspeed.sequence.vocab_parallel_ops`` owns the all-reduces and the
autograd functions, and keeps an eager fallback with the same semantics.

``global_max`` and ``log_sum_exp`` are the global per-row softmax statistics, kept separate as in
the eager path, and a log-probability is ``(z - global_max) - log_sum_exp``.
"""

import torch

from deepspeed.ops.triton_ops._triton import _TRITON_AVAILABLE, triton, tl

_MAX_BLOCK = 4096
_NUM_WARPS = 16

if _TRITON_AVAILABLE:

    @triton.jit
    def _log_probs(ptr, cols, mask, global_max, log_sum_exp):
        z = tl.load(ptr + cols, mask=mask, other=float("-inf")).to(tl.float32)
        return (z - global_max) - log_sum_exp

    @triton.jit
    def _local_softmax_stats_kernel(x_ptr, max_ptr, sum_ptr, vocab, stride, BLOCK: tl.constexpr):
        # A single read computes the local max together with the sum-exp relative to it, by
        # rescaling each lane's running sum whenever its running max grows.
        row = tl.program_id(0).to(tl.int64)
        x_ptr += row * stride
        lane_max = tl.full([BLOCK], float("-inf"), tl.float32)
        lane_sum = tl.zeros([BLOCK], tl.float32)
        for start in range(0, vocab, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            z = tl.load(x_ptr + cols, mask=cols < vocab, other=float("-inf")).to(tl.float32)
            new_max = tl.maximum(lane_max, z)
            # exp(-inf - -inf) is NaN; a lane that has only seen -inf keeps a zero sum.
            empty = new_max == float("-inf")
            lane_sum = tl.where(empty, 0.0, lane_sum * tl.exp(lane_max - new_max) + tl.exp(z - new_max))
            lane_max = new_max
        row_max = tl.max(lane_max, axis=0)
        row_sum = tl.sum(tl.where(lane_max == float("-inf"), 0.0, lane_sum * tl.exp(lane_max - row_max)), axis=0)
        tl.store(max_ptr + row, row_max)
        tl.store(sum_ptr + row, row_sum)

    @triton.jit
    def _entropy_forward_kernel(x_ptr, max_ptr, lse_ptr, out_ptr, vocab, stride, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        x_ptr += row * stride
        global_max = tl.load(max_ptr + row)
        log_sum_exp = tl.load(lse_ptr + row)
        acc = tl.zeros([BLOCK], tl.float32)
        for start in range(0, vocab, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            log_p = _log_probs(x_ptr, cols, cols < vocab, global_max, log_sum_exp)
            # Masked tokens contribute nothing; 0 * -inf would otherwise be NaN.
            acc += tl.where(log_p == float("-inf"), 0.0, tl.exp(log_p) * log_p)
        tl.store(out_ptr + row, -tl.sum(acc, axis=0))

    @triton.jit
    def _entropy_backward_kernel(x_ptr, max_ptr, lse_ptr, entropy_ptr, grad_out_ptr, grad_ptr, vocab, stride,
                                 BLOCK: tl.constexpr):
        # dH / dz_j = -p_j * (log p_j + H)
        row = tl.program_id(0).to(tl.int64)
        x_ptr += row * stride
        grad_ptr += row * stride
        global_max = tl.load(max_ptr + row)
        log_sum_exp = tl.load(lse_ptr + row)
        entropy = tl.load(entropy_ptr + row)
        grad_out = tl.load(grad_out_ptr + row)
        undefined_row = global_max == float("-inf")
        for start in range(0, vocab, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            mask = cols < vocab
            log_p = _log_probs(x_ptr, cols, mask, global_max, log_sum_exp)
            grad = tl.where(log_p == float("-inf"), 0.0, tl.exp(log_p) * (log_p + entropy)) * -grad_out
            grad = tl.where(undefined_row, 0.0, grad)
            tl.store(grad_ptr + cols, grad.to(grad_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _logprobs_backward_kernel(x_ptr, max_ptr, lse_ptr, grad_sum_ptr, zero_row_ptr, index_ptr, index_grad_ptr,
                                  grad_ptr, vocab, stride, num_index, BLOCK: tl.constexpr):
        # d logp(i) / d z_j = onehot(i)_j - softmax(z)_j, summed over the requested indices.
        row = tl.program_id(0).to(tl.int64)
        x_ptr += row * stride
        grad_ptr += row * stride
        index_ptr += row * num_index
        index_grad_ptr += row * num_index
        global_max = tl.load(max_ptr + row)
        log_sum_exp = tl.load(lse_ptr + row)
        grad_sum = tl.load(grad_sum_ptr + row)
        zero_row = tl.load(zero_row_ptr + row) != 0
        for start in range(0, vocab, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            mask = cols < vocab
            log_p = _log_probs(x_ptr, cols, mask, global_max, log_sum_exp)
            grad = tl.where(zero_row, 0.0, tl.exp(log_p) * -grad_sum)
            # The one-hot terms are added in fp32 before the single cast, as in the eager path.
            for k in range(0, num_index):
                local_index = tl.load(index_ptr + k)
                if (local_index >= start) & (local_index < start + BLOCK):
                    grad += tl.where(cols == local_index, tl.load(index_grad_ptr + k), 0.0)
            tl.store(grad_ptr + cols, grad.to(grad_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _kl_forward_kernel(a_ptr, b_ptr, a_max_ptr, a_lse_ptr, b_max_ptr, b_lse_ptr, out_ptr, vocab, stride,
                           BLOCK: tl.constexpr):
        # KL(A || B) over the local shard.
        row = tl.program_id(0).to(tl.int64)
        a_ptr += row * stride
        b_ptr += row * stride
        a_max = tl.load(a_max_ptr + row)
        a_lse = tl.load(a_lse_ptr + row)
        b_max = tl.load(b_max_ptr + row)
        b_lse = tl.load(b_lse_ptr + row)
        acc = tl.zeros([BLOCK], tl.float32)
        for start in range(0, vocab, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            mask = cols < vocab
            a_log_p = _log_probs(a_ptr, cols, mask, a_max, a_lse)
            b_log_p = _log_probs(b_ptr, cols, mask, b_max, b_lse)
            a_masked = a_log_p == float("-inf")
            terms = tl.where(a_masked, 0.0, tl.exp(a_log_p) * (a_log_p - b_log_p))
            # Where A has support that B masks out, KL(A || B) is infinite even if p_A underflowed to 0.
            terms = tl.where((b_log_p == float("-inf")) & (~a_masked), float("inf"), terms)
            acc += terms
        tl.store(out_ptr + row, tl.sum(acc, axis=0))

    @triton.jit
    def _kl_backward_kernel(a_ptr, b_ptr, a_max_ptr, a_lse_ptr, b_max_ptr, b_lse_ptr, kl_ptr, grad_out_ptr, a_grad_ptr,
                            b_grad_ptr, vocab, stride, NEED_A_GRAD: tl.constexpr, NEED_B_GRAD: tl.constexpr,
                            BLOCK: tl.constexpr):
        # For KL(A || B): dKL/dz_A = p_A * (log p_A - log p_B - KL) and dKL/dz_B = p_B - p_A.
        row = tl.program_id(0).to(tl.int64)
        a_ptr += row * stride
        b_ptr += row * stride
        a_grad_ptr += row * stride
        b_grad_ptr += row * stride
        a_max = tl.load(a_max_ptr + row)
        a_lse = tl.load(a_lse_ptr + row)
        b_max = tl.load(b_max_ptr + row)
        b_lse = tl.load(b_lse_ptr + row)
        kl = tl.load(kl_ptr + row)
        grad_out = tl.load(grad_out_ptr + row)
        # KL is undefined when either distribution is; both sides then get zero gradient.
        undefined_row = (a_max == float("-inf")) | (b_max == float("-inf"))
        for start in range(0, vocab, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            mask = cols < vocab
            a_log_p = _log_probs(a_ptr, cols, mask, a_max, a_lse)
            b_log_p = _log_probs(b_ptr, cols, mask, b_max, b_lse)
            a_p = tl.exp(a_log_p)
            if NEED_A_GRAD:
                a_grad = tl.where(a_log_p == float("-inf"), 0.0, a_p * (a_log_p - b_log_p - kl)) * grad_out
                a_grad = tl.where(undefined_row, 0.0, a_grad)
                tl.store(a_grad_ptr + cols, a_grad.to(a_grad_ptr.dtype.element_ty), mask=mask)
            if NEED_B_GRAD:
                b_grad = tl.where(undefined_row, 0.0, (tl.exp(b_log_p) - a_p) * grad_out)
                tl.store(b_grad_ptr + cols, b_grad.to(b_grad_ptr.dtype.element_ty), mask=mask)


def _launch(kernel, logits, *args, **kwargs):
    num_rows, vocab = logits.shape
    if num_rows == 0:
        return
    block = min(_MAX_BLOCK, max(16, triton.next_power_of_2(vocab)))
    kernel[(num_rows, )](*args, vocab, logits.stride(0), **kwargs, BLOCK=block, num_warps=_NUM_WARPS)


def local_softmax_stats(logits):
    """Per-row max of a ``[rows, local_vocab]`` shard and ``sum(exp(z - max))`` relative to it."""
    local_max = torch.empty(logits.shape[0], dtype=torch.float32, device=logits.device)
    local_sum = torch.empty_like(local_max)
    _launch(_local_softmax_stats_kernel, logits, logits, local_max, local_sum)
    return local_max, local_sum


def entropy_forward(logits, global_max, log_sum_exp):
    out = torch.empty_like(global_max)
    _launch(_entropy_forward_kernel, logits, logits, global_max, log_sum_exp, out)
    return out


def entropy_backward(logits, global_max, log_sum_exp, entropy, grad_output):
    grad = torch.empty_like(logits)
    _launch(_entropy_backward_kernel, logits, logits, global_max, log_sum_exp, entropy, grad_output, grad)
    return grad


def logprobs_backward(logits, global_max, log_sum_exp, grad_sum, zero_grad_row, local_index, index_grad):
    grad = torch.empty_like(logits)
    _launch(_logprobs_backward_kernel,
            logits,
            logits,
            global_max,
            log_sum_exp,
            grad_sum,
            zero_grad_row.to(torch.int8),
            local_index.contiguous(),
            index_grad.contiguous(),
            grad,
            num_index=local_index.shape[-1])
    return grad


def kl_forward(a_logits, b_logits, a_max, a_lse, b_max, b_lse):
    out = torch.empty_like(a_max)
    _launch(_kl_forward_kernel, a_logits, a_logits, b_logits, a_max, a_lse, b_max, b_lse, out)
    return out


def kl_backward(a_logits, b_logits, a_max, a_lse, b_max, b_lse, kl, grad_output, need_a_grad, need_b_grad):
    a_grad = torch.empty_like(a_logits) if need_a_grad else None
    b_grad = torch.empty_like(b_logits) if need_b_grad else None
    # Unused gradient pointers still need a valid tensor; the constexpr flags skip their stores.
    _launch(_kl_backward_kernel,
            a_logits,
            a_logits,
            b_logits,
            a_max,
            a_lse,
            b_max,
            b_lse,
            kl,
            grad_output,
            a_logits if a_grad is None else a_grad,
            b_logits if b_grad is None else b_grad,
            NEED_A_GRAD=need_a_grad,
            NEED_B_GRAD=need_b_grad)
    return a_grad, b_grad
