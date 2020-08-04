"""
Copyright 2020 The Microsoft DeepSpeed Team
"""

import torch.nn as nn
from torch.nn.functional import *
import torch
from collections import namedtuple
import deepspeed_sparse_transformer_util
from deepspeed.pt.sparse_transformer import MatMul, Softmax
import sys

make_layout = deepspeed_sparse_transformer_util.make_layout


class SparsityConfig:
    """Configuration class to store sparsity configuration of a self attention layer`.
    """
    def __init__(self,
                 mode='fixed',
                 block=16,
                 stride=64,
                 attention='bidirectional',
                 numverts=1,
                 vertsize=1):
        """Initialize the Sparsity Pattern Config.

        For more details about sparsity config, please see `Generative Modeling with Sparse Transformers`: https://arxiv.org/abs/1904.10509

        For usage example please see, TODO DeepSpeed Sparse Transformer Tutorial

        Arguments:
             mode: optional: a string determining the sparsity mode. In addition to `dense`, currently we support `fixed` mode that combines local and global attention suitable for document modeling.
             block: optional: an integer determining the block size. Current implementation of sparse self-attention is based on blocked sparse matrices. In which this parameter defines size of such blocks, `Block X Block`.
             stride: optional: an integer determining the local attention window size.
             attention: optional: a string determining attention type. Attention can be `unidirectional`, such as autoregressive models, in which tokens attend only to tokens appear before them in the context. Considering that, the upper triangular of attention matrix is empty as above figure. Or it can be `bidirectional`, such as BERT, in which tokens can attend to any other tokens before or after them. Then, the upper triangular part of the attention matrix is mirror of the lower triangular in the above figure.
             numverts: optional: an integer determining number of different global attentions. While global attention can be fixed by which block/s are representative of any local window, since there are multi-heads, each head can use a different global representative. For example, with 4 blocks local window and global attention size of 1, we can have 4 different versions in which first, Second, third, or forth block of each local window be global representative of that window. This parameter determines how many of such patterns we want. Of course, there is a limitation based on block and stride size. As an example, considering one block as global representative, in the above figure, there are maximum of four versions of global attention.
             vertsize: optional: an integer determining how many consecutive blocks a local window is used for global attention.
    """
        if mode != 'dense' and mode != 'fixed':
            raise NotImplementedError(
                'only \"dense\" and \"fixed\" modes are supported for now')
        self.mode = mode
        self.block = block
        self.stride = stride
        if attention != 'unidirectional' and attention != 'bidirectional':
            raise NotImplementedError(
                'only \"uni/bi-directional\" attentions are supported for now')
        self.attention = attention
        self.numverts = numverts
        self.vertsize = vertsize


class SparseSelfAttention(nn.Module):
    """Implements an efficient Sparse Self Attention of Transformer layer based on `Generative Modeling with Sparse Transformers`: https://arxiv.org/abs/1904.10509

    For more information please see, TODO DeepSpeed Sparse Transformer.

    For usage example please see, TODO DeepSpeed Sparse Transformer Tutorial.
    """
    def __init__(self,
                 sparsity_config=SparsityConfig(),
                 key_padding_mask_mode='add',
                 attn_mask_mode='mul'):
        """Initialize the sparse self attention layer.
        Arguments:
            sparsity_config: optional: this parameter determins sparsity pattern configuration; it is based on SparsityConfig class.
            key_padding_mask_mode: optional: a string determining if key padding mask needs to be added, `add`, or be multiplied, `mul`.
            attn_mask_mode: optional: a string determining if attention mask needs to be added, `add`, or be multiplied, `mul`.
        """
        super().__init__()

        # sparsity information
        self.sparsity_config = sparsity_config

        # mask modes
        self.key_padding_mask_mode = key_padding_mask_mode
        self.attn_mask_mode = attn_mask_mode

    @staticmethod
    def _set_local_layout(layout, h, num_blocks, block_stride, attention):
        """Sets local attantion layout used by the given head in the sparse attention.

        Arguments:
             layout: required: sparsity layout tensor
             h: required: an integer determining head index
             num_blocks: required: an integer determining the block size. Current implementation of sparse self-attention is based on blocked sparse matrices. In which this parameter defines size of such blocks, `Block X Block`.
             block_stride: required: an integer determining the local attention window size.
             attention: required: a string determining attention type. Attention can be `unidirectional`, such as autoregressive models, in which tokens attend only to tokens appear before them in the context. Considering that, the upper triangular of attention matrix is empty as above figure. Or it can be `bidirectional`, such as BERT, in which tokens can attend to any other tokens before or after them. Then, the upper triangular part of the attention matrix is mirror of the lower triangular in the above figure.
        Return:
             layout: the layout tensor in which the local attention for head #h has been set
        """

        for i in range(0, num_blocks, block_stride):
            for j in range(i, i + block_stride):
                for k in range(
                        i,
                    (j + 1 if attention == 'unidirectional' else i + block_stride)):
                    layout[h, j, k] = 1
        return layout

    @staticmethod
    def _set_global_layout(layout,
                           h,
                           num_blocks,
                           block_stride,
                           attention,
                           numverts,
                           vertsize):
        """Sets global attantion layout used by the given head in the sparse attention.

        Arguments:
             layout: required: sparsity layout tensor
             h: required: an integer determining head index
             num_blocks: required: an integer determining the block size. Current implementation of sparse self-attention is based on blocked sparse matrices. In which this parameter defines size of such blocks, `Block X Block`.
             block_stride: required: an integer determining the local attention window size.
             attention: required: a string determining attention type. Attention can be `unidirectional`, such as autoregressive models, in which tokens attend only to tokens appear before them in the context. Considering that, the upper triangular of attention matrix is empty as above figure. Or it can be `bidirectional`, such as BERT, in which tokens can attend to any other tokens before or after them. Then, the upper triangular part of the attention matrix is mirror of the lower triangular in the above figure.
             numverts: required: an integer determining number of different global attentions. While global attention can be fixed by which block/s are representative of any local window, since there are multi-heads, each head can use a different global representative. For example, with 4 blocks local window and global attention size of 1, we can have 4 different versions in which first, Second, third, or forth block of each local window be global representative of that window. This parameter determines how many of such patterns we want. Of course, there is a limitation based on block and stride size. As an example, considering one block as global representative, in the above figure, there are maximum of four versions of global attention.
             vertsize: required: an integer determining how many consecutive blocks a local window is used for global attention.
        Return:
             layout: the layout tensor in which the global attention for head #h has been set
        """

        start = block_stride - (1 + h % numverts) * vertsize
        for i in range(0, num_blocks):
            end = i if attention == 'unidirectional' else num_blocks
            for j in range(start, end, block_stride):
                for k in range(j, min(j + vertsize, num_blocks)):
                    layout[h, i, k] = 1
        return layout

    @staticmethod
    def _make_layout_python(num_heads,
                            mode,
                            num_blocks,
                            block_stride,
                            attention,
                            numverts,
                            vertsize):
        """Generates sparsity layout used by each head in the sparse attention.
        Currently this function is able to create 'dense' or 'fixed' layout as described here: https://arxiv.org/abs/1904.10509

        This function can be extend to add any block-base sparsity  following the 'fixed' part below.

        Arguments:
             num_heads: required: an integer determining number of heads of the model
             mode: required: a string determining the sparsity mode. In addition to `dense`, currently we support `fixed` mode that combines local and global attention suitable for document modeling.
             num_blocks: required: an integer determining the block size. Current implementation of sparse self-attention is based on blocked sparse matrices. In which this parameter defines size of such blocks, `Block X Block`.
             block_stride: required: an integer determining the local attention window size.
             attention: required: a string determining attention type. Attention can be `unidirectional`, such as autoregressive models, in which tokens attend only to tokens appear before them in the context. Considering that, the upper triangular of attention matrix is empty as above figure. Or it can be `bidirectional`, such as BERT, in which tokens can attend to any other tokens before or after them. Then, the upper triangular part of the attention matrix is mirror of the lower triangular in the above figure.
             numverts: required: an integer determining number of different global attentions. While global attention can be fixed by which block/s are representative of any local window, since there are multi-heads, each head can use a different global representative. For example, with 4 blocks local window and global attention size of 1, we can have 4 different versions in which first, Second, third, or forth block of each local window be global representative of that window. This parameter determines how many of such patterns we want. Of course, there is a limitation based on block and stride size. As an example, considering one block as global representative, in the above figure, there are maximum of four versions of global attention.
             vertsize: required: an integer determining how many consecutive blocks a local window is used for global attention.
        Return:
             layout: a tensor determining the sparsity layout of each head
        """

        if (block_stride % vertsize) != 0:
            raise ValueError(
                f'Number of blocks in a stride window {block_stride} must be dividable by vertical block size {vertsize}'
            )

        if numverts > (block_stride / vertsize):
            raise ValueError(
                f'Number of layout versions {num_verts} cannot be larger than blocks in a stride window divided by vertical block size {block_stride} / {vertsize} = {block_stride/vertsize}'
            )

        layout = torch.zeros((num_heads, num_blocks, num_blocks), dtype=torch.int64)
        if mode == "dense":
            layout[:, :, :] = 1
        elif mode == "fixed":
            for i in range(0, num_heads):
                layout = SparseSelfAttention._set_local_layout(
                    layout,
                    i,
                    num_blocks,
                    block_stride,
                    attention)
                layout = SparseSelfAttention._set_global_layout(
                    layout,
                    i,
                    num_blocks,
                    block_stride,
                    attention,
                    numverts,
                    vertsize)
        else:
            raise NotImplementedError(
                'Only \"dense\" and \"fixed\" modes are supported for now!')

        return layout

    @staticmethod
    def _make_layout(num_heads,
                     mode,
                     num_blocks,
                     block_stride,
                     attention,
                     numverts,
                     vertsize):
        return make_layout(num_heads,
                           mode,
                           num_blocks,
                           block_stride,
                           attention,
                           numverts,
                           vertsize)

    ops = dict()

    # add to cache
    def get_ops(self, H, L):
        import sys
        if L not in SparseSelfAttention.ops:
            spConfig = self.sparsity_config

            if L % spConfig.block != 0:
                raise ValueError(
                    f'Sequence length {L} must be dividable by block size {spConfig.block}'
                )
            num_blocks = L // spConfig.block

            if spConfig.stride % spConfig.block != 0:
                raise ValueError(
                    f'Stride {spConfig.stride} must be dividable by block size {spConfig.block}'
                )
            block_stride = spConfig.stride // spConfig.block

            layout = SparseSelfAttention._make_layout(H,
                                                      num_blocks,
                                                      spConfig.mode,
                                                      block_stride,
                                                      spConfig.attention,
                                                      spConfig.numverts,
                                                      spConfig.vertsize)

            sparse_dot_sdd_nt = MatMul(layout,
                                       spConfig.block,
                                       'sdd',
                                       trans_a=False,
                                       trans_b=True)

            sparse_dot_dsd_nn = MatMul(layout,
                                       spConfig.block,
                                       'dsd',
                                       trans_a=False,
                                       trans_b=False)

            sparse_softmax = Softmax(layout, spConfig.block)

            SparseSelfAttention.ops[L] = (sparse_dot_sdd_nt,
                                          sparse_dot_dsd_nn,
                                          sparse_softmax)
        return SparseSelfAttention.ops[L]

    def transpose_key_for_scores(self, x, L):
        bsz, num_heads, seq_len, head_dim = x.size()
        if seq_len != L:
            return x.permute(0, 1, 3, 2)
        return x

    def transpose_mask_for_sparse(self, qtype, x, is_key_padding_mask=False):
        x = x.type(qtype)
        if is_key_padding_mask:
            xdim = x.dim()
            for d in range(xdim - 1, 0, -1):
                x = x.squeeze(dim=d)
            return x
        return x.squeeze()

    # forward pass
    def forward(self,
                query,
                key,
                value,
                rpe=None,
                key_padding_mask=None,
                attn_mask=None):
        """Applies forward phase of sparse self attention

        Arguments:
            query: required: query tensor
            key: required: key tensor
            value: required: value tensor
            rpe: optional: a tensor same dimension as x that is used as relative position embedding
            key_padding_mask: optional: a mask tensor of size (BatchSize X SequenceLength)
            attn_mask: optional: a mask tensor of size (SequenceLength X SequenceLength); currently only 2D is supported
            key_padding_mask_mode: optional: a boolean determining if key_padding_mask needs to be added or multiplied
            attn_mask_mode: optional: a boolean determining if attn_mask needs to be added or multiplied

        Return:
             attn_output: a dense tensor containing attnetion context
        """
        bsz, num_heads, tgt_len, head_dim = query.size()

        # transpose back key if it is already transposed
        key = self.transpose_key_for_scores(key, tgt_len)

        # check that operation is supported
        if query.shape != key.shape or key.shape != value.shape:
            raise NotImplementedError('only self-attention is supported for now')

        # squeeze key_padding_mask if it is given
        if key_padding_mask is not None:
            key_padding_mask = self.transpose_mask_for_sparse(query.dtype,
                                                              key_padding_mask,
                                                              is_key_padding_mask=True)

        # squeeze attn_mask if it is given
        if attn_mask is not None:
            attn_mask = self.transpose_mask_for_sparse(query.dtype, attn_mask)

        # cache look-up table computations etc
        sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax = self.get_ops(num_heads, tgt_len)

        scaling = float(head_dim)**-0.5

        # attention scores
        attn_output_weights = sparse_dot_sdd_nt(query, key)
        attn_output_weights = sparse_softmax(
            attn_output_weights,
            scale=scaling,
            rpe=rpe,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            key_padding_mask_mode=self.key_padding_mask_mode,
            attn_mask_mode=self.attn_mask_mode)

        # outputs
        attn_output = sparse_dot_dsd_nn(attn_output_weights, value)
        return attn_output
