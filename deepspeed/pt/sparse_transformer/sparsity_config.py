"""
Copyright 2020 The Microsoft DeepSpeed Team
"""

import torch
import random


class SparsityConfig:
    """Abstract Configuration class to store `sparsity configuration of a self attention layer`.
    It contains shared property of different block-sparse sparsity patterns. However, each class needs to extend it based on required property and functionality.
    """
    def __init__(self, num_heads, seq_len, block=16, different_layout_per_head=False):
        """Initialize the Sparsity Pattern Config.

        For usage example please see, TODO DeepSpeed Sparse Transformer Tutorial

        Arguments:
             num_heads: required: an integer determining number of attention heads of the layer.
             seq_len: required: an integer determining number of attention heads of the layer.
             block: optional: an integer determining the block size. Current implementation of sparse self-attention is based on blocked sparse matrices. In which this parameter defines size of such blocks, `Block X Block`.
             different_layout_per_head: optional: a boolean determining if each head should be assigned a different sparsity layout; default is false and this will be satisfied based on availability.
        """

        self.num_heads = num_heads
        if (seq_len % block != 0):
            raise ValueError(
                f'Sequence Length, {seq_len}, needs to be dividable by Block size {block}!'
            )
        self.seq_len = seq_len
        self.block = block
        self.num_blocks = seq_len // block
        self.different_layout_per_head = different_layout_per_head
        # TODO (ARASH) Currently we allocate layout per head; needs to be updated if heads share a single layout.
        self.layout = torch.zeros((self.num_heads,
                                   self.num_blocks,
                                   self.num_blocks),
                                  dtype=torch.int64)
        self.num_layout_heads = num_heads if different_layout_per_head else 1

    def check_and_propagate_first_head_layout(self):
        if not self.different_layout_per_head:
            self.layout[1:self.num_heads, :, :] = self.layout[0, :, :]


class DenseSparsityConfig(SparsityConfig):
    def __init__(self,
                 num_heads=8,
                 seq_len=1024,
                 block=16,
                 different_layout_per_head=False):
        """Initialize the Dense Sparsity Pattern Config.
        In reality, this is not sparse and all blocks are used. We keep it for the sake of comparison and comprehension.

        Arguments:
             num_heads: required: an integer determining number of attention heads of the layer.
             seq_len: required: an integer determining number of attention heads of the layer.
             block: optional: an integer determining the block size. Current implementation of sparse self-attention is based on blocked sparse matrices. In which this parameter defines size of such blocks, `Block X Block`.
             different_layout_per_head: optional: a boolean determining if each head should be assigned a different sparsity layout; default is false and this will be satisfied based on availability. This argument can be completely ignored and we kept it here for consistency with the parent class.
        """
        super().__init__(num_heads, seq_len, block, different_layout_per_head)
        self.make_layout()

    def make_layout(self):
        """Set 1 to all blocks of the layout meanins the pattern is dense; not sparse.
        """
        self.layout[:, :, :] = 1
        return self.layout


class FixedSparsityConfig(SparsityConfig):
    """Configuration class to store `Fixed` sparsity configuration.
    For more details about this sparsity config, please see `Generative Modeling with Sparse Transformers`: https://arxiv.org/abs/1904.10509; this has been customized.
    This class extends parent class of `SparsityConfig` and customizes it for `Fixed` sparsity.
    """
    def __init__(self,
                 num_heads,
                 seq_len,
                 block=16,
                 different_layout_per_head=False,
                 num_local_blocks=4,
                 num_global_blocks=1,
                 attention='bidirectional',
                 horizontal_global_attention=False,
                 num_differnt_global_patterns=1):
        """Initialize `Fixed` Sparsity Pattern Config.

        For usage example please see, TODO DeepSpeed Sparse Transformer Tutorial

        Arguments:
             num_heads: required: an integer determining number of attention heads of the layer.
             seq_len: required: an integer determining number of attention heads of the layer.
             block: optional: an integer determining the block size. Current implementation of sparse self-attention is based on blocked sparse matrices. In which this parameter defines size of such blocks, `Block X Block`.
             different_layout_per_head: optional: a boolean determining if each head should be assigned a different sparsity layout; default is false and this will be satisfied based on availability.
             num_local_blocks: optional: an integer determining the number of blocks in local attention window.
             num_global_blocks: optional: an integer determining how many consecutive blocks in a local window is used as the representetive of the window for global attention.
             attention: optional: a string determining attention type. Attention can be `unidirectional`, such as autoregressive models, in which tokens attend only to tokens appear before them in the context. Considering that, the upper triangular of attention matrix is empty as above figure. Or it can be `bidirectional`, such as BERT, in which tokens can attend to any other tokens before or after them. Then, the upper triangular part of the attention matrix is mirror of the lower triangular in the above figure.
             horizontal_global_attention: optional: a boolean determining if blocks that are global representative of a local window, also attend to all other blocks. This is valid only if attention type is `bidirectional`. Looking at the attention matrix, that means global attention not only includes the vertical blocks, but aso horizontal blocks.
             num_differnt_global_patterns: optional: an integer determining number of different global attentions layouts. While global attention can be fixed by which block/s are representative of any local window, since there are multi-heads, each head can use a different global representative. For example, with 4 blocks local window and global attention size of 1 block, we can have 4 different versions in which the first, Second, third, or forth block of each local window can be global representative of that window. This parameter determines how many of such patterns we want. Of course, there is a limitation based on num_local_blocks and num_global_blocks.
        """
        super().__init__(num_heads, seq_len, block, different_layout_per_head)

        if (seq_len % (block * num_local_blocks) != 0):
            raise ValueError(
                f'Sequence Length, {seq_len}, must be dividable by local window, {block*num_local_blocks}!'
            )
        self.num_local_blocks = num_local_blocks

        if (num_local_blocks % num_global_blocks != 0):
            raise ValueError(
                f'Number of blocks in a local window, {num_local_blocks}, must be dividable by number of global blocks, {num_global_blocks}!'
            )
        self.num_global_blocks = num_global_blocks

        if (attention != 'unidirectional' and attention != 'bidirectional'):
            raise NotImplementedError(
                'only \"uni/bi-directional\" attentions are supported for now!')
        self.attention = attention

        if (attention != 'bidirectional' and horizontal_global_attention):
            raise ValueError(
                'only \"bi-directional\" attentions can support horizontal global attention!'
            )
        self.horizontal_global_attention = horizontal_global_attention

        if (num_differnt_global_patterns > 1 and not different_layout_per_head):
            raise ValueError(
                f'Number of different layouts cannot be more than one when you have set a single layout for all heads! Set different_layout_per_head to True.'
            )
        if (num_differnt_global_patterns > (num_local_blocks // num_global_blocks)):
            raise ValueError(
                f'Number of layout versions (num_differnt_global_patterns), {num_differnt_global_patterns}, cannot be larger than number of local window blocks divided by number of global blocks, {num_local_blocks} / {num_global_blocks} = {num_local_blocks//num_global_blocks}!'
            )
        self.num_differnt_global_patterns = num_differnt_global_patterns
        self.make_layout()

    def set_local_layout(self, h):
        """Sets local attantion layout used by the given head in the sparse attention.

        Arguments:
             h: required: an integer determining head index
        """
        for i in range(0, self.num_blocks, self.num_local_blocks):
            for row in range(i, i + self.num_local_blocks):
                for col in range(i,
                                 (row + 1 if self.attention == 'unidirectional' else i +
                                  self.num_local_blocks)):
                    self.layout[h, row, col] = 1

    def set_global_layout(self, h):
        """Sets global attantion layout used by the given head in the sparse attention.

        Currently we set global blocks starting from the last block of a local window to the first one. That means if a local window consists of 4 blocks and global attention size is one block, we use block #4 in each local window as global. If we have different layout per head, then other heads will get #3, #2, and #1. And if we have more heads (and different layout has set) than num of global attentions, multiple head may have same global attentions.
        Note) if horizontal_global_attention is set, global blocks will be set both horizontally and vertically.

        Arguments:
             h: required: an integer determining head index
        """
        first_global_block_idx = self.num_local_blocks - (
            1 + h % self.num_differnt_global_patterns) * self.num_global_blocks
        for i in range(first_global_block_idx, self.num_blocks, self.num_local_blocks):

            # vertical global attention
            first_row = 0 if self.attention == 'bidirectional' else ((
                (i // self.num_local_blocks) + 1) * self.num_local_blocks)
            self.layout[h, first_row:, i:i + self.num_global_blocks] = 1
            '''
            for row in range(first_row, self.num_blocks):
                for col in range(i, i + self.num_global_blocks):
                    self.layout[h, row, col] = 1
            '''

            # horizontal global attention
            if (self.horizontal_global_attention):
                self.layout[h, i:i + self.num_global_blocks, :] = 1
                '''
                for row in range(i, i + self.num_global_blocks):
                    for col in range(0, self.num_blocks):
                        self.layout[h, row, col] = 1
                '''

    def make_layout(self):
        """Generates `Fixed` sparsity layout used by each head in the sparse attention.

        Return:
             layout: a tensor determining the sparsity layout of each head
        """

        for h in range(0, self.num_layout_heads):
            self.set_local_layout(h)
            self.set_global_layout(h)

        self.check_and_propagate_first_head_layout()
        return self.layout


class VariableSparsityConfig(SparsityConfig):
    """Configuration class to store `Variable` sparsity configuration.
    This layout is an extension of FixedSparsityConfig in which:
      - user can set random layout; default value is zero means no random block
      - user can provide a list of local block sizes
      - user can provide a list of global block indices.

    For more details about `Fixed` sparsity config, please see `Generative Modeling with Sparse Transformers`: https://arxiv.org/abs/1904.10509; this has been customized.
    This class extends parent class of `SparsityConfig` and customizes it for `Fixed` sparsity.
    """
    def __init__(self,
                 num_heads,
                 seq_len,
                 block=16,
                 different_layout_per_head=False,
                 num_random_blocks=0,
                 local_window_blocks=[4],
                 global_block_indices=[0],
                 global_block_end_indices=None,
                 attention='bidirectional',
                 horizontal_global_attention=False):
        """Initialize `Variable` Sparsity Pattern Config.

        For usage example please see, TODO DeepSpeed Sparse Transformer Tutorial

        Arguments:
             num_heads: required: an integer determining number of attention heads of the layer.
             seq_len: required: an integer determining number of attention heads of the layer.
             block: optional: an integer determining the block size. Current implementation of sparse self-attention is based on blocked sparse matrices. In which this parameter defines size of such blocks, `Block X Block`.
             different_layout_per_head: optional: a boolean determining if each head should be assigned a different sparsity layout; default is false and this will be satisfied based on availability. Currently this sparsity config can only assign single layout to all heads; needs to be extended for different layout per head.
             num_random_blocks: optional: an integer determining the number of random blocks in each block row.
             local_window_blocks: optional: a list of integers determining the number of blocks in each local attention window. It assumes first number determines # of blocks in the first local window, second the second window, ..., and the last number determines the number of blocks in the remaining local windows.
             global_block_indices: optional: a list of integers determining which blocks are considered as global attention. Given indices, determine the blocks that all other token blocks attend to and they attend to all other token blocks. Default value is only index 0. Notice that if global_block_end_indices parameter is set, this parameter is used as starting index of each global window.
             global_block_end_indices: optional: a list of integers determining end indices of global window blocks. By default this is not used. But if it is set, it must have the same size of global_block_indices parameter, and combining this two parameters, for each index i, blocks from global_block_indices[i] to global_block_end_indices[i] (exclusive) are considered as global attention.
             num_global_blocks: optional: an integer determining how many consecutive blocks in a local window is used as the representetive of the window for global attention.
             attention: optional: a string determining attention type. Attention can be `unidirectional`, such as autoregressive models, in which tokens attend only to tokens appear before them in the context. Considering that, the upper triangular of attention matrix is empty as above figure. Or it can be `bidirectional`, such as BERT, in which tokens can attend to any other tokens before or after them. Then, the upper triangular part of the attention matrix is mirror of the lower triangular in the above figure.
             horizontal_global_attention: optional: a boolean determining if blocks that are global representative of a local window, also attend to all other blocks. This is valid only if attention type is `bidirectional`. Looking at the attention matrix, that means global attention not only includes the vertical blocks, but aso horizontal blocks.
        """
        super().__init__(num_heads, seq_len, block, different_layout_per_head)

        all_local_blocks = 0
        for local_blocks in local_window_blocks:
            all_local_blocks += local_blocks
        if (self.num_blocks < all_local_blocks):
            raise ValueError(
                f'Number of all local window blocks, {all_local_blocks}, must be smaller than overal number of blocks in a row, {self.num_blocks}!'
            )
        self.local_window_blocks = local_window_blocks
        if (self.num_blocks < len(global_block_indices)):
            raise ValueError(
                f'Number of global blocks indices, {global_block_indices}, must be smaller than overal number of blocks in a row, {self.num_blocks}!'
            )
        for idx in global_block_indices:
            if idx >= self.num_blocks:
                raise ValueError(
                    f'Global block index, {global_block_indices[idx]}, must be smaller than overal number of blocks in a row, {self.num_blocks}!'
                )
        self.global_block_indices = global_block_indices

        if (global_block_end_indices is not None):
            if (len(global_block_indices) != len(global_block_end_indices)):
                raise ValueError(
                    f'Global block start indices length, {len(global_block_indices)}, must be same as global block end indices length, {len(global_block_end_indices)}!'
                )
            for _, (start_idx, end_idx) in enumerate(zip(global_block_indices, global_block_end_indices)):
                if end_idx > self.num_blocks:
                    raise ValueError(
                        f'Global block end index, {global_block_end_indices[idx]}, must be smaller (equal) than overal number of blocks in a row, {self.num_blocks}!'
                    )
                if start_idx >= end_idx:
                    raise ValueError(
                        f'Global block start index, {start_idx}, must be smaller than global block end index, {end_idx}!'
                    )
        self.global_block_end_indices = global_block_end_indices

        if (attention != 'unidirectional' and attention != 'bidirectional'):
            raise NotImplementedError(
                'only \"uni/bi-directional\" attentions are supported for now!')
        self.attention = attention

        if (attention != 'bidirectional' and horizontal_global_attention):
            raise ValueError(
                'only \"bi-directional\" attentions can support horizontal global attention!'
            )
        self.horizontal_global_attention = horizontal_global_attention
        self.make_layout()

    def set_random_layout(self, h):
        """Sets random attantion layout used by the given head in the sparse attention.
        Note) By default, it assumes there will be a unique random block layout for all heads; unless `different_layout_per_head` parameter is set in which each head can have a different random layout.

        Arguments:
             h: required: an integer determining head index
        """

        for row in range(0, self.num_blocks):
            rnd_cols = random.sample(range(0, self.num_blocks), self.num_random_blocks)
            self.layout[h, row, rnd_cols] = 1

    def set_local_layout(self, h):
        """Sets local attantion layout used by the given head in the sparse attention.

        Arguments:
             h: required: an integer determining head index
        """
        start_block_idx = 0
        end_block_idx = 0
        for block_size in self.local_window_blocks:
            end_block_idx += block_size
            end_block_idx = min(end_block_idx, self.num_blocks)
            for row in range(start_block_idx, end_block_idx):
                for col in range(
                        start_block_idx,
                    (row + 1 if self.attention == 'unidirectional' else end_block_idx)):
                    self.layout[h, row, col] = 1
            start_block_idx += block_size

        # if there is any remaining not attended part, use the lats local window block size as local window for the remaining applicable local windows
        for i in range(start_block_idx, self.num_blocks, block_size):
            end_block_idx = min(i + block_size, self.num_blocks)
            for row in range(i, end_block_idx):
                for col in range(
                        i,
                    (row + 1 if self.attention == 'unidirectional' else end_block_idx)):
                    self.layout[h, row, col] = 1

    def set_global_layout(self, h):
        """Sets global attantion layout used by the given head in the sparse attention.

        Arguments:
             h: required: an integer determining head index
        """
        if (self.global_block_end_indices is None):
            for idx in self.global_block_indices:
                #global rows
                if (self.horizontal_global_attention):
                    self.layout[h, idx, :] = 1

                #global columns
                self.layout[h, :, idx] = 1
        else:
            for _, (start_idx, end_idx) in enumerate(zip(self.global_block_indices, self.global_block_end_indices)):
                #global rows
                if (self.horizontal_global_attention):
                    self.layout[h, start_idx:end_idx, :] = 1

                #global columns
                self.layout[h, :, start_idx:end_idx] = 1

    def make_layout(self):
        """Generates `Fixed` sparsity layout used by each head in the sparse attention.

        Return:
             layout: a tensor determining the sparsity layout of each head
        """

        for h in range(0, self.num_layout_heads):
            self.set_random_layout(h)
            self.set_local_layout(h)
            self.set_global_layout(h)

        self.check_and_propagate_first_head_layout()
        return self.layout


class BigBirdSparsityConfig(SparsityConfig):
    """Configuration class to store `BigBird` sparsity configuration.
    For more details about this sparsity config, please see `Big Bird: Transformers for Longer Sequences`: https://arxiv.org/pdf/2007.14062.pdf
    This class extends parent class of `SparsityConfig` and customizes it for `BigBird` sparsity.
    """
    def __init__(self,
                 num_heads=8,
                 seq_len=1024,
                 block=16,
                 different_layout_per_head=False,
                 num_random_blocks=1,
                 num_sliding_window_blocks=3,
                 num_global_blocks=1):
        """Initialize the BigBird Sparsity Pattern Config.

        For usage example please see, TODO DeepSpeed Sparse Transformer Tutorial

        Arguments:
             num_heads: required: an integer determining number of attention heads of the layer.
             seq_len: required: an integer determining number of attention heads of the layer.
             block: optional: an integer determining the block size. Current implementation of sparse self-attention is based on blocked sparse matrices. In which this parameter defines size of such blocks, `Block X Block`.
             different_layout_per_head: optional: a boolean determining if each head should be assigned a different sparsity layout; default is false and this will be satisfied based on availability.
             num_random_blocks: optional: an integer determining the number of random blocks in each block row.
             num_sliding_window_blocks: optional: an integer determining the number of blocks in sliding local attention window.
             num_global_blocks: optional: an integer determining how many consecutive blocks, starting from index 0, are considered as global attention. Global block tokens will be attended by all other block tokens and will attend to all other block tokens as well.
        """
        super().__init__(num_heads, seq_len, block, different_layout_per_head)

        if (self.num_blocks < num_random_blocks):
            raise ValueError(
                f'Number of random blocks, {num_random_blocks}, must be smaller than overal number of blocks in a row, {self.num_blocks}!'
            )
        self.num_random_blocks = num_random_blocks

        if (self.num_blocks < num_sliding_window_blocks):
            raise ValueError(
                f'Number of sliding window blocks, {num_sliding_window_blocks}, must be smaller than overal number of blocks in a row, {self.num_blocks}!'
            )
        self.num_sliding_window_blocks = num_sliding_window_blocks

        if (self.num_blocks < num_global_blocks):
            raise ValueError(
                f'Number of global blocks, {num_global_blocks}, must be smaller than overal number of blocks in a row, {self.num_blocks}!'
            )
        self.num_global_blocks = num_global_blocks
        self.make_layout()

    def set_random_layout(self, h):
        """Sets random attantion layout used by the given head in the sparse attention.
        Note) By default, it assumes there will be a unique random block layout for all heads; unless `different_layout_per_head` parameter is set in which each head can have a different random layout.

        Arguments:
             h: required: an integer determining head index
        """

        for row in range(0, self.num_blocks):
            rnd_cols = random.sample(range(0, self.num_blocks), self.num_random_blocks)
            self.layout[h, row, rnd_cols] = 1

    def set_sliding_window_layout(self, h):
        """Sets sliding local attantion layout used by the given head in the sparse attention.

        Arguments:
             h: required: an integer determining head index
        """

        w = self.num_sliding_window_blocks // 2
        for row in range(0, self.num_blocks):
            start = max(0, row - w)
            end = min(row + w + 1, self.num_blocks)
            self.layout[h, row, start:end] = 1

    def set_global_layout_itc(self, h):
        """Sets global attantion layout used by the given head in the sparse attention.

        Arguments:
             h: required: an integer determining head index
        """

        #global rows
        self.layout[h, 0:self.num_global_blocks, :] = 1

        #global columns
        self.layout[h, :, 0:self.num_global_blocks] = 1

    def make_layout(self):
        """Generates `BigBird` sparsity layout used by each head in the sparse attention.

        Return:
             layout: a tensor determining the sparsity layout of each head
        """

        for h in range(0, self.num_layout_heads):
            self.set_random_layout(h)
            self.set_sliding_window_layout(h)
            self.set_global_layout_itc(h)

        self.check_and_propagate_first_head_layout()
        return self.layout


class BSLongformerSparsityConfig(SparsityConfig):
    """Configuration class to store edited `Longformer` sparsity configuration.

    Note) this is a block-sparse version of the Longformer which is slightly different than original Longformer; which is element-wise sparsity.

    For more details about this sparsity config, please see `Longformer: The Long-Document Transformer`: https://arxiv.org/pdf/2004.05150.pdf
    This class extends parent class of `SparsityConfig` and customizes it for `Longformer` sparsity.
    """
    def __init__(self,
                 num_heads=8,
                 seq_len=1024,
                 block=16,
                 different_layout_per_head=False,
                 num_sliding_window_blocks=3,
                 global_block_indices=[0],
                 global_block_end_indices=None):
        """Initialize the edited `Longformer` Sparsity Pattern Config.

        For usage example please see, TODO DeepSpeed Sparse Transformer Tutorial

        Arguments:
             num_heads: required: an integer determining number of attention heads of the layer.
             seq_len: required: an integer determining number of attention heads of the layer.
             block: optional: an integer determining the block size. Current implementation of sparse self-attention is based on blocked sparse matrices. In which this parameter defines size of such blocks, `Block X Block`.
             different_layout_per_head: optional: a boolean determining if each head should be assigned a different sparsity layout; default is false and this will be satisfied based on availability.

             num_sliding_window_blocks: optional: an integer determining the number of blocks in sliding local attention window.
             global_block_indices: optional: a list of integers determining which blocks are considered as global attention. Given indices, determine the blocks that all other token blocks attend to and they attend to all other token blocks. Default value is only index 0. Notice that if global_block_end_indices parameter is set, this parameter is used as starting index of each global window.
             global_block_end_indices: optional: a list of integers determining end indices of global window blocks. By default this is not used. But if it is set, it must have the same size of global_block_indices parameter, and combining this two parameters, for each index i, blocks from global_block_indices[i] to global_block_end_indices[i] (exclusive) are considered as global attention.
        """
        super().__init__(num_heads, seq_len, block, different_layout_per_head)

        if (self.num_blocks < num_sliding_window_blocks):
            raise ValueError(
                f'Number of sliding window blocks, {num_sliding_window_blocks}, must be smaller than overal number of blocks in a row, {self.num_blocks}!'
            )
        self.num_sliding_window_blocks = num_sliding_window_blocks

        if (self.num_blocks < len(global_block_indices)):
            raise ValueError(
                f'Number of global blocks indices, {global_block_indices}, must be smaller than overal number of blocks in a row, {self.num_blocks}!'
            )
        for idx in global_block_indices:
            if idx >= self.num_blocks:
                raise ValueError(
                    f'Global block index, {global_block_indices[idx]}, must be smaller than overal number of blocks in a row, {self.num_blocks}!'
                )
        self.global_block_indices = global_block_indices

        if (global_block_end_indices is not None):
            if (len(global_block_indices) != len(global_block_end_indices)):
                raise ValueError(
                    f'Global block start indices length, {len(global_block_indices)}, must be same as global block end indices length, {len(global_block_end_indices)}!'
                )
            for _, (start_idx, end_idx) in enumerate(zip(global_block_indices, global_block_end_indices)):
                if end_idx > self.num_blocks:
                    raise ValueError(
                        f'Global block end index, {global_block_end_indices[idx]}, must be smaller (equal) than overal number of blocks in a row, {self.num_blocks}!'
                    )
                if start_idx >= end_idx:
                    raise ValueError(
                        f'Global block start index, {start_idx}, must be smaller than global block end index, {end_idx}!'
                    )
        self.global_block_end_indices = global_block_end_indices
        self.make_layout()

    def set_sliding_window_layout(self, h):
        """Sets sliding local attantion layout used by the given head in the sparse attention.

        Arguments:
             h: required: an integer determining head index
        """

        w = self.num_sliding_window_blocks // 2
        for row in range(0, self.num_blocks):
            start = max(0, row - w)
            end = min(row + w + 1, self.num_blocks)
            self.layout[h, row, start:end] = 1

    def set_global_layout(self, h):
        """Sets global attantion layout used by the given head in the sparse attention.

        Arguments:
             h: required: an integer determining head index
        """
        if (self.global_block_end_indices is None):
            for idx in self.global_block_indices:
                #global rows
                self.layout[h, idx, :] = 1

                #global columns
                self.layout[h, :, idx] = 1
        else:
            for _, (start_idx, end_idx) in enumerate(zip(self.global_block_indices, self.global_block_end_indices)):
                #global rows
                self.layout[h, start_idx:end_idx, :] = 1

                #global columns
                self.layout[h, :, start_idx:end_idx] = 1

    def make_layout(self):
        """Generates edited `Longformer` sparsity layout used by each head in the sparse attention.

        Return:
             layout: a tensor determining the sparsity layout of each head
        """

        for h in range(0, self.num_layout_heads):
            self.set_sliding_window_layout(h)
            self.set_global_layout(h)

        self.check_and_propagate_first_head_layout()
        return self.layout
