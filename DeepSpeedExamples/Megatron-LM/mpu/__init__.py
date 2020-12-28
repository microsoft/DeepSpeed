# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model parallel utility interface."""

from .cross_entropy import vocab_parallel_cross_entropy

from .data import broadcast_data

from .grads import clip_grad_norm

from .initialize import destroy_model_parallel
from .initialize import get_data_parallel_group
from .initialize import get_data_parallel_rank
from .initialize import get_data_parallel_world_size
from .initialize import get_model_parallel_group
from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_src_rank
from .initialize import get_model_parallel_world_size
from .initialize import initialize_model_parallel
from .initialize import model_parallel_is_initialized

from .layers import ColumnParallelLinear
from .layers import ParallelEmbedding
from .layers import RowParallelLinear
from .layers import VocabParallelEmbedding

from .mappings import copy_to_model_parallel_region
from .mappings import gather_from_model_parallel_region
from .mappings import reduce_from_model_parallel_region
from .mappings import scatter_to_model_parallel_region

from .random import checkpoint
from .random import partition_activations_in_checkpoint
from .random import get_cuda_rng_tracker
from .random import model_parallel_cuda_manual_seed

from .transformer import BertParallelSelfAttention
from .transformer import BertParallelTransformerLayer
from .transformer import GPT2ParallelTransformer
from .transformer import LayerNorm
