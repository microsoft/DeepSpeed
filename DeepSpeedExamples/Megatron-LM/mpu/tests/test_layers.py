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

import random
import sys
sys.path.append("../..")

import torch
import torch.nn.init as init
from torch.nn.parameter import Parameter
import mpu

from commons import initialize_distributed
from commons import print_separator
from commons import set_random_seed
from mpu import layers


def test_parallel_embedding(model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing parallel embedding with model parallel size {} ...'.
              format(model_parallel_size))

    mpu.initialize_model_parallel(model_parallel_size)
    model_parallel_size = mpu.get_model_parallel_world_size()

    batch_size = 17
    seq_length = 23
    vocab_size = 48
    hidden_size = 16
    seed = 1236

    set_random_seed(123)
    input_data = torch.LongTensor(
        size=(batch_size,seq_length)).random_(0, vocab_size).cuda()
    loss_weight = torch.randn([batch_size, seq_length, hidden_size]).cuda()

    set_random_seed(seed)
    embedding_original = torch.nn.Embedding(vocab_size, hidden_size).cuda()

    output = embedding_original(input_data)
    loss_original = torch.mul(output, loss_weight).sum()
    loss_original.backward()

    set_random_seed(seed)
    embedding_parallel = layers.ParallelEmbedding(
                vocab_size, hidden_size, init_method=init.normal_).cuda()
    output = embedding_parallel(input_data)
    loss_parallel = torch.mul(output, loss_weight).sum()
    loss_parallel.backward()

    set_random_seed(seed)
    embedding_vocab_parallel = layers.VocabParallelEmbedding(
        vocab_size, hidden_size, init_method=init.normal_).cuda()
    output = embedding_vocab_parallel(input_data)
    loss_vocab_parallel = torch.mul(output, loss_weight).sum()
    loss_vocab_parallel.backward()

    torch.distributed.barrier()
    error = loss_parallel.sub(loss_original).abs()
    print('   error in loss (parallel) on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-12, 'error: {}'.format(error)

    torch.distributed.barrier()
    error = loss_vocab_parallel.sub(loss_original).abs()
    print('   error in loss (vocab parallel) on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-12, 'error: {}'.format(error)

    weight_grad_orig = torch.split(embedding_original.weight.grad,
                                   hidden_size // model_parallel_size,
                                   1)[mpu.get_model_parallel_rank()]
    error = embedding_parallel.weight.grad.sub(weight_grad_orig).abs().max()
    print('   error in grad (parallel) on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-12, 'error: {}'.format(error)

    weight_grad_orig = torch.split(embedding_original.weight.grad,
                                   vocab_size // model_parallel_size,
                                   0)[mpu.get_model_parallel_rank()]
    error = embedding_vocab_parallel.weight.grad.sub(
        weight_grad_orig).abs().max()
    print('   error in grad (vocab parallel) on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-12, 'error: {}'.format(error)

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('>> passed the test :-)')


def test_initialize_affine_weight(model_parallel_size):

    mpu.initialize_model_parallel(model_parallel_size)
    if torch.distributed.get_rank() == 0:
        print('> testing initialize_affine_weight with model parallel '
              'size: {}'.format(model_parallel_size))
    model_parallel_size = mpu.get_model_parallel_world_size()

    seed = 12345
    input_size_coeff = 13
    input_size = input_size_coeff * model_parallel_size
    output_size_coeff = 17
    output_size = output_size_coeff * model_parallel_size

    # ---------------
    # Column parallel
    # ---------------
    weight = torch.empty(output_size_coeff, input_size)
    set_random_seed(seed)
    layers._initialize_affine_weight(weight, output_size, input_size,

                                     output_size_coeff, 0,
                                     torch.nn.init.normal_)
    # Target.
    set_random_seed(seed)
    master_weight = torch.empty(output_size, input_size)
    torch.nn.init.normal_(master_weight)
    rank = mpu.get_model_parallel_rank()
    my_weight = torch.split(master_weight, output_size_coeff,
                            dim=0)[rank].contiguous().clone()

    # Compare.
    error = weight.sub(my_weight).abs().max()
    torch.distributed.barrier()
    print('   column parallel max error (should be zero) on global rank '
          '{}: {}'.format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # ------------
    # Row parallel
    # ------------
    weight = torch.empty(output_size, input_size_coeff)
    set_random_seed(seed)
    mpu.layers._initialize_affine_weight(weight, output_size, input_size,
                                         input_size_coeff, 1,
                                         torch.nn.init.normal_)
    # Target.
    set_random_seed(seed)
    master_weight = torch.empty(output_size, input_size)
    torch.nn.init.normal_(master_weight)
    rank = mpu.get_model_parallel_rank()
    my_weight = torch.split(master_weight, input_size_coeff,
                            dim=1)[rank].contiguous().clone()

    # Compare.
    error = weight.sub(my_weight).abs().max()
    torch.distributed.barrier()
    print('   row parallel max error (should be zero) on global rank '
          '{}: {}'.format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(' >> passed the test :-)')


class IdentityLayer2D(torch.nn.Module):
    def __init__(self, m , n):
        super(IdentityLayer2D, self).__init__()
        self.weight = Parameter(torch.Tensor(m, n))
        torch.nn.init.xavier_normal_(self.weight)
    def forward(self):
        return self.weight


def test_column_parallel_linear(model_parallel_size):

    mpu.initialize_model_parallel(model_parallel_size)
    if torch.distributed.get_rank() == 0:
        print('> testing ColumnParallelLinear with model parallel '
              'size: {}'.format(model_parallel_size))
    model_parallel_size = mpu.get_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)
    input_size_coeff = 13
    input_size = input_size_coeff * model_parallel_size
    output_size_coeff = 17
    output_size = output_size_coeff * model_parallel_size
    batch_size = 7

    # Network
    identity_layer = IdentityLayer2D(batch_size, input_size).cuda()
    linear_layer = mpu.ColumnParallelLinear(
        input_size, output_size, keep_master_weight_for_test=True).cuda()
    loss_weight = torch.randn([batch_size, output_size]).cuda()
    # Forward
    input_ = identity_layer()
    output = linear_layer(input_)
    loss = torch.mul(output, loss_weight).sum()
    # Backward
    loss.backward()

    # Values.
    dLdY = loss_weight
    X = identity_layer.weight
    A = linear_layer.master_weight.cuda()
    dLdA = torch.matmul(dLdY.t(), X)
    dLdb = torch.matmul(torch.ones(batch_size, 1).cuda().t(), dLdY).view(-1)
    dLdX = torch.matmul(dLdY, A)

    rank = mpu.get_model_parallel_rank()
    my_dLdA = torch.split(dLdA, output_size_coeff,
                          dim=0)[rank].contiguous().clone()
    error = my_dLdA.sub(linear_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   error in dLdA on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    my_dLdb = torch.split(dLdb, output_size_coeff,
                          dim=0)[rank].contiguous().clone()
    error = my_dLdb.sub(linear_layer.bias.grad).abs().max()
    torch.distributed.barrier()
    print('   error in dLdb on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    error = dLdX.sub(identity_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   error in dLdX on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(' >> passed the test :-)')


def test_row_parallel_linear(model_parallel_size):

    mpu.initialize_model_parallel(model_parallel_size)
    if torch.distributed.get_rank() == 0:
        print('> testing RowParallelLinear with model parallel '
              'size: {}'.format(model_parallel_size))
    model_parallel_size = mpu.get_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)
    input_size_coeff = 13
    input_size = input_size_coeff * model_parallel_size
    output_size_coeff = 17
    output_size = output_size_coeff * model_parallel_size
    batch_size = 7

    # Network
    identity_layer = IdentityLayer2D(batch_size, input_size).cuda()
    linear_layer = mpu.RowParallelLinear(
        input_size, output_size, keep_master_weight_for_test=True).cuda()
    loss_weight = torch.randn([batch_size, output_size]).cuda()
    # Forward
    input_ = identity_layer()
    output = linear_layer(input_)
    loss = torch.mul(output, loss_weight).sum()
    # Backward
    loss.backward()

    # Values.
    dLdY = loss_weight
    X = identity_layer.weight
    A = linear_layer.master_weight.cuda()
    dLdA = torch.matmul(dLdY.t(), X)
    dLdb = torch.matmul(torch.ones(batch_size, 1).cuda().t(), dLdY).view(-1)
    dLdX = torch.matmul(dLdY, A)

    rank = mpu.get_model_parallel_rank()
    my_dLdA = torch.split(dLdA, input_size_coeff,
                          dim=1)[rank].contiguous().clone()
    error = my_dLdA.sub(linear_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   error in dLdA on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    error = dLdb.sub(linear_layer.bias.grad).abs().max()
    torch.distributed.barrier()
    print('   error in dLdb on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    error = dLdX.sub(identity_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   error in dLdX on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(' >> passed the test :-)')


class IdentityLayer3D(torch.nn.Module):
    def __init__(self, m , n, k):
        super(IdentityLayer3D, self).__init__()
        self.weight = Parameter(torch.Tensor(m, n, k))
        torch.nn.init.xavier_normal_(self.weight)
    def forward(self):
        return self.weight


def parallel_self_attention(model_parallel_size, num_att_heads_per_partition,
                            hidden_size_per_att_head, dropout_prob, batch_size,
                            sequence_length):
    mpu.initialize_model_parallel(model_parallel_size)
    model_parallel_size = mpu.get_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)

    num_att_heads = num_att_heads_per_partition * \
                    torch.distributed.get_world_size()
    hidden_size = hidden_size_per_att_head * num_att_heads

    # Network
    identity_layer = IdentityLayer3D(batch_size, sequence_length,
                                     hidden_size).cuda()
    attention_layer = mpu.BertParallelSelfAttention(hidden_size, num_att_heads,
                                                dropout_prob).cuda()
    loss_weight = torch.randn([batch_size, sequence_length, hidden_size]).cuda()
    attention_mask = torch.randn([batch_size, 1, 1, sequence_length]).cuda()
    # Forward
    input_ = identity_layer()
    output = attention_layer(input_, attention_mask)
    loss = torch.mul(output, loss_weight).sum()
    # Backward
    loss.backward()

    rank = mpu.get_model_parallel_rank()
    mpu.destroy_model_parallel()
    return rank, hidden_size, model_parallel_size, loss, \
        attention_layer, identity_layer


def test_parallel_self_attention(model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing ParallelSelfAttention with model parallel '
              'size: {}'.format(model_parallel_size))

    num_att_heads_per_partition = 3
    hidden_size_per_att_head = 7
    dropout_prob = 0.0 # has to be zero
    batch_size = 5
    sequence_length = 13

    rank_1, hideen_size_1, model_parallel_size_1, loss_1, \
        attention_layer_1, identity_layer_1 =parallel_self_attention(
            1, num_att_heads_per_partition,
            hidden_size_per_att_head, dropout_prob, batch_size, sequence_length)

    rank, hidden_size, model_parallel_size, loss, \
        attention_layer, identity_layer =parallel_self_attention(
            model_parallel_size, num_att_heads_per_partition,
            hidden_size_per_att_head, dropout_prob, batch_size, sequence_length)
    assert hideen_size_1 == hidden_size

    error = loss_1.sub(loss).abs().max()
    torch.distributed.barrier()
    print('   loss error on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 5.0e-6

    my_lin_grad_list = torch.split(
        attention_layer_1.query_key_value.weight.grad,
        hidden_size // model_parallel_size, 0)[rank::model_parallel_size]
    my_lin_grad = torch.cat(my_lin_grad_list, dim=0)
    error = my_lin_grad.sub(
        attention_layer.query_key_value.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   weight gradient error on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 5.0e-6

    error = identity_layer_1.weight.grad.sub(
        identity_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   input gradient error on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 5.0e-6

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(' >> passed the test :-)')

def parallel_transformer(model_parallel_size, num_att_heads_per_partition,
                         hidden_size_per_att_head, batch_size, sequence_length):

    mpu.initialize_model_parallel(model_parallel_size)
    model_parallel_size = mpu.get_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)

    num_att_heads = num_att_heads_per_partition * \
                    torch.distributed.get_world_size()
    hidden_size = hidden_size_per_att_head * num_att_heads
    intermediate_size = 4 * hidden_size

    # Network
    identity_layer = IdentityLayer3D(batch_size, sequence_length,
                                     hidden_size).cuda()
    transformer_layer = mpu.BertParallelTransformerLayer(
        hidden_size, intermediate_size, num_att_heads, 0.0, 0.0,
        torch.nn.functional.relu, 1.0e-5).cuda()

    loss_weight = torch.randn([batch_size, sequence_length, hidden_size]).cuda()
    attention_mask = torch.randn([batch_size, 1, 1, sequence_length]).cuda()
    # Forward
    input_ = identity_layer()
    output = transformer_layer(input_, attention_mask)
    loss = torch.mul(output, loss_weight).sum()
    # Backward
    loss.backward()

    rank = mpu.get_model_parallel_rank()
    mpu.destroy_model_parallel()
    return rank, hidden_size, model_parallel_size, loss, \
        transformer_layer, identity_layer


def test_parallel_transformer_layer(model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing ParallelTransformerLayer with model parallel '
              'size: {}'.format(model_parallel_size))

    num_att_heads_per_partition = 3
    hidden_size_per_att_head = 7
    batch_size = 5
    sequence_length = 13

    rank_1, hidden_size_1, model_parallel_size_1, loss_1, \
        transformer_layer_1, identity_layer_1 = parallel_transformer(
            1, num_att_heads_per_partition,
            hidden_size_per_att_head, batch_size, sequence_length)

    rank, hidden_size, model_parallel_size, loss, \
        transformer_layer, identity_layer = parallel_transformer(
            model_parallel_size, num_att_heads_per_partition,
            hidden_size_per_att_head, batch_size, sequence_length)

    error = loss_1.sub(loss).abs().max()
    torch.distributed.barrier()
    print('   loss error on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 5.0e-5, 'error: {}'.format(error)

    error = identity_layer_1.weight.grad.sub(
        identity_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   input gradient error on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 5.0e-5, 'error: {}'.format(error)

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(' >> passed the test :-)')


if __name__ == '__main__':

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    initialize_distributed()
    world_size = torch.distributed.get_world_size()

    print_separator('test initialize affine weight')
    model_parallel_size = 1
    while model_parallel_size <= world_size:
        test_initialize_affine_weight(model_parallel_size)
        model_parallel_size *= 2

    model_parallel_size = 1
    while model_parallel_size <= world_size:
        print_separator('test parallel embedding')
        test_parallel_embedding(model_parallel_size)
        model_parallel_size *= 2

    print_separator('test column-parallel linear')
    model_parallel_size = 1
    while model_parallel_size <= world_size:
        test_column_parallel_linear(model_parallel_size)
        model_parallel_size *= 2

    print_separator('test row-parallel linear')
    model_parallel_size = 1
    while model_parallel_size <= world_size:
        test_row_parallel_linear(model_parallel_size)
        model_parallel_size *= 2

    print_separator('test parallel self-attention')
    model_parallel_size = 1
    while model_parallel_size <= world_size:
        test_parallel_self_attention(model_parallel_size)
        model_parallel_size *= 2

    print_separator('test parallel transformer')
    model_parallel_size = 1
    while model_parallel_size <= world_size:
        test_parallel_transformer_layer(model_parallel_size)
        model_parallel_size *= 2
