# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import deepspeed.comm as dist
from deepspeed import initialize
from transformers import AutoModel
from unit.common import DistributedTest
from deepspeed.sequence.layer import _SeqAllToAll
from deepspeed.sequence.fpdt_layer import _FPDTGPUOffloadingAttentionImpl_
from unit.util import skip_on_arch


#Use mesh device to create data and sequence parallel group
class TestUlyssesUtils(DistributedTest):
    world_size = 2

    def test_mesh_device_creation(self) -> None:
        skip_on_arch(min_arch=8)
        model = AutoModel.from_pretrained('bert-base-uncased')
        sp_size = 1
        dp_size = 2
        ds_engine, _, _, _ = initialize(
            model=model,
            config_params={
                "train_batch_size": 8,
                "data_parallel_size": dp_size,
                "sequence_parallel_size": sp_size
            },
        )
        assert ds_engine.seq_parallel_group is not None
        assert ds_engine.data_parallel_group is not None
        assert dist.get_world_size(group=ds_engine.seq_parallel_group) == sp_size
        assert dist.get_world_size(group=ds_engine.data_parallel_group) == dp_size
        assert dist.get_world_size() == sp_size * dp_size


#Sweep b,s,h,d to test all2all consistency
@pytest.mark.parametrize("d0", [2, 4])  #batch or sequence dimension
@pytest.mark.parametrize("d1", [4, 8])  #batch or sequence dimension
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [16, 32])
class TestUlyssesAll2All(DistributedTest):
    world_size = 2

    def test_alltoall_output_consistency(self, d0: int, d1: int, head_dim: int, num_heads: int) -> None:
        skip_on_arch(min_arch=8)
        model = AutoModel.from_pretrained('bert-base-uncased')
        ds_engine, _, _, _ = initialize(model=model, config_params={"train_batch_size": 8}, mesh_param=(2, 2))
        #4D tensor : b,s,h,d or s,b,h,d
        input_tensor = torch.randn(d0, d1, num_heads, head_dim, device=ds_engine.device)
        scatter_idx = 2
        batch_dim_idx = 0
        outputs = []
        seq_dims = [0]  #seq first API
        #TODO: Add support for batch first (that seq_dims=[0,1]) after PR for bs>1 issue with batch first is fixed
        ## See discussion in : https://github.com/microsoft/DeepSpeed/issues/5808
        for seq_dim in seq_dims:
            gather_idx = seq_dim
            #first all2all: sequence parallel to head parallel
            s2h_tensor = _SeqAllToAll.apply(ds_engine.seq_parallel_group, input_tensor, scatter_idx, gather_idx,
                                            batch_dim_idx)

            #No op
            # second all2all: head parallel to sequence parallel
            h2s_tensor = _SeqAllToAll.apply(ds_engine.seq_parallel_group, s2h_tensor, gather_idx, scatter_idx,
                                            batch_dim_idx)
            print(
                f'[{dist.get_rank()}] s={seq_dim} input: {input_tensor.shape} s2h: {s2h_tensor.shape} h2s_tensor: {h2s_tensor.shape}'
            )
            outputs.append(h2s_tensor)

        # Check outputs are the same as input
        for i in range(1, len(outputs)):
            assert torch.allclose(input_tensor, outputs[i]), f"Outputs differ for sequence dim {seq_dims[i]}"


@pytest.mark.parametrize("d0", [1, 4])  #batch dimension
@pytest.mark.parametrize("d1", [2048, 4096])  #sequence dimension
@pytest.mark.parametrize("chunk_size", [512, 1024])  #size of chunk
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [16, 32])
class TestFPDTAttention(DistributedTest):
    world_size = 1

    def test_FPDT_attention_offloading_output_consistency(self, d0: int, d1: int, chunk_size: int, head_dim: int,
                                                          num_heads: int) -> None:
        skip_on_arch(min_arch=8)

        try:
            from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
        except ImportError:
            _flash_attn_forward = None
            _flash_attn_backward = None

        if _flash_attn_forward is None or _flash_attn_backward is None:
            pytest.skip("Flash Attention is not available.")

        model = AutoModel.from_pretrained('bert-base-uncased')
        ds_engine, _, _, _ = initialize(
            model=model,
            config_params={
                "train_batch_size": 8,
                "data_parallel_size": 1,
                "sequence_parallel_size": 1
            },
        )
        #3D tensor : l, b, d
        dim = head_dim * num_heads
        input_tensor = torch.randn(d1, d0, dim, device=ds_engine.device)
        spg = ds_engine.seq_parallel_group

        qkv_linear_weight = Parameter(torch.empty(dim + 2 * dim, dim, device=ds_engine.device, dtype=torch.half))

        qkv_linear_bias = Parameter(torch.empty(dim + 2 * dim, device=ds_engine.device, dtype=torch.half))

        num_chunks_attn = input_tensor.shape[0] * dist.get_world_size(spg) // chunk_size
        fpdt_output = _FPDTGPUOffloadingAttentionImpl_.apply(input_tensor, None, None, None, spg, 2, 0, dim, dim,
                                                             head_dim, dim, qkv_linear_weight, qkv_linear_bias, 0,
                                                             num_chunks_attn, True)

        # baseline
        qkv = torch.matmul(input_tensor, qkv_linear_weight.t()) + qkv_linear_bias
        q = qkv[:, :, :dim].contiguous().reshape(qkv.shape[0], qkv.shape[1], -1, head_dim).permute(1, 2, 0,
                                                                                                   3).contiguous()
        k = qkv[:, :, dim:dim * 2].contiguous().reshape(qkv.shape[0], qkv.shape[1], -1,
                                                        head_dim).permute(1, 2, 0, 3).contiguous()
        v = qkv[:, :, dim * 2:dim * 3].contiguous().reshape(qkv.shape[0], qkv.shape[1], -1,
                                                            head_dim).permute(1, 2, 0,
                                                                              3).contiguous()  # b, nhead, l, d

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(dim, dtype=torch.half))

        causal_mask = torch.triu(torch.ones(d1, d1), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(causal_mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        assert torch.allclose(fpdt_output, output)
