# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import deepspeed.comm as dist
import torch

from unit.common import DistributedTest, preferred_dtype
import deepspeed
from deepspeed.accelerator import get_accelerator
from unit.simple_model import SimpleModel, random_dataloader
from deepspeed.utils import groups
from contextlib import contextmanager
from torch import nn
from deepspeed.module_inject.layers import LinearAllreduce, LinearLayer

# test group         done
# test daloader check      done
# test fwd/ bwd
# test gather/partition
# test save/load ckpt
# test save model
# test grad_norm


@contextmanager
def should_assert_with_msg(expected_message):
    try:
        yield  
    except AssertionError as e:
        # ignoe blank
        if dist.get_rank()==0:
            print(expected_message)
            print(str(e))
        if str(e) == expected_message:
            pass  
        else:
            raise e  
        
class TestTpParallelStates(DistributedTest):
    world_size = 4
    def test(self):
        tp_size=4

        dp_size = 4 / dist.get_world_size()
        hidden_dim = 128
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 0,
                "autotp_size":tp_size
          
            }
        }
        model = SimpleModel(hidden_dim=hidden_dim)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        assert groups.get_tensor_model_parallel_world_size()==tp_size
        assert groups.get_data_parallel_world_size()==dp_size

        
class TestTpDataloaderCorrectness(DistributedTest):
    world_size = 4
    reuse_dist_env = True
    
    def test(self):
        tp_size=4
        hidden_dim = 128
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "zero_optimization": {
                "stage": 0,
                "autotp_size":tp_size
          
            }
        }
        if preferred_dtype() is torch.float16:
            config_dict["fp16"] = {"enabled": True}
        elif preferred_dtype() is torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

        model = SimpleModel(hidden_dim=hidden_dim)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        data_loader = random_dataloader(model=model,
                                        total_samples=3,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=preferred_dtype())
        dist.barrier()
        with should_assert_with_msg("Data inconsistency within the TP group. Please check the Dataloader implementation to ensure consistency."):
            for batch in data_loader:
                # batch[0].requires_grad = requires_grad
                batch[0]+= dist.get_rank()
                model(batch[0], batch[1])
                
        model = SimpleModel(hidden_dim=hidden_dim)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        data_loader = random_dataloader(model=model,
                                        total_samples=3,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=preferred_dtype())
        for batch in data_loader:
            dist.broadcast(batch[0],src=groups.get_tensor_model_parallel_src_rank(),group=groups.get_tensor_model_parallel_group())
            dist.broadcast(batch[1],src=groups.get_tensor_model_parallel_src_rank(),group=groups.get_tensor_model_parallel_group())
            model(batch[0], batch[1])

class TestTpLayerfwdandbwd(DistributedTest):
    world_size = 4
    reuse_dist_env = True
    
    def test1(self):
        tp_size=4
        hidden_dim = 128
        batch_size_per_device=1
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "zero_optimization": {
                "stage": 0,
                "autotp_size":tp_size
          
            }
        }
        if preferred_dtype() is torch.float16:
            config_dict["fp16"] = {"enabled": True}
        elif preferred_dtype() is torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

        torch.manual_seed(42)
        model = SimpleModel(hidden_dim=hidden_dim)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        input = torch.randn(batch_size_per_device, hidden_dim, dtype=preferred_dtype(), requires_grad=True,device="cpu")

        torch_linear = nn.Linear(hidden_dim, hidden_dim, dtype=preferred_dtype(),device="cpu", bias=None)
        torch_out = torch_linear(input)
        torch_loss=torch_out.sum()
        torch_loss.backward()
        torch_norm =  torch.norm(torch_linear.weight.grad)
        torch_linear.zero_grad()

        linear = LinearAllreduce(torch_linear, groups.get_tensor_model_parallel_group())
        input.to(get_accelerator().current_device())
        
        input_=torch.chunk(input, tp_size, dim=-1)[groups.get_tensor_model_parallel_rank()]
        out = linear(input_.to(get_accelerator().current_device()))
        loss = out.sum()
        loss.backward()
        norm = torch.norm(linear.weight.grad)
        norm_pow =norm**2
        dist.all_reduce(norm_pow,group=groups.get_tensor_model_parallel_group())
        norm=torch.sqrt(norm_pow)
        assert torch.equal(norm, torch_norm.to(get_accelerator().current_device()))
        assert torch.equal(out, torch_out.to(get_accelerator().current_device()))
    def test2(self):
        
        tp_size=4
        hidden_dim = 128
        batch_size_per_device=1
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "zero_optimization": {
                "stage": 0,
                "autotp_size":tp_size
          
            }
        }
        if preferred_dtype() is torch.float16:
            config_dict["fp16"] = {"enabled": True}
        elif preferred_dtype() is torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

        torch.manual_seed(42)
        model = SimpleModel(hidden_dim=hidden_dim)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        input = torch.randn(batch_size_per_device, hidden_dim, dtype=preferred_dtype(), requires_grad=True,device="cpu")

        
        torch_linear = nn.Linear(hidden_dim, hidden_dim, dtype=preferred_dtype(),device="cpu", bias=None)
        torch_out = torch_linear(input)
        torch_loss=torch_out.sum()
        torch_loss.backward()
        torch_norm =  torch.norm(torch_linear.weight.grad)
        torch_linear.zero_grad()
        
        
        linear = LinearLayer(torch_linear, groups.get_tensor_model_parallel_group())
        
        out = linear(input.to(get_accelerator().current_device()))
        
        loss = out.sum()
        loss.backward()
        norm = torch.norm(linear.weight.grad)
        norm_pow =norm**2
        dist.all_reduce(norm_pow,group=groups.get_tensor_model_parallel_group())
        
        

        
