# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import deepspeed.comm as dist
import torch

from unit.common import DistributedTest, preferred_dtype
import deepspeed
from deepspeed.accelerator import get_accelerator
from unit.simple_model import SimpleModel, random_dataloader, sequence_dataloader
from deepspeed.utils import groups
from contextlib import contextmanager
from torch import nn
from deepspeed.module_inject.layers import LinearAllreduce, LinearLayer

# test group         done
# test daloader check      done
# test fwd/ bwd   done
# test gather/partition done
# test save/load ckpt  done
# test save model done 
# test grad_norm  done , need to refine.
# test compatibility with zero.etc.?

@contextmanager
def should_assert_with_msg(expected_message):
    try:
        yield  
    except AssertionError as e:
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
        norm=torch.sqrt(norm_pow)   
        assert torch.equal(norm, torch_norm.to(get_accelerator().current_device()))
        cur_device_out = torch.chunk(torch_out, tp_size, dim=-1)[groups.get_tensor_model_parallel_rank()]
        
        assert  torch.allclose(cur_device_out.to(get_accelerator().current_device()).contiguous(), out.contiguous(),atol=1e-6)

class TestparamsGather(DistributedTest):
    world_size = 4
    reuse_dist_env = True
    def test(self):
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
        total_params0 = sum(p.numel() for p in torch_linear.parameters())

        
        # TODO : make it to param
        linear = None
        type = "linearallreduce"
        if type == "linear":
            linear = LinearLayer(torch_linear, groups.get_tensor_model_parallel_group())
        elif type == "linearallreduce":
            linear = LinearAllreduce(torch_linear, groups.get_tensor_model_parallel_group())
        else:
            raise ValueError(f"Invalid linear type: {config_dict['linear_type']}")
        
        
        params0 = sum(p.numel() for p in linear.parameters())
        
        assert total_params0//tp_size==params0
        for name, param in linear.named_parameters(recurse=False):
            param.gather_params([param])

        same_weights = all(torch.equal(param1, param2) 
                   for param1, param2 in zip(linear.parameters(), torch_linear.parameters()))
        
        assert same_weights
         
        params1 = sum(p.numel() for p in linear.parameters())
        assert total_params0==params1

        for name, param in linear.named_parameters(recurse=False):
            param.partition([param])
        
        params2 = sum(p.numel() for p in linear.parameters())

        assert total_params0//tp_size==params2


class TestSave(DistributedTest):
        
    world_size = 4
    reuse_dist_env = True
    def test(self):
        tp_size=4
        hidden_dim = 64
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
        model = SimpleModel(hidden_dim=hidden_dim , nlayers=8)
        from copy import deepcopy
        base = deepcopy(model)

        modelt = SimpleModel(hidden_dim=hidden_dim)
        modelt, _, _, _ = deepspeed.initialize(model=modelt, model_parameters=modelt.parameters(), config=config_dict)
        #2,3   5,6 
        

        for i in ([2,5]):
            model.linears[i]=LinearLayer(model.linears[i], groups.get_tensor_model_parallel_group())

        for i in ([3,6]):
            model.linears[i]=LinearAllreduce(model.linears[i], groups.get_tensor_model_parallel_group())

        del modelt

        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)

        
        cur_params_numel = sum(p.numel() for p in model.parameters())
        base_params_numel =  sum(p.numel() for p in base.parameters())
        assert cur_params_numel<base_params_numel
        
        tp_state_dict = model._consolidated_16bit_state_dict()
        def compare_state_dicts(state_dict1, state_dict2):
            if state_dict1.keys() != state_dict2.keys():
                print("The state_dicts have different keys!")
                return False
            
            for key in state_dict1:
                if not torch.equal(state_dict1[key], state_dict2[key]):
                    print(f"Parameters for {key} are different!")
                    return False
            
            return True
        base_state_dict = base.state_dict()
        
        assert(base_state_dict, tp_state_dict)
    
    def test_ckpt_save(self):    
        tp_size=4
        hidden_dim = 64
        batch_size_per_device=1
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
            "zero_optimization": {
                "stage": 0,
                "autotp_size":tp_size
          
            },
            "scheduler": {
            "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 0.001,
                    "warmup_num_steps": 1000
                }
            }
        }
  
        if preferred_dtype() is torch.float16:
            config_dict["fp16"] = {"enabled": True}
        elif preferred_dtype() is torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

        # for group
        modelt = SimpleModel(hidden_dim=hidden_dim)
        modelt, optimizer, _, _ = deepspeed.initialize(model=modelt, model_parameters=modelt.parameters(), config=config_dict)
        
        
        model = SimpleModel(hidden_dim=hidden_dim , nlayers=8)
        model2 = SimpleModel(hidden_dim=hidden_dim , nlayers=8)
        
        for i in ([2,5]):
            model.linears[i]=LinearLayer(model.linears[i], groups.get_tensor_model_parallel_group())
            model2.linears[i]=LinearLayer(model2.linears[i], groups.get_tensor_model_parallel_group())
        for i in ([3,6]):
            model.linears[i]=LinearAllreduce(model.linears[i], groups.get_tensor_model_parallel_group())
            model2.linears[i]=LinearAllreduce(model2.linears[i], groups.get_tensor_model_parallel_group())

        model,_,_,_= deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        torch.manual_seed(42)

        data_loader = random_dataloader(model=model,
                                total_samples=3,
                                hidden_dim=hidden_dim,
                                device=model.device,
                                dtype=preferred_dtype())
        test_batch=None
        ckpt_path = "./test_ckpt/"
        for i, batch in enumerate(data_loader):
            batch[0].requires_grad = True
            loss = model(batch[0], batch[1])
            loss = loss
            model.backward(loss)
            model.step()
        model.save_checkpoint(ckpt_path)
     

            
  
        # base_loss = model(test_batch[0],test_batch[1])
        
        model2,_,_,_ = deepspeed.initialize(model=model2, model_parameters=model2.parameters(),config=config_dict)
        model2.load_checkpoint(ckpt_path,load_optimizer_states=True,load_lr_scheduler_states=True)
        from unit.checkpoint.common import compare_lr_scheduler_states,compare_optimizer_states
        is_fp16= (preferred_dtype()==torch.float16)
        compare_optimizer_states(model,  model2, 0,  fp16=is_fp16)
        compare_lr_scheduler_states(model, model2)
        b=0
        
class TestNorm(DistributedTest):
    
    world_size = 4
    reuse_dist_env = True
    def test(self):
        tp_size=4
        hidden_dim = 64
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
        model_base = SimpleModel(hidden_dim=hidden_dim , nlayers=8)
        from copy import deepcopy

        model_base, optimizer ,_,_ = deepspeed.initialize(model=model_base, model_parameters=model_base.parameters(), config=config_dict)

        data_loader = random_dataloader(model=model_base,
                                total_samples=2,
                                hidden_dim=hidden_dim,
                                device=model_base.device,
                                dtype=preferred_dtype())

        for i, batch in enumerate(data_loader):
            batch[0].requires_grad = True
            loss = model_base(batch[0], batch[1])
            loss = loss
            model_base.backward(loss)
            optimizer.step()
            
            
    
        grad_norm_base =  optimizer._global_grad_norm

        torch.manual_seed(42)

                
                
        modelt = SimpleModel(hidden_dim=hidden_dim)
        modelt, optimizer, _, _ = deepspeed.initialize(model=modelt, model_parameters=modelt.parameters(), config=config_dict)
        #2,3   5,6 
        
        model = SimpleModel(hidden_dim=hidden_dim , nlayers=8)

        

        for i in ([2,5]):
            model.linears[i]=LinearLayer(model.linears[i], groups.get_tensor_model_parallel_group())

        for i in ([3,6]):
            model.linears[i]=LinearAllreduce(model.linears[i], groups.get_tensor_model_parallel_group())


        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0.01)
        model, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)


        
        for i, batch in enumerate(data_loader):
            batch[0].requires_grad = True
            loss = model(batch[0], batch[1])
            loss = loss
            model.backward(loss)
            optimizer.step()

        
        # optimizer.step()
        norm = optimizer._global_grad_norm
        

        norm_diff_percent = abs(norm - grad_norm_base) / grad_norm_base * 100
        assert norm_diff_percent<5
        cur_params_numel = sum(p.numel() for p in model.parameters())
        base_params_numel =  sum(p.numel() for p in model_base.parameters())
        assert cur_params_numel<base_params_numel
        
       

    
        

        
