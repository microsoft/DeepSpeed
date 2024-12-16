# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import deepspeed.comm as dist
import torch
import math

from unit.common import DistributedTest, preferred_dtype
import deepspeed
from deepspeed.accelerator import get_accelerator
from unit.simple_model import SimpleModel, random_dataloader, sequence_dataloader
from deepspeed.utils import groups
from contextlib import contextmanager
from torch import nn
from deepspeed.module_inject.layers import LinearAllreduce, LinearLayer, set_autotp_mode

# test group         done
# test daloader check      done
# test fwd/ bwd   done
# test gather/partition done
# test save/load ckpt  done
# test save model done 
# test grad_norm  done , need to refine.
# test compatibility with zero.etc.?
# todo:add more batch_size/hidden_dim  test

class SequentialLinearModel(torch.nn.Module):

    def __init__(self, hidden_dim, empty_grad=False, nlayers=1):
        super(SequentialLinearModel, self).__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim,bias=None) for i in range(nlayers)])
        if empty_grad:
            self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim,bias=None)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.empty_grad = empty_grad

    def forward(self, x, y):
        if len(self.linears) == 1:
            x = self.linears[0](x)
        else:
            for i, l in enumerate(self.linears):
                x = self.linears[i](x)
        return self.cross_entropy_loss(x, y)


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
        set_autotp_mode(training=True)
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
        set_autotp_mode(training=True)
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

def process_linear_layer(hidden_dim, input):
    torch_linear = nn.Linear(hidden_dim, hidden_dim, dtype=preferred_dtype(), device="cpu", bias=None)
    torch_out = torch_linear(input)
    torch_loss = torch_out.sum()
    torch_loss.backward()
    torch_norm = torch.norm(torch_linear.weight.grad)
    torch_linear.zero_grad()
    return torch_linear, torch_out, torch_norm


class TestTpLayerFwdBwd(DistributedTest):
    world_size = 4
    reuse_dist_env = True
    def testRowParallel(self):
        tp_size=4
        hidden_dim = 128
        batch_size_per_device=1
        set_autotp_mode(training=True)
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

        model = SequentialLinearModel(hidden_dim=hidden_dim)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        input = torch.randn(batch_size_per_device, hidden_dim, dtype=preferred_dtype(), requires_grad=True,device="cpu")

        torch_linear, torch_out, torch_norm = process_linear_layer(hidden_dim, input)
        
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
        assert torch.allclose(out, torch_out.to(get_accelerator().current_device()),atol=1e-3)
 
    def testColumnParallel(self):
        
        tp_size=4
        hidden_dim = 128
        batch_size_per_device=1
        set_autotp_mode(training=True)
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

        model = SequentialLinearModel(hidden_dim=hidden_dim)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        input = torch.randn(batch_size_per_device, hidden_dim, dtype=preferred_dtype(), requires_grad=True,device="cpu")

        torch_linear, torch_out, torch_norm = process_linear_layer(hidden_dim, input)
        
        linear = LinearLayer(torch_linear, groups.get_tensor_model_parallel_group())
        
        out = linear(input.to(get_accelerator().current_device()))
        loss = out.sum()
        loss.backward()
        norm = torch.norm(linear.weight.grad)
        norm_pow =norm**2
        dist.all_reduce(norm_pow,group=groups.get_tensor_model_parallel_group())
        norm=torch.sqrt(norm_pow)   
        cur_device_out = torch.chunk(torch_out, tp_size, dim=-1)[groups.get_tensor_model_parallel_rank()]
        assert torch.equal(norm, torch_norm.to(get_accelerator().current_device()))
        assert  torch.allclose(cur_device_out.to(get_accelerator().current_device()).contiguous(), out.contiguous(),atol=1e-3)

class TestParamsGather(DistributedTest):
    world_size = 4
    reuse_dist_env = True
    @pytest.mark.parametrize("layer_type", ["linear", "linearallreduce"])
    def test(self, layer_type):
        tp_size=4
        hidden_dim = 128
        set_autotp_mode(training=True)
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
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
        model = SequentialLinearModel(hidden_dim=hidden_dim)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)

        torch_linear = nn.Linear(hidden_dim, hidden_dim, dtype=preferred_dtype(),device="cpu", bias=None)
        total_params = sum(p.numel() for p in torch_linear.parameters())

        tp_layer = None
        if layer_type == "linear":
            tp_layer = LinearLayer(torch_linear, groups.get_tensor_model_parallel_group())
        elif layer_type == "linearallreduce":
            tp_layer = LinearAllreduce(torch_linear, groups.get_tensor_model_parallel_group())
        else:
            raise ValueError(f"Invalid linear type: {config_dict['linear_type']}")
        
        tp_params = sum(p.numel() for p in tp_layer.parameters())
        
        assert total_params//tp_size==tp_params
        for name, param in tp_layer.named_parameters(recurse=False):
            param.gather_params([param])

        is_same_weights = all(torch.equal(param1, param2) 
                   for param1, param2 in zip(tp_layer.parameters(), torch_linear.parameters()))
        
        assert is_same_weights
         
        params1 = sum(p.numel() for p in tp_layer.parameters())
        assert total_params==params1

        for name, param in tp_layer.named_parameters(recurse=False):
            param.partition([param])
        
        tp_params2 = sum(p.numel() for p in tp_layer.parameters())

        assert total_params//tp_size==tp_params2

def dummy_init_engine(config):
    # This is a dummy initialization function for the DeepSpeed engine.
    # We only need to use the config to initialize the distributed settings for the test.
    model = SequentialLinearModel(hidden_dim=8)
    model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)

def prepare_tp_model(hidden_dim, nlayers, linear_indices, allreduce_indices, group, return_global_copy=False):
    model = SequentialLinearModel(hidden_dim=hidden_dim, nlayers=nlayers).to(preferred_dtype())
    base_model=None
    from copy import deepcopy
    if return_global_copy:
        base_model = deepcopy(model)
    for i in linear_indices:
        layer = LinearLayer(model.linears[i], group)
        model.linears[i] = layer
    
    for i in allreduce_indices:
        layer = LinearAllreduce(model.linears[i], group)
        model.linears[i] = layer
    
    return model, base_model
class TestSave(DistributedTest):
    
    world_size = 4
    reuse_dist_env = True
    def test_save_original_weight(self):
        tp_size=4
        hidden_dim = 64
        set_autotp_mode(training=True)
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
        dummy_init_engine(config_dict)
        torch.manual_seed(42)
        
        model, base_model=prepare_tp_model(hidden_dim, 8, [2,5], [3,6], groups.get_tensor_model_parallel_group(),return_global_copy=True)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)

        cur_params_numel = sum(p.numel() for p in model.parameters())
        base_params_numel =  sum(p.numel() for p in base_model.parameters())
        assert cur_params_numel<base_params_numel
        
        tp_state_dict = model._consolidated_16bit_state_dict()
        def compare_state_dicts(state_dict1, state_dict2):
            if state_dict1.keys() != state_dict2.keys():
                print("The state_dicts have different keys!")
                return False
            
            for key in state_dict1:
                if not torch.allclose(state_dict1[key], state_dict2[key],atol=1e-3):
                    assert state_dict1[key].device=="cpu"
                    print(f"Parameters for {key} are different!")
                    return False
            
            return True
        base_state_dict = base_model.state_dict()
        if dist.get_rank()==0:
        # we should consider the case when zero3 is used in the future.
            assert compare_state_dicts(base_state_dict, tp_state_dict), f"State_dict is not the same!"
        else:
            assert tp_state_dict is None, f"noly rank0 should have the state_dict"
      
    def test_ckpt_save(self):    
        tp_size=4
        hidden_dim = 64
        set_autotp_mode(training=True)
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

        dummy_init_engine(config_dict)

        trained_model, _=prepare_tp_model(hidden_dim, 8, [2,5], [3,6], groups.get_tensor_model_parallel_group())
        loaded_model, _=prepare_tp_model(hidden_dim, 8, [2,5], [3,6], groups.get_tensor_model_parallel_group())

        trained_model,_,_,_= deepspeed.initialize(model=trained_model, model_parameters=trained_model.parameters(), config=config_dict)
        torch.manual_seed(42)

        data_loader = random_dataloader(model=trained_model,
                                total_samples=3,
                                hidden_dim=hidden_dim,
                                device=trained_model.device,
                                dtype=preferred_dtype())
        ckpt_path = "./test_ckpt/"
        for i, batch in enumerate(data_loader):
            batch[0].requires_grad = True
            loss = trained_model(batch[0], batch[1])
            loss = loss
            trained_model.backward(loss)
            trained_model.step()
        trained_model.save_checkpoint(ckpt_path)
                 
        loaded_model,_,_,_ = deepspeed.initialize(model=loaded_model, model_parameters=loaded_model.parameters(),config=config_dict)
        loaded_model.load_checkpoint(ckpt_path,load_optimizer_states=True,load_lr_scheduler_states=True)
        from unit.checkpoint.common import compare_lr_scheduler_states,compare_optimizer_states
        compare_optimizer_states(trained_model,  loaded_model, hidden_dim,  fp16=(preferred_dtype()==torch.float16))
        compare_lr_scheduler_states(trained_model, loaded_model)
        
class TestTpGradNorm(DistributedTest):
    
    world_size = 4
    reuse_dist_env = True
    def test(self):
        tp_size=4
        hidden_dim = 64
        set_autotp_mode(training=True)
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
                "autotp_size":4  
            }
        }
        if preferred_dtype() is torch.float16:
            config_dict["fp16"] = {"enabled": True}
        elif preferred_dtype() is torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

        torch.manual_seed(42)
    
        dummy_init_engine(config=config_dict)
        tp_model,base_model = prepare_tp_model(hidden_dim, 8, [2,5], [3,6], groups.get_tensor_model_parallel_group(),return_global_copy=True)
        
        base_model, base_optimizer, _, _ = deepspeed.initialize(model=base_model, model_parameters=base_model.parameters(), config=config_dict)
        data_loader = random_dataloader(model=base_model,
                                total_samples=2,
                                hidden_dim=hidden_dim,
                                device=base_model.device,
                                dtype=preferred_dtype())
        with base_model.no_sync():
            # duplicate each rank training.
            for i, batch in enumerate(data_loader):
                batch[0].requires_grad = True
                loss = base_model(batch[0], batch[1])
                loss = loss
                base_model.backward(loss)
                base_model.step()
            
        base_norm =  base_optimizer._global_grad_norm
        
        base_model.destroy()
        
        tp_model, tp_optimizer, _, _ = deepspeed.initialize(model=tp_model, model_parameters=tp_model.parameters(), config=config_dict)
        for i, batch in enumerate(data_loader):
            batch[0].requires_grad = True
            loss = tp_model(batch[0], batch[1])
            loss = loss
            tp_model.backward(loss)
            tp_model.step()

        tp_norm = tp_optimizer._global_grad_norm
        
        assert math.isclose(base_norm,tp_norm,abs_tol=1e-3)
        tp_params_numel = sum(p.numel() for p in tp_model.parameters())
        base_params_numel =  sum(p.numel() for p in base_model.parameters())
        assert tp_params_numel<base_params_numel
