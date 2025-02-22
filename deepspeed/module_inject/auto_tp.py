# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Automatic Tensor Parallelism
import re

from torch import nn
from .replace_policy import replace_policies
from typing import Optional
import torch
from deepspeed import comm as dist
from .layers import LinearAllreduce, LinearLayer, LmHeadLinearAllreduce, Yuan_LinearAllreduce, Yuan_LinearLayer, GateUpPack_LinearLayer, Conv_LinearALlreduce, fused_LinearLayer, conv_LinearLayer
from deepspeed.accelerator import get_accelerator
from .fusedqkv_utils import require_tp_fused_qkvw
from deepspeed.module_inject.tp_shard import get_shard_size, get_shard_size_list
from deepspeed.utils import groups
from deepspeed.module_inject.layers import is_autotp_training_mode


def move(tensor, device, copy=True):
    if tensor.is_meta:
        return torch.empty_like(tensor, device=device)
    else:
        # Using new tensors help in freeing memory (after split for example) was done before by calling clone().
        # Using copy=True instead of clone() will help in case of cpu --> cpu.
        # Otherwise to() will not create a new copy for the view of the full tensor, and it will not be de-referenced.
        return tensor.to(device, copy=copy)


class ReplaceWithTensorSlicing:

    def __init__(self, mp_group=None, mp_size=1, out_dim=1, in_dim=0):
        if mp_group is not None:
            self.gpu_index = dist.get_rank(group=mp_group)
        else:
            self.gpu_index = 0
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.mp_size = mp_size

    def merge_assert(self, dim1, dim2):
        assert dim1 > dim2, \
            'Merging tensors is not allowed here! Please use deepspeed load_checkpoint\
            for merging your checkpoints before replacing the transformer layer with\
            inference-kernels'

    def strided_copy(self,
                     dst: Optional[torch.Tensor],
                     src: Optional[torch.Tensor],
                     num_splits: int,
                     int8: bool = False,
                     allocate_tensor: bool = False):
        if src is None:
            return src
        src_shape = src.shape
        dst_shape = dst.shape

        outer_dim = 0 if int8 else -1

        if allocate_tensor:
            dst = torch.empty_like(dst)

        src_split = torch.split(src.data, src.shape[outer_dim] // num_splits, dim=outer_dim)
        if (len(src_shape) == 2 and len(dst_shape) == 2):
            if src_shape[outer_dim] == dst_shape[self.out_dim]:
                try:
                    dst = dst.reshape(-1).data.copy_(src.data.reshape(-1)).reshape(src.shape)
                except:
                    print(dst.shape, src.shape)
                    exit()
                dst = torch.nn.parameter.Parameter(dst, requires_grad=False)
                if hasattr(src, 'scale'):
                    dst.scale = src.scale
                return dst
            self.merge_assert(src_shape[outer_dim], dst_shape[self.out_dim])
            qkv_size = dst_shape[self.out_dim] // num_splits
            qkv_split = [torch.split(src_s, qkv_size, dim=outer_dim) for src_s in src_split]
            weight_split = [
                torch.cat([qkv_s[i] for qkv_s in qkv_split], axis=outer_dim) for i in range(len(qkv_split[0]))
            ]
            dst = dst.reshape(-1).data.copy_(weight_split[self.gpu_index].contiguous().reshape(-1)).reshape(
                weight_split[self.gpu_index].shape)
        else:
            if src_shape[0] == dst_shape[0]:
                return torch.nn.parameter.Parameter(src)
            qkv_size = dst_shape[0] // num_splits
            qkv_split = [torch.split(src_s, qkv_size, dim=0) for src_s in src_split]
            bias_split = [torch.cat([qkv_s[i] for qkv_s in qkv_split], axis=0) for i in range(len(qkv_split[0]))]
            dst.data.copy_(bias_split[self.gpu_index].contiguous())

        dst = torch.nn.parameter.Parameter(dst, requires_grad=False)
        if hasattr(src, 'scale'):
            dst.scale = src.scale
        return dst

    def copy(self, dst, src, int8=False, allocate_tensor=False):
        if src is None:
            return src
        assert not dst.data.is_meta  # the torch.Tensor.copy_ method used below will silently fail on meta tensors
        if allocate_tensor:
            dst = torch.empty_like(dst)
        outer_dim = 0 if int8 else 1
        inner_dim = 1 if int8 else 0
        src_shape = src.shape
        dst_shape = dst.shape
        if (len(src_shape) == 2 and len(dst_shape) == 2):

            if src_shape[inner_dim] == dst_shape[self.in_dim] and src_shape[outer_dim] == dst_shape[self.out_dim]:
                dst = dst.reshape(-1).data.copy_(src.data.reshape(-1)).reshape(src.shape)
            else:
                if src_shape[inner_dim] != dst_shape[self.in_dim]:
                    self.merge_assert(src_shape[inner_dim], dst_shape[self.in_dim])
                    dst.data.copy_(src[:, self.gpu_index * dst_shape[self.in_dim]: (self.gpu_index + 1) * dst_shape[self.in_dim]] if inner_dim == 1 else \
                                   src[self.gpu_index * dst_shape[self.in_dim]: (self.gpu_index + 1) * dst_shape[self.in_dim], :])
                else:
                    self.merge_assert(src_shape[outer_dim], dst_shape[self.out_dim])
                    dst.data.copy_(src[:, self.gpu_index * dst_shape[self.out_dim]: (self.gpu_index + 1) * dst_shape[self.out_dim]] if outer_dim == 1 else \
                                   src[self.gpu_index * dst_shape[self.out_dim]: (self.gpu_index + 1) * dst_shape[self.out_dim], :])
        else:
            if src_shape[0] == dst_shape[0]:
                dst = src if src.dtype == dst.dtype else dst.data.copy_(src)
            else:
                dst.data.copy_(src[self.gpu_index * dst_shape[-1]:(self.gpu_index + 1) * dst_shape[-1]])
        dst = torch.nn.parameter.Parameter(dst, requires_grad=False)
        if hasattr(src, 'scale'):
            dst.scale = src.scale
        return dst


class Loading():

    def is_load_module(module):
        load_layers = [nn.Linear, nn.Embedding, nn.LayerNorm]
        load_layer_names = [
            "LPLayerNorm", "SharedEmbedding", "OPTLearnedPositionalEmbedding", "LlamaRMSNorm", "FalconLinear",
            "MistralRMSNorm", "T5LayerNorm", "MixtralRMSNorm", "Phi3RotaryEmbedding", "Phi3SuScaledRotaryEmbedding",
            "Phi3RMSNorm", "YuanRMSNorm", "YuanRotaryEmbedding", "Phi3LongRoPEScaledRotaryEmbedding", "Qwen2RMSNorm",
            "DeepseekV2RMSNorm", "DeepseekV3RMSNorm", "DeepseekV2YarnRotaryEmbedding", "DeepseekV3YarnRotaryEmbedding",
            "MoEGate"
        ]
        return module.__class__ in load_layers or module._get_name() in load_layer_names

    def load_buffer(module, state_dict, prefix):
        for name in module._buffers.keys():
            if module._buffers[name].data.is_meta:
                module._buffers[name] = torch.nn.parameter.Parameter(
                    data=torch.empty_like(module._buffers[name].data, device="cpu"),
                    requires_grad=module._buffers[name].data.requires_grad)
            if prefix + name in state_dict.keys():
                module._buffers[name].data.copy_(state_dict[prefix + name])

    def load(module, state_dict, prefix, mp_group=None):
        mp_replace = ReplaceWithTensorSlicing(mp_group=mp_group)
        if hasattr(module, 'weight'):
            if module.weight.data.is_meta:
                # meta tensor cannot be casted or copied to, so we need to replace it with a normal tensor here
                module.weight = torch.nn.parameter.Parameter(data=torch.empty_like(module.weight.data, device="cpu"),
                                                             requires_grad=module.weight.data.requires_grad)
                if 'query_key_value' in prefix:
                    module.weight = mp_replace.strided_copy(module.weight.data,
                                                            state_dict[prefix + 'weight'],
                                                            num_splits=3)
                else:
                    module.weight = mp_replace.copy(module.weight.data, state_dict[prefix + 'weight'])
        else:
            if hasattr(module, 'norm') and hasattr(module.norm, 'weight'):
                if module.norm.weight.data.is_meta:
                    # meta tensor cannot be casted or copied to, so we need to replace it with a normal tensor here
                    module.norm.weight = torch.nn.parameter.Parameter(
                        data=torch.empty_like(module.norm.weight.data, device="cpu"),
                        requires_grad=module.norm.weight.data.requires_grad)
                module.norm.weight = mp_replace.copy(module.norm.weight.data, state_dict[prefix + 'weight'])

        if prefix + 'bias' in state_dict.keys():
            if hasattr(module, 'bias'):
                if module.bias.data.is_meta:
                    # meta tensor cannot be casted or copied to, so we need to replace it with a normal tensor here
                    module.bias = torch.nn.parameter.Parameter(data=torch.empty_like(module.bias.data, device="cpu"),
                                                               requires_grad=module.bias.data.requires_grad)
                module.bias = mp_replace.copy(module.bias, state_dict[prefix + 'bias'])
            else:
                if hasattr(module, 'norm') and hasattr(module.norm, 'bias'):
                    if module.norm.bias.data.is_meta:
                        # meta tensor cannot be casted or copied to, so we need to replace it with a normal tensor here
                        module.norm.bias = torch.nn.parameter.Parameter(
                            data=torch.empty_like(module.norm.bias.data, device="cpu"),
                            requires_grad=module.norm.bias.data.requires_grad)
                    module.norm.bias = mp_replace.copy(module.norm.bias, state_dict[prefix + 'bias'])


class AutoTP():

    def __init__(self,
                 module,
                 all_reduce_linears,
                 prefix,
                 state_dict,
                 linear_layer_setting,
                 orig_layer_impl,
                 keep_module_on_host=False):
        self.module = module
        self.all_reduce_linears = all_reduce_linears
        self.prefix = prefix
        self.state_dict = state_dict

        self.mp_size = None
        self.mp_group = None
        self.linear_layer_setting = linear_layer_setting
        self.orig_layer_impl = orig_layer_impl
        self.linear_policies = None
        self.conv_linear_layer = False
        self.keep_module_on_host = keep_module_on_host

    def in_module_list(module, module_list):
        for item in module_list:
            if type(item).__name__ == type(module).__name__:
                return True
        return False

    def get_module_list(model):
        mlist = []
        for child in model.children():
            if isinstance(child, nn.ModuleList):
                for module in child.children():
                    if not mlist:
                        mlist = [module]
                    elif not AutoTP.in_module_list(module, mlist):
                        mlist = mlist + [module]
            else:
                mlist = mlist + AutoTP.get_module_list(child)
        return mlist

    def supported(model):
        unsupported = ['deberta', 'flaubert', 'fsmt', 'gpt2', 'led', 'longformer', 'xlm', 'xlnet']
        model = str(model)
        key = re.search(r": (.*?)Model", model)
        if key is None:
            key = re.search(r": (.*?)Stack", model)
        if key is None:
            key = re.match(r"(.*?)Model", model)
        assert key is not None, "Not able to determine model policy automatically. Please provide policy."
        if key.group(1).lower() in unsupported:
            return False
        return True

    def get_layers(parent, module):
        layer_list = []
        for key, submodule in module._modules.items():
            if isinstance(submodule, nn.Linear):
                layer_list = layer_list + [parent + "." + key]
            elif isinstance(submodule, nn.LayerNorm) or key == 'LayerNorm' or key == 'layer_norm':
                layer_list = layer_list + ["ln"]
            else:
                layer_list = layer_list + AutoTP.get_layers(key, submodule)
        return layer_list

    def update_policy_list(policy_list, new_module, new_gems):
        if len(policy_list):
            for i, policy in enumerate(policy_list):
                # if module already exists in policy, combine gems and remove duplicates
                if policy[0] == type(new_module):
                    new_gems = set(new_gems + policy[1])
                    policy_list[i] = tuple([type(new_module), new_gems])
                    return policy_list
        policy_list.append(tuple([type(new_module), new_gems]))
        return policy_list

    def kernel_supported(module_list):
        policy = []
        for plcy in replace_policies:
            # instantiate a throw-away policy in order to populate the _orig_layer_class
            _ = plcy(None)
            if isinstance(plcy._orig_layer_class, list):
                for orig_layer_class in plcy._orig_layer_class:
                    policy.append(orig_layer_class)
            elif plcy._orig_layer_class is not None:
                policy.append(plcy._orig_layer_class)
        for child in module_list:
            if child.__class__ in policy:
                return True
        return False

    def tp_parser(model):
        policy_list = []
        module_list = []
        layer_list = []
        gem_list = []

        module_list = AutoTP.get_module_list(model)
        assert AutoTP.supported(model), "AutoTP not supported for model. Please use kernel injection since container policy for model exists." \
        if AutoTP.kernel_supported(module_list) else "AutoTP not supported for model. Please provide policy."
        norm_layer_name_list = ['LayerNorm', 'layer_norm', 'ln_1', 'ln_2']
        #ln_1 , ln_2 for Qwen
        for module in module_list:
            for key, submodule in module._modules.items():
                if isinstance(submodule, nn.Linear):
                    layer_list = layer_list + ["." + key]
                elif isinstance(submodule, nn.LayerNorm) or key in norm_layer_name_list:
                    layer_list = layer_list + ["ln"]
                else:
                    layer_list = layer_list + AutoTP.get_layers(key, submodule)
            for i, layer in enumerate(layer_list):
                if layer == 'ln':
                    if layer_list[i - 1] != 'ln':
                        gem_list = gem_list + [layer_list[i - 1]]
                elif 'out_proj' in layer:
                    gem_list = gem_list + [layer]
                elif 'o_proj' in layer:
                    gem_list = gem_list + [layer]
                elif 'down_proj' in layer:
                    gem_list = gem_list + [layer]
                elif 'attention.dense' in layer and 'GPTNeoX' in str(model):
                    gem_list = gem_list + [layer]
                elif 'self_attention.dense' in layer and 'falcon' in str(
                        type(module)):  # this is a hack to get the right linear layer for this model!
                    gem_list = gem_list + [layer]
                # Mixtral-7x8b used w2*act(w1*w3) linear. need to replace w2 to linearallreduce.
                elif 'w2' in layer and 'Mixtral' in str(type(module)):
                    gem_list = gem_list + [layer]
                elif 'self_attn.dense' in layer and 'Phi' in str(type(module)):
                    gem_list = gem_list + [layer]
                elif 'self_attention.dense' in layer and 'ChatGLM' in str(model):
                    gem_list = gem_list + [layer]
                elif 'dense_4h_to_h' in layer and 'ChatGLM' in str(model):
                    gem_list = gem_list + [layer]

            layer_list = []
            if gem_list != []:
                gem_list = list(set(gem_list))
                policy_list = AutoTP.update_policy_list(policy_list, module, gem_list)
                gem_list = []
        assert len(policy_list), "AutoTP not supported for model. Please use kernel injection since container policy for model exists." \
        if AutoTP.kernel_supported(module_list) else "Not able to determine model policy automatically. Please provide policy."
        return policy_list

    def set_tensor_parallel_config(self, mp_size, mp_group):

        if is_autotp_training_mode():
            self.mp_group = groups.get_tensor_model_parallel_group()
            self.mp_size = groups.get_tensor_model_parallel_world_size()
            return

        self.mp_size = mp_size
        self.mp_group = mp_group

    def _replace(self, child, name, conv_linear_layer):
        # This function should clearly define the routing rules for specific layers
        # and avoid any complex shard-related logic.
        if getattr(child, "replaced", False) == True:
            return
        device_name = 'cpu' if self.keep_module_on_host else get_accelerator().current_device_name()
        # keep_module_on_host is used to keep the module on the host. Checkpoints are loaded to the host first (in some
        # cases it can be done from the disk even to prevent filling host's memory), thus no need to create a new copy.
        return_new_copy = not self.keep_module_on_host
        weight_shape = child.weight.shape
        mp_replace = ReplaceWithTensorSlicing(mp_group=self.mp_group)
        # For TP layer skip, e.g., MoE gate, deepseek low rank layer skip
        if "q_a_proj" in name or "kv_a_proj_with_mqa" in name or name == "block_sparse_moe.gate" or (
            ('mlp.shared_expert_gate' == name or 'mlp.gate' == name) and 'qwen2_moe' in str(type(self.module))):
            return child
        # For Yuan model
        if 'Yuan' in str(self.module):
            if 'v_proj' in name:
                return Yuan_LinearLayer(child, self.mp_group)

            elif 'o_proj' in name:
                return Yuan_LinearAllreduce(child, self.mp_group)

        # For MLP including chunk layer.
        if 'gate_up_proj' in name or ('dense_h_to_4h' in name and 'GLM' in str(self.module)):
            return GateUpPack_LinearLayer(child, self.mp_group)
            # For Arctic model, bypass to all_reduce replacement for w2 weights
        arctic_w2_all_reduce_linear = False
        if 'Arctic' in str(self.module) and 'w2' in name:
            arctic_w2_all_reduce_linear = True
        # For MoE MLP model, e.g., deepseek and jamba
        down_proj = False
        if 'down_proj' in name:
            down_proj = True
        if name in self.all_reduce_linears or arctic_w2_all_reduce_linear or down_proj:

            setattr(child, "replaced", True)
            if self.conv_linear_layer:
                return Conv_LinearALlreduce(child, self.mp_group, name=name)
            elif name == "lm_head" or name == 'embed_out':
                return LmHeadLinearAllreduce(child, self.mp_group)

            return LinearAllreduce(child, self.mp_group, name=name)
        else:

            setattr(child, "replaced", True)
            if self.conv_linear_layer:
                conv_LinearLayer(child, self.mp_group)
            elif require_tp_fused_qkvw(name, self.mp_size):
                #Check and handle fused qkv for TP
                return fused_LinearLayer(child, self.mp_group, fused_module=self.module)

            return LinearLayer(child, self.mp_group, name=name)

    def _slice_embedding(self, child, name, conv_linear_layer):
        if getattr(child, "replaced", False) == True:
            return
        mp_replace = ReplaceWithTensorSlicing(mp_group=self.mp_group)

        if hasattr(child.weight, 'ds_tensor'):
            data = child.weight.ds_tensor.data.split(get_shard_size_list(child.weight.shape[1], self.mp_size), dim=1)
        else:
            data = child.weight.data.split(get_shard_size_list(child.weight.shape[1], self.mp_size, name), dim=1)
        data = data[mp_replace.gpu_index].to(get_accelerator().current_device_name())
        data = torch.nn.parameter.Parameter(data, requires_grad=False)

        new_embedding = nn.Embedding(child.weight.shape[0], get_shard_size(child.weight.shape[1], self.mp_size, name))
        new_embedding.weight.data.copy_(data)
        setattr(child, "replaced", True)
        return new_embedding

    def update_mp_params(self, child):
        if getattr(child, "replaced", False) == True:
            return
        param_list = [
            "n_heads", "inner_dim", "num_heads", "num_kv", "num_attention_heads", "num_attn_heads", "all_head_size",
            "embed_dim", "hidden_size", "num_key_value_heads", "num_kv_heads", "kv_n_heads", "d_model",
            "num_attention_heads_per_partition", "num_multi_query_groups_per_partition", "hidden_size_per_partition"
        ]
        for param in param_list:
            if "Yuan" in str(child) and 'embed_dim' in param_list:
                param_list.remove('embed_dim')
            if hasattr(child, param):
                param_val = getattr(child, param)
                setattr(child, param, get_shard_size(param_val, self.mp_size))
        setattr(child, "replaced", True)

    def update_linear_policies(self):
        self.conv_linear_layer = False
        if self.linear_layer_setting is not None:
            self.linear_policies = {self.linear_layer_setting[0]: self._replace}
            if len(self.linear_layer_setting) == 2:
                self.linear_policies.update({self.linear_layer_setting[1]: self._slice_embedding})
        else:
            import transformers
            if self.orig_layer_impl is transformers.models.gpt2.modeling_gpt2.GPT2Block:
                try:
                    self.conv_linear_layer = True
                    self.linear_policies = {transformers.pytorch_utils.Conv1D: self._replace}
                except ImportError:
                    self.linear_policies = {nn.Linear: self._replace}
            else:
                self.linear_policies = {nn.Linear: self._replace, nn.Embedding: self._slice_embedding}

    def _replace_module(self, r_module, prev_name='', prev_class_name=''):
        for name, child in r_module.named_children():
            if prev_class_name == "":
                class_name = prev_name
            elif prev_name == "":
                class_name = prev_class_name
            else:
                class_name = prev_class_name + '.' + prev_name
            checking_key = self.prefix + '.' + class_name + '.' + name + '.' if class_name != "" else self.prefix + '.' + name + '.'
            if Loading.is_load_module(child) and self.state_dict is not None:
                if any(checking_key in item for item in self.state_dict):
                    Loading.load(child, self.state_dict, checking_key, self.mp_group)
                else:
                    continue
            if len(child._buffers) != 0 and self.state_dict is not None:
                Loading.load_buffer(child, self.state_dict, checking_key)
            if child.__class__ in self.linear_policies:
                setattr(r_module, name, self.linear_policies[child.__class__](child, prev_name + '.' + name,
                                                                              self.conv_linear_layer))
            elif any(isinstance(child, lp) for lp in self.linear_policies):
                # Added for falcon model support
                # Note: isinstance will account for class inheritance, child.__class__ does not
                key = None
                for lp in self.linear_policies:
                    if isinstance(child, lp):
                        key = lp
                        break
                assert key is not None
                setattr(r_module, name, self.linear_policies[key](child, prev_name + '.' + name,
                                                                  self.conv_linear_layer))
            else:
                self.update_mp_params(child)
                self._replace_module(child, name, class_name)
        return r_module

    def get_model_num_kv_heads(self, config):
        num_kv_heads = None
        # multi_query_group_num is for chatglm2 & chatglm3
        kv_head_names = [
            'multi_query_group_num', 'num_kv_heads', 'num_key_value_heads', 'num_attention_heads', 'n_heads',
            'attention_heads'
        ]
        for name in kv_head_names:
            if hasattr(config, name):
                num_kv_heads = getattr(config, name)
                if num_kv_heads is not None:
                    break
        return num_kv_heads

    def _replace_last_linear_module(self, r_module):
        if hasattr(r_module, "lm_head"):
            name = "lm_head"
            child = r_module.lm_head
        elif hasattr(r_module, "embed_out"):
            name = "embed_out"
            child = r_module.embed_out
        else:
            return r_module
        if child.__class__ in self.linear_policies:
            setattr(r_module, name, self.linear_policies[child.__class__](child, name, self.conv_linear_layer))
        return r_module
