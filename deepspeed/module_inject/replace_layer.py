import torch
from torch import nn

import deepspeed
from .layers import LinearAllreduce, LinearLayer

from deepspeed import comm as dist
from .replace_module import ReplaceWithTensorSlicing, GroupQuantizer, _replace_module, get_transformer_name
# Do not import replace_module from .replace_module as it uses the old policies from replace_policy.py

from .load_checkpoint import load_model_with_checkpoint

from .utils import policy_to_ds_container

# import new policy files
from .policies import HFGPT2LayerPolicy
from .policies import HFBertLayerPolicy
from .policies import BLOOMLayerPolicy
from .policies import HFGPTJLayerPolicy
from .policies import HFGPTNEOLayerPolicy
from .policies import GPTNEOXLayerPolicy
from .policies import HFOPTLayerPolicy
from .policies import MegatronLayerPolicy
from .policies import HFDistilBertLayerPolicy

import time
import tqdm
import os

# Local list of replacement policies to use instead of the original list imported from replace_policy.py
# TODO (lekurile): Is this the correct place for this to live?
replace_policies = [
    HFGPT2LayerPolicy,
    HFBertLayerPolicy,
    BLOOMLayerPolicy,
    HFGPTJLayerPolicy,
    HFGPTNEOLayerPolicy,
    GPTNEOXLayerPolicy,
    HFOPTLayerPolicy,
    MegatronLayerPolicy,
    HFDistilBertLayerPolicy,
]

# TODO (lekurile): Need to test the generic_policies
# non-transformer-based policies
#generic_policies = [UNetPolicy, VAEPolicy]


# This function is called by replace_transormer_layer() in replace_layer.py and used _replace_module() helper from the replace_module.py file
def replace_module(model, orig_class, replace_fn, _replace_policy):
    """ Scan the model for instances of ``orig_clas:`` to replace using ``replace_fn``.
    Arguments:
        model (torch.nn.Module): the model to augment
        orig_class (torch.nn.Module): the module to search for
        replace_fn (method): a method to convert instances of ``orig_class`` to the
                             desired type and return a new instance.
    Returns:
        A modified ``model``.
    """
    # policy is a mapping dictionary of the form: {user supplied module_name: replacement_fn, replacement_policy}
    policy = {}
    if orig_class is not None:
        print('>> replace_module: orig_class is not None')
        policy.update({orig_class: (replace_fn, _replace_policy)})
    else:
        print(f'>> replace_module: orig_class is {orig_class}')
        for plcy in replace_policies:
            # instantiate a throw-away policy in order to populate the _orig_layer_class
            _ = plcy(None)
            if isinstance(plcy._orig_layer_class, list):
                for orig_layer_class in plcy._orig_layer_class:
                    policy.update({orig_layer_class: (replace_fn, plcy)})
            elif plcy._orig_layer_class is not None:
                policy.update({plcy._orig_layer_class: (replace_fn, plcy)})
    assert len(policy.items()) > 0,\
        "No default policy found! Please specify your policy injection_policy (like {BertLayer:HFBEertLayerPolicy})." +\
        "You can find some samples here: https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/module_inject/replace_policy.py"

    print(
        f"> ---- calling _replace_module in replace_layer.py with model = {model.__class__}"
    )
    for k, v in policy.items():
        print(f"> ---- policy: {k} -> {v}")
    #exit(0)

    # call the recursive function with model and the policy-to-replacement mapping
    replaced_module, _ = _replace_module(model, policy)
    return replaced_module


# TODO (lekurile): Do these need to be defined here, if they're defined as globals in replace_with_policy()?
selected_policy_g = None
megatron_v2_g = False
transformer_config_g = None


def replace_transformer_layer(orig_layer_impl,
                              model,
                              checkpoint_dict,
                              config,
                              model_config):
    """ Replace bert-style transformer layers with DeepSpeed's transformer layer
    Arguments:
        orig_layer_impl (torch.nn.Module): the original transformer layer implementation to look for,
            e.g., transformers.modeling_bert.BertLayer.
        model (torch.nn.Module): user's nn.module representing their model
        checkpoint_dict: Dictionary for checkpoint passed from the Inference Engine
        config: top-level DS Inference config defined in inference/config.py
        model_config: HuggingFace model config passed from the inference/engine.py
    Returns:
        Updated nn.module with replaced transformer layers
    """
    # TODO (lekurile): Get clarity on intention of these definitions
    # defining globals as internally defined functions inherit these everywhere
    fp16 = (config.dtype == torch.float16 or config.dtype == torch.int8)
    quantize = (config.dtype == torch.int8)
    # todo: Refactor later. In future, let's minimize the style used above and use config.** instead

    linear_layer_setting = None
    '''
        linear_layer_setting (tuple of modules) [Optional]: shows which two classes are used for linear layers and embedding layers
    '''
    micro_batch_size = -1
    seed = -1
    local_rank = -1

    mp_replace = ReplaceWithTensorSlicing(
        mp_group=config.tensor_parallel.tp_group,
        mp_size=config.tensor_parallel.tp_size)  #, out_dim=0, in_dim=1)

    def replace_with_policy(child,
                            policy_cls,
                            triangular_masking,
                            inference=False,
                            layer_id=0):
        print(f">-- replace_with_policy(): {policy_cls.__name__}")

        policy = policy_cls(child, inference=inference)
        if not policy.cuda_graph_supported:
            # policy says cuda graph is not supported raise an error if set
            assert not config.enable_cuda_graph, "cuda graph is not supported with this model, please disable"

        from deepspeed.moe.layer import MoE
        moe = False
        if hasattr(child, 'mlp') and isinstance(child.mlp, MoE):
            num_experts = child.mlp.num_experts
            moe = True

        print(f">-- replace_with_policy(): {policy}")

        # 1. Create a model-specific container object using the policy object.
        _container = policy_to_ds_container(policy=policy,
                                            config=config,
                                            model_config=model_config,
                                            layer_id=layer_id,
                                            child=child)
        _container.set_dtype(fp16)
        _container.set_moe(moe)

        # 2. Set the tensor parallelism config
        _container.set_tensor_parallel_config(config.tensor_parallel.tp_size,
                                              config.tensor_parallel.tp_group)

        # 3. Initialize tensors
        _container.initialize_tensors()

        # 4. deal with data types -- needs refactor to use dtype instead of fp16
        if fp16:
            _container.convert_to_required_dtype(dtype=torch.half)

        # 5. Set the quantization config
        # TODO (lekurile): Move quantizer creation into set_quantization_config?
        quantizer = GroupQuantizer(q_int8=quantize)
        _container.set_quantization_config(quantize, quantizer)

        # 6. create a DS Inference config object
        _container.create_config()
        #from rich.pretty import pprint
        #pprint(_container.config.__dict__)
        #exit(0)

        # 7. use the config and create the module
        _container.create_module()

        # 8. transpose the weights and bias if needed
        _container.transpose()

        # 9. deal with tensor parallelism. todo: when bloom and other models are ready, we can move this into step 10.
        _container.apply_tensor_parallelism(mp_replace)

        # 10. copy the tensors from the model-specific container to the new module
        _container.copy_data_to_new_module()

        # 11. set globals for generic checkpoint loading
        global selected_policy_g
        global megatron_v2_g
        global transformer_config_g

        if selected_policy_g is None:
            selected_policy_g = _container.policy

        megatron_v2_g = _container.megatron_v2
        transformer_config_g = _container.config

        return _container.module

    def replace_wo_policy(module, all_reduce_linears):
        mp_size = config.tensor_parallel.tp_size
        mp_group = config.tensor_parallel.tp_group

        def _replace(child, name, conv_linear_layer):
            mp_replace = ReplaceWithTensorSlicing(mp_group=mp_group)
            z_inference = (len(list(child.parameters())) > 0) and (list(
                child.parameters())[0].numel() == 0)
            if z_inference:
                weight_shape = child.weight.ds_shape
            else:
                weight_shape = child.weight.shape
            if name in all_reduce_linears:
                new_weight = torch.empty((
                    weight_shape[1] if conv_linear_layer else weight_shape[0],
                    (weight_shape[0] if conv_linear_layer else weight_shape[1]) //
                    mp_size,
                ),
                                         device=child.weight.device,
                                         dtype=child.weight.dtype)
                if z_inference:
                    with deepspeed.zero.GatheredParameters(child.weight,
                                                           modifier_rank=0):
                        data = child.weight.data.to(new_weight.device)
                        if conv_linear_layer:
                            data = data.transpose(-1, -2).contiguous()
                        data = mp_replace.copy(new_weight, data)
                    child.weight.ds_tensor = torch.empty(1)
                else:
                    if conv_linear_layer:
                        child.weight.data = child.weight.data.transpose(-1,
                                                                        -2).contiguous()
                    data = mp_replace.copy(new_weight, child.weight.data)
                new_bias = torch.empty((weight_shape[0]),
                                       device=child.weight.device,
                                       dtype=child.weight.dtype)
                if z_inference:
                    with deepspeed.zero.GatheredParameters(child.bias, modifier_rank=0):
                        new_bias.data.copy_(child.bias.data)
                elif child.bias is not None:
                    new_bias.data.copy_(child.bias.data)
                return LinearAllreduce(data, child.bias if child.bias is None else \
                            torch.nn.parameter.Parameter(new_bias.to(torch.cuda.current_device())), mp_group)
            else:
                new_weight = torch.empty((
                    (weight_shape[1] if conv_linear_layer else weight_shape[0]) //
                    mp_size,
                    weight_shape[0] // mp_size if conv_linear_layer else weight_shape[1],
                ),
                                         device=child.weight.device,
                                         dtype=child.weight.dtype)
                if z_inference:
                    with deepspeed.zero.GatheredParameters(child.weight,
                                                           modifier_rank=0):
                        data = child.weight.data.to(new_weight.device)
                        if conv_linear_layer:
                            data = data.transpose(-1, -2).contiguous()
                        data = mp_replace.copy(new_weight, data)
                    child.weight.ds_tensor = torch.empty(1)
                else:
                    if conv_linear_layer:
                        child.weight.data = child.weight.data.transpose(-1,
                                                                        -2).contiguous()
                    data = mp_replace.copy(new_weight, child.weight.data)

                new_bias = torch.empty((weight_shape[0] // mp_size),
                                       device=child.weight.device,
                                       dtype=child.weight.dtype)
                if z_inference:
                    with deepspeed.zero.GatheredParameters(child.bias, modifier_rank=0):
                        bias_data = None if child.bias is None else mp_replace.copy(
                            new_bias,
                            child.bias.data).to(torch.cuda.current_device())
                else:
                    bias_data = None if child.bias is None else mp_replace.copy(
                        new_bias,
                        child.bias.data).to(torch.cuda.current_device())
                return LinearLayer(weight=data.to(torch.cuda.current_device()),
                                   bias=bias_data)

        def _slice_embedding(child, name, conv_linear_layer):
            mp_replace = ReplaceWithTensorSlicing(mp_group=mp_group)
            new_weight = torch.empty((child.weight.shape[0],
                                      child.weight.shape[1] // mp_size),
                                     device=child.weight.device,
                                     dtype=child.weight.dtype)
            data = mp_replace.copy(new_weight,
                                   child.weight.ds_tensor.data if hasattr(child.weight, 'ds_tensor') else \
                                   child.weight.data)
            new_embedding = nn.Embedding(child.weight.shape[0],
                                         child.weight.shape[1] // mp_size)
            new_embedding.weight.data.copy_(data)
            return new_embedding

        def update_mp_params(child):
            if hasattr(child, 'n_heads'):
                child.n_heads = child.n_heads // mp_size
            if hasattr(child, 'inner_dim'):
                child.inner_dim = child.inner_dim // mp_size
            if hasattr(child, 'num_heads'):
                child.num_heads = child.num_heads // mp_size
            if hasattr(child, 'num_attention_heads'):
                child.num_attention_heads = child.num_attention_heads // mp_size
            if hasattr(child, 'all_head_size'):
                child.all_head_size = child.all_head_size // mp_size
            if hasattr(child, 'embed_dim'):
                child.embed_dim = child.embed_dim // mp_size
            if hasattr(child, 'hidden_size'):
                child.hidden_size = child.hidden_size // mp_size

        conv_linear_layer = False
        if linear_layer_setting is not None:
            linear_policies = {linear_layer_setting[0]: _replace}
            if len(linear_layer_setting) == 2:
                linear_policies.update({linear_layer_setting[1]: _slice_embedding})
        else:
            if orig_layer_impl is HFGPT2LayerPolicy._orig_layer_class:
                try:
                    import transformers
                    conv_linear_layer = True
                    linear_policies = {transformers.model_utils.Conv1D: _replace}
                except ImportError:
                    linear_policies = {nn.Linear: _replace}
            else:
                linear_policies = {nn.Linear: _replace, nn.Embedding: _slice_embedding}

        def _replace_module(r_module, prev_name=''):
            for name, child in r_module.named_children():
                if child.__class__ in linear_policies:
                    setattr(
                        r_module,
                        name,
                        linear_policies[child.__class__](child,
                                                         prev_name + '.' + name,
                                                         conv_linear_layer))
                else:
                    update_mp_params(child)
                    _replace_module(child, name)
            return r_module

        return _replace_module(module)

    def replace_fn(child, _policy, layer_id=0):
        training = False  # todo: refactor this part to go in the config
        if training:
            # copy relevant state from child -> new module
            new_module = replace_with_policy(child, _policy, config.triangular_masking)

        else:
            # copy relevant state from child -> new module
            if config.replace_with_kernel_inject:
                new_module = replace_with_policy(child,
                                                 _policy,
                                                 config.triangular_masking,
                                                 inference=True,
                                                 layer_id=layer_id)
            else:
                print("  >--- Step 2a (replace_module.py) replace_wo_policy")
                new_module = replace_wo_policy(child, _policy)

        return new_module

    print(">-- calling replace_module with _replace_policy = {policy}")
    replaced_module = replace_module(model=model,
                                     orig_class=orig_layer_impl,
                                     replace_fn=replace_fn,
                                     _replace_policy=config.injection_policy_tuple)

    #print(f">-- replace_module done, replaced_module = {replaced_module}")
    #exit(0)
    quantizer = GroupQuantizer(q_int8=quantize)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if checkpoint_dict is not None:
        start_time = time.time()
        checkpoint = checkpoint_dict['checkpoints']
        ckpt_list = checkpoint["tp"] if type(checkpoint) is dict else checkpoint
        ckpt_type = checkpoint_dict.get('parallelization', 'pp')
        ckpt_mp_size = checkpoint_dict.get('tp_size', len(ckpt_list))
        ckpt_mp_size = checkpoint_dict.get('mp_size', ckpt_mp_size)
        base_dir1 = checkpoint_dict.get('base_dir', config.base_dir)

        if ckpt_type == 'pp' and type(checkpoint) is list:
            pbar = tqdm.tqdm(total=len(checkpoint),
                             desc=f"Loading {len(checkpoint)} checkpoint shards")

            for i in range(len(checkpoint)):
                sd = [
                    torch.load(os.path.join(base_dir1,
                                            checkpoint[i]),
                               map_location='cpu')
                ]
                load_model_with_checkpoint(
                    replaced_module,
                    sd,
                    mp_replace,
                    ckpt_type,
                    quantizer,
                    param_names=selected_policy_g.get_param_names(),
                    transformer_config=transformer_config_g,
                    megatron_v2=megatron_v2_g)
                pbar.update(1)
        else:
            import gc
            num_checkpoints = len(ckpt_list) // ckpt_mp_size
            tp_split_size = (world_size / ckpt_mp_size)
            sd_offset = int(rank / tp_split_size)
            sd_count = int((rank + max(1, tp_split_size)) / tp_split_size) - sd_offset
            pbar = tqdm.tqdm(total=num_checkpoints,
                             desc=f"Loading {num_checkpoints} checkpoint shards")
            for i in range(num_checkpoints):
                pbar.update(1)
                ckpt_index = i * ckpt_mp_size + sd_offset
                ckpt_files = [
                    os.path.join(base_dir1,
                                 ckpt_list[ckpt_index +
                                           j]) if base_dir1 else ckpt_list[ckpt_index +
                                                                           j]
                    for j in range(sd_count)
                ]
                sds = [
                    torch.load(ckpt_file,
                               map_location='cpu') for ckpt_file in ckpt_files
                ]
                load_model_with_checkpoint(
                    replaced_module,
                    sds,
                    mp_replace,
                    ckpt_type,
                    quantizer,
                    int(rank % tp_split_size),
                    param_names=selected_policy_g.get_param_names(),
                    transformer_config=transformer_config_g,
                    megatron_v2=megatron_v2_g)
                sds = [None for _ in sds]
                gc.collect()

            if "non_tp" in checkpoint:
                pbar = tqdm.tqdm(
                    total=len(checkpoint["non_tp"]),
                    desc=f"Loading {len(checkpoint['non_tp'])} checkpoint shards")

                for i in range(len(checkpoint["non_tp"])):
                    pbar.update(1)
                    ckpt_file = os.path.join(base_dir1,
                                             checkpoint["non_tp"][i]
                                             ) if base_dir1 else checkpoint["non_tp"][i]
                    sds = [torch.load(ckpt_file, map_location='cpu')]
                    load_model_with_checkpoint(
                        replaced_module,
                        sds,
                        mp_replace,
                        ckpt_type,
                        quantizer,
                        int(rank % tp_split_size),
                        param_names=selected_policy_g.get_param_names(),
                        transformer_config=transformer_config_g,
                        megatron_v2=megatron_v2_g)
                    sds = [None for _ in sds]
                    gc.collect()
        print(f"checkpoint loading time at rank {rank}: {time.time()-start_time} sec")

    if config.save_mp_checkpoint_path is not None:
        from collections import OrderedDict
        import json
        num_partitions = 8

        if checkpoint_dict is None:
            ckpt_name = "ds_model"
            try:
                from transformers.models.bloom.modeling_bloom import BloomForCausalLM
                if isinstance(model, BloomForCausalLM):
                    ckpt_name = "bloom"
            except ImportError:
                ckpt_name = "ds_model"
        else:
            ckpt_name = checkpoint_dict['type']
        if dist.is_initialized():
            dist.barrier()
        transformer_name = get_transformer_name(replaced_module)
        non_tp_ckpt_name = f'non-tp.pt'
        ckpt_files = [non_tp_ckpt_name]
        os.makedirs(config.save_mp_checkpoint_path, exist_ok=True)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print("Saving tp-sharded checkpoints")
            torch.save(
                OrderedDict({
                    k: v
                    for k,
                    v in dict(replaced_module.state_dict()).items()
                    if transformer_name not in k
                }),
                f'{config.save_mp_checkpoint_path}/{non_tp_ckpt_name}')
            ckpt_config = json.dumps({
                'type':
                ckpt_name,
                'base_dir':
                f'{config.save_mp_checkpoint_path}',
                'checkpoints': {
                    "non_tp":
                    ckpt_files,
                    "tp": [
                        f'tp_{r:0>2d}_{m:0>2d}.pt' for m in range(num_partitions)
                        for r in range(world_size)
                    ]
                },
                'version':
                1.0,
                'parallelization':
                'tp',
                'tp_size':
                world_size,
                'dtype':
                'int8' if quantize else ('float16' if fp16 else 'float32')
            })
            with open(f"{config.save_mp_checkpoint_path}/ds_inference_config.json",
                      "w") as cfg:
                cfg.write(ckpt_config)

        rep_sd = replaced_module.state_dict()
        for n, p in replaced_module.named_parameters():
            if hasattr(p, 'scale'):
                rep_sd[n] = [p, p.scale]
        keys = list(rep_sd.keys())
        partition_size = (len(keys) // num_partitions + 1)
        for m in range(num_partitions):
            torch.save(
                OrderedDict({
                    k: [rep_sd[k],
                        rep_sd[k].scale] if hasattr(rep_sd[k],
                                                    'scale') else rep_sd[k]
                    for k in keys[m * partition_size:(m + 1) * partition_size]
                    if transformer_name in k
                }),
                f'{config.save_mp_checkpoint_path}/tp_{rank:0>2d}_{m:0>2d}.pt')

    return replaced_module
