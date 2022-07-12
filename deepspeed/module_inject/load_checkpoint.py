import deepspeed
import torch
from torch import nn
from torch.nn import functional as F
import deepspeed.ops.transformer as transformer_inference
from ..runtime.zero import GatheredParameters
from .layers import LinearAllreduce, LinearLayer, Normalize, EmbeddingLayer


def load_model_with_checkpoint(r_module, sd, mp_replace):
    error_msgs = []

    def transpose(data):
        data.reshape(-1).copy_(data.transpose(-1, -2).contiguous().reshape(-1))
        data = data.reshape(data.shape[-1], data.shape[-2])
        return data

    def load(module, prefix):
        args = (sd, prefix, {}, True, [], [], error_msgs)

        if len(list(module.parameters())) > 0 and list(
                module.parameters())[0].numel() == 0:
            with GatheredParameters(list(module.parameters(recurse=False)),
                                    modifier_rank=0):
                module._load_from_sd(*args)
        else:
            if hasattr(module, 'weight'):
                weight_data = sd[prefix + 'weight']
                module.weight = mp_replace.copy(module.weight.data, weight_data)
            if prefix + 'bias' in sd.keys():
                data = sd[prefix + 'bias']
                data = data.to(torch.cuda.current_device())
                module.bias = mp_replace.copy(module.bias.data, data)

    def load_transformer_layer(module, prefix):
        module.norm_w.data.copy_(sd[prefix + 'input_layernorm.' + 'weight'])
        module.norm_b.data.copy_(sd[prefix + 'input_layernorm.' + 'bias'])
        module.attention.attn_qkvw = mp_replace.copy(
            module.attention.attn_qkvw.data,
            transpose(sd[prefix + 'self_attention.query_key_value.' + 'weight']))
        module.attention.attn_qkvb = mp_replace.copy(
            module.attention.attn_qkvb.data,
            sd[prefix + 'self_attention.query_key_value.' + 'bias'])
        module.attention.attn_ow = mp_replace.copy(
            module.attention.attn_ow.data,
            transpose(sd[prefix + 'self_attention.dense.' + 'weight']))
        module.attention.attn_ob = mp_replace.copy(
            module.attention.attn_ob.data,
            sd[prefix + 'self_attention.dense.' + 'bias'])
        module.mlp.attn_nw.data.copy_(sd[prefix + 'post_attention_layernorm.' +
                                         'weight'])
        module.mlp.attn_nb.data.copy_(sd[prefix + 'post_attention_layernorm.' + 'bias'])
        module.mlp.inter_w = mp_replace.copy(
            module.mlp.inter_w.data,
            transpose(sd[prefix + 'mlp.dense_h_to_4h.' + 'weight']))
        module.mlp.inter_b = mp_replace.copy(module.mlp.inter_b.data,
                                             sd[prefix + 'mlp.dense_h_to_4h.' + 'bias'])
        module.mlp.output_w = mp_replace.copy(
            module.mlp.output_w.data,
            transpose(sd[prefix + 'mlp.dense_4h_to_h.' + 'weight']))
        module.mlp.output_b = mp_replace.copy(module.mlp.output_b.data,
                                              sd[prefix + 'mlp.dense_4h_to_h.' + 'bias'])

    layer_policies = {
        nn.Linear: load,
        nn.Embedding: load,
        nn.LayerNorm: load,
        EmbeddingLayer: load,
        LinearLayer: load,
        Normalize: load,
        transformer_inference.DeepSpeedTransformerInference: load_transformer_layer
    }

    all_ds_ids = {}

    def load_module_recursive(module, prefix='', level=0):
        for name, child in module.named_children():
            if child.__class__ in layer_policies:
                checking_key = prefix + name + '.'
                if not any(checking_key in item for item in sd.keys()):
                    if hasattr(child, 'weight') and \
                        hasattr(child.weight, 'ds_id') and \
                        child.weight.ds_id in all_ds_ids:
                        prefix1 = all_ds_ids[child.weight.ds_id]
                        if child.__class__ is nn.Linear:
                            child = LinearLayer(weight=all_ds_ids[child.weight.ds_id])
                            setattr(module, name, child)
                    continue
                if len(list(child.parameters())) > 0 and list(
                        child.parameters())[0].numel() == 0:
                    if child.__class__ is nn.LayerNorm:
                        child = Normalize(dim=child.weight.ds_shape[-1],
                                          dtype=child.weight.dtype,
                                          eps=child.eps)
                        setattr(module, name, child)
                    else:
                        ds_id = None
                        if hasattr(child.weight, 'ds_id'):
                            ds_id = child.weight.ds_id
                        child = EmbeddingLayer(weight_shape=child.weight.ds_shape,
                                               dtype=child.weight.dtype)
                        if ds_id is not None:
                            all_ds_ids[ds_id] = child.weight
                        setattr(module, name, child)

                layer_policies[child.__class__](child, prefix + name + '.')
            else:
                load_module_recursive(child,
                                      prefix if level == 0 else prefix + name + '.',
                                      level + 1)

    load_module_recursive(r_module)
