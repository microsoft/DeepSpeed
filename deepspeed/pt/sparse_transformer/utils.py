"""
Copyright 2020 The Microsoft DeepSpeed Team
"""

from torch import nn
from deepspeed.pt.sparse_transformer import BertSparseSelfAttention, SparsityConfig
'''
This file contains few utility functions to handle adapting pretrained model with sparse self-attention module.
'''


def extend_position_embedding(model, max_position):
    """This function extends the position embedding weights of a model loaded from a checkpoint.
    It assumes the new max position is bigger than the original max length.

    Arguments:
        model: required: a transformer model
        max_position: required: an integer determining new position embedding size
    Return:
        model: updated model; in which position embedding weights have been extended based on new size
    """

    if hasattr(model, 'bert'):
        original_max_position = model.bert.embeddings.position_embeddings.weight.size(0)
        assert max_position > original_max_position
        extend_multiples = max(1, max_position // original_max_position)
        model.bert.embeddings.position_embeddings.weight.data = model.bert.embeddings.position_embeddings.weight.repeat(
            extend_multiples,
            1)
    elif hasattr(model, 'roberta'):
        # RoBERTa has positions 0 & 1 reserved, so embedding size is max position + 2
        original_max_position, embed_size = model.roberta.embeddings.position_embeddings.weight.shape
        original_max_position -= 2
        extend_multiples = max(1, max_position // original_max_position)
        assert max_position > original_max_position
        max_position += 2
        extended_position_embedding = model.roberta.embeddings.position_embeddings.weight.new_empty(
            max_position,
            embed_size)
        k = 2
        for i in range(extend_multiples):
            extended_position_embedding[k:(
                k + original_max_position
            )] = model.roberta.embeddings.position_embeddings.weight[2:]
            k += original_max_position
        model.roberta.embeddings.position_embeddings.weight.data = extended_position_embedding
    else:
        raise ValueError(
            'Please extend \"extend_position_embedding\" function to support your model type. It currently only supports \"bert\" & \"roberta\"!'
        )

    model.config.max_position_embeddings = max_position
    print(f'Extended position embeddings to {original_max_position * extend_multiples}')

    return model


def update_tokenizer_model_max_length(tokenizer, max_position):
    """This function updates the position embedding length of a tokenizer to a new max position.

    Arguments:
        tokenizer: required: a transformer tokenizer
        max_position: required: an integer determining new position embedding size
    Return:
        tokenizer: updated tokenizer; in which model maximum length has been extended based on new size
    """

    tokenizer.model_max_length = max_position
    tokenizer.init_kwargs['model_max_length'] = max_position
    print(f'updated tokenizer model max imum length to {max_position}')

    return tokenizer


def replace_model_self_attention_with_sparse_self_attention(
    model,
    max_position,
    # SparsityConfig parameters needs to be set accordingly
    sparsity_config=SparsityConfig(num_heads=4,
                                   seq_len=1024)):
    """This function replaces the self attention layers in model encoder with sparse self attention.
    It currently supports bert and roberta model and can be easily extended to any other models following similar steps here.
    For sparsityConfig, refer to the config class.

    Arguments:
        model: required: a transformer model
        max_position: required: an integer determining new position embedding size
        sparsity_config: optional: this parameter determins sparsity pattern configuration; it is based on SparsityConfig class

    Return:
        model: updated model; in which self attention layer has been repleaced with DeepSpeed Sparse Self Attention layer.
    """

    if hasattr(model, 'bert'):
        model.config.max_position_embeddings = max_position
        replace_self_attention_layer_with_sparse_self_attention_layer(
            model.config,
            model.bert.encoder.layer,
            sparsity_config)
    elif hasattr(model, 'roberta'):
        model.config.max_position_embeddings = max_position + 2
        replace_self_attention_layer_with_sparse_self_attention_layer(
            model.config,
            model.roberta.encoder.layer,
            sparsity_config)
    else:
        raise ValueError(
            'Please extend \"update_model_self_attention_to_sparse_self_attention\" function to support \
				 your model type. It currently only supports \"bert\" & \"roberta\"!')
    return model


def replace_self_attention_layer_with_sparse_self_attention_layer(
    config,
    layers,
    # SparsityConfig parameters needs to be set accordingly
    sparsity_config=SparsityConfig(num_heads=4,
                                   seq_len=1024)):
    """This function replaces the self attention layers in attention layer with sparse self attention.
    For sparsityConfig, refer to the config class.

    Arguments:
        config: required: transformer model config
        layers: required: transformer model attention layers
        sparsity_config: optional: this parameter determins sparsity pattern configuration; it is based on SparsityConfig class

    Return:
        layers: updated attention layers; in which self attention layers have been repleaced with DeepSpeed Sparse Self Attention layer.
    """

    for layer in layers:
        deepspeed_sparse_self_attn = BertSparseSelfAttention(config, sparsity_config)
        deepspeed_sparse_self_attn.query = layer.attention.self.query
        deepspeed_sparse_self_attn.key = layer.attention.self.key
        deepspeed_sparse_self_attn.value = layer.attention.self.value

        layer.attention.self = deepspeed_sparse_self_attn

    return layers
