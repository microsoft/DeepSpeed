# coding=utf-8
# This script references to below file from HuggingFace:
#   https://github.com/huggingface/transformers/blob/d541938/src/transformers/modeling_bert.py
#
# It converts Tensorflow and Huggingface checkpoint files to DeepSpeed.

import os
import argparse
import logging
import torch
import re
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_data(param, array):
    try:
        assert param.shape == array.shape
    except AssertionError as e:
        e.args += (param.shape, array.shape)
        raise
    param.data = torch.from_numpy(array)

def load_tf_weights_in_bert_kernel(model, ckpt_path, voc_size_diff):
    """ Load tf checkpoints in DeepSpeed model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in DeepSpeed, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(ckpt_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    qkv = {}
    for name_str, array in zip(names, arrays):
        name = name_str.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        key = None
        skipping = False
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]

            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            # Special in deepspeed.
            elif name_str.find("bert/pooler/dense") >= 0 and scope_names[0] == "dense":
                pointer = getattr(pointer, "dense_act")
            elif name_str.find("bert/embeddings/LayerNorm/gamma") >= 0 and scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif name_str.find("bert/embeddings/LayerNorm/beta") >= 0 and scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    skipping = True
                    break

            if len(scope_names) >= 2:
                num = int(scope_names[1])

                pointer = pointer[num]

                # For transofrmer kernel layers.
                if scope_names[0] == 'layer':
                    if name_str.find("attention/self/query/kernel") > 0:
                        key = "qw"
                    elif name_str.find("attention/self/query/bias") > 0:
                        key = "qb"
                    elif name_str.find("attention/self/key/kernel") > 0:
                        key = "kw"
                    elif name_str.find("attention/self/key/bias") > 0:
                        key = "kb"
                    elif name_str.find("attention/self/value/kernel") > 0:
                        key = "vw"
                    elif name_str.find("attention/self/value/bias") > 0:
                        key = "vb"
                    elif name_str.find("attention/output/dense/kernel") > 0:
                        pointer = getattr(pointer, "attn_ow")
                    elif name_str.find("attention/output/dense/bias") > 0:
                        pointer = getattr(pointer, "attn_ob")
                    elif name_str.find("attention/output/LayerNorm/gamma") > 0:
                        pointer = getattr(pointer, "attn_nw")
                    elif name_str.find("attention/output/LayerNorm/beta") > 0:
                        pointer = getattr(pointer, "attn_nb")
                    elif name_str.find("intermediate/dense/kernel") > 0:
                        pointer = getattr(pointer, "inter_w")
                    elif name_str.find("intermediate/dense/bias") > 0:
                        pointer = getattr(pointer, "inter_b")
                    elif name_str.find("output/dense/kernel") > 0 and name_str.find("attention") < 0:
                        pointer = getattr(pointer, "output_w")
                    elif name_str.find("output/dense/bias") > 0 and name_str.find("attention") < 0:
                        pointer = getattr(pointer, "output_b")
                    elif name_str.find("output/LayerNorm/gamma") > 0 and name_str.find("attention") < 0:
                        pointer = getattr(pointer, "norm_w")
                    elif name_str.find("output/LayerNorm/beta") > 0 and name_str.find("attention") < 0:
                        pointer = getattr(pointer, "norm_b")
                    else:
                        raise ValueError(f"unexpect scope name {name_str} in transformer layer.")
                    break

        if skipping:
            continue

        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif "kernel" in name:
            array = np.transpose(array)

        if key is not None:
            qkv[key] = array

        if all(k in qkv for k in ("qw", "kw", "vw")):
            array = np.concatenate((qkv["qw"], qkv["kw"], qkv["vw"]), axis=0)
            pointer = getattr(pointer, "attn_qkvw")
            qkv.pop("qw")
            qkv.pop("kw")
            qkv.pop("vw")
        elif all(k in qkv for k in ("qb", "kb", "vb")):
            array = np.concatenate((qkv["qb"], qkv["kb"], qkv["vb"]), axis=0)
            pointer = getattr(pointer, "attn_qkvb")
            qkv.pop("qb")
            qkv.pop("kb")
            qkv.pop("vb")
        elif key is not None:
            # For Q/K/V weight/bias in TF, do nothing if not all ready to merge.
            continue

        # DeepSpeed BERT model has voc_size 8 aligned.
        if voc_size_diff > 0 and name_str.find("embeddings/word_embeddings") >= 0:
            z = np.zeros((voc_size_diff, array.shape[1]), dtype=array.dtype)
            array = np.concatenate((array, z), axis=0)

        set_data(pointer, array)
        logger.info("Initialize DeepSpeed weight {}".format(name))

    return model

def load_hf_weights_in_bert_kernel(model, ckpt_path, voc_size_diff):
    """ Load huggingface checkpoints and convert to a deepspeed model.
    """
    hf_path = os.path.abspath(ckpt_path)
    logger.info("Converting Huggingface checkpoint from {}".format(hf_path))
    # Load weights from Huggingface model
    ckpt = torch.load(hf_path, map_location=torch.device("cpu"))

    qkv = {}
    for name_str in ckpt.keys():
        array = ckpt[name_str].numpy()
        logger.info("Loading Huggingface weight {} with shape {}".format(name_str, array.shape))
        name = name_str.split(".")
        pointer = model
        key = None
        is_layer = False
        skipping = False
        for m_name in name:
            # Special in deepspeed.
            if name_str.find("bert.pooler.dense") >= 0 and m_name == "dense":
                pointer = getattr(pointer, "dense_act")
            elif is_layer:
                pass
            else:
                try:
                    pointer = getattr(pointer, m_name)
                except AttributeError:
                    logger.info("Skipping {}".format(".".join(name)))
                    skipping = True
                    break

            if m_name == "layer":
                is_layer = True
                continue

            if m_name.isnumeric() and is_layer:
                num = int(m_name)
                pointer = pointer[num]
                is_layer = False

                # For transofrmer kernel layers.
                if name_str.find("attention.self.query.weight") > 0:
                    key = "qw"
                elif name_str.find("attention.self.query.bias") > 0:
                    key = "qb"
                elif name_str.find("attention.self.key.weight") > 0:
                    key = "kw"
                elif name_str.find("attention.self.key.bias") > 0:
                    key = "kb"
                elif name_str.find("attention.self.value.weight") > 0:
                    key = "vw"
                elif name_str.find("attention.self.value.bias") > 0:
                    key = "vb"
                elif name_str.find("attention.output.dense.weight") > 0:
                    pointer = getattr(pointer, "attn_ow")
                elif name_str.find("attention.output.dense.bias") > 0:
                    pointer = getattr(pointer, "attn_ob")
                elif name_str.find("attention.output.LayerNorm.weight") > 0:
                    pointer = getattr(pointer, "attn_nw")
                elif name_str.find("attention.output.LayerNorm.bias") > 0:
                    pointer = getattr(pointer, "attn_nb")
                elif name_str.find("intermediate.dense.weight") > 0:
                    pointer = getattr(pointer, "inter_w")
                elif name_str.find("intermediate.dense.bias") > 0:
                    pointer = getattr(pointer, "inter_b")
                elif name_str.find("output.dense.weight") > 0 and name_str.find("attention") < 0:
                    pointer = getattr(pointer, "output_w")
                elif name_str.find("output.dense.bias") > 0 and name_str.find("attention") < 0:
                    pointer = getattr(pointer, "output_b")
                elif name_str.find("output.LayerNorm.weight") > 0 and name_str.find("attention") < 0:
                    pointer = getattr(pointer, "norm_w")
                elif name_str.find("output.LayerNorm.bias") > 0 and name_str.find("attention") < 0:
                    pointer = getattr(pointer, "norm_b")
                else:
                    raise ValueError(f"unexpect scope name {name_str} in transformer layer.")
                break

        if skipping:
            continue

        if key is not None:
            qkv[key] = array

        if all(k in qkv for k in ("qw", "kw", "vw")):
            array = np.concatenate((qkv["qw"], qkv["kw"], qkv["vw"]), axis=0)
            pointer = getattr(pointer, "attn_qkvw")
            qkv.pop("qw")
            qkv.pop("kw")
            qkv.pop("vw")
        elif all(k in qkv for k in ("qb", "kb", "vb")):
            array = np.concatenate((qkv["qb"], qkv["kb"], qkv["vb"]), axis=0)
            pointer = getattr(pointer, "attn_qkvb")
            qkv.pop("qb")
            qkv.pop("kb")
            qkv.pop("vb")
        elif key is not None:
            # For Q/K/V weight/bias in HF, do nothing if not all ready to merge.
            continue

        # DeepSpeed BERT model has voc_size 8 aligned.
        if voc_size_diff > 0 and name_str.find("embeddings.word_embeddings") >= 0:
            z = np.zeros((voc_size_diff, array.shape[1]), dtype=array.dtype)
            array = np.concatenate((array, z), axis=0)

        set_data(pointer, array)
        logger.info("Initialize DeepSpeed weight {}".format(name))

    return model

def load_hf_weights_in_bert_torch(model, ckpt_path, voc_size_diff):
    """ Load huggingface checkpoints and convert to a deepspeed model.
    """
    hf_path = os.path.abspath(ckpt_path)
    logger.info("Converting Huggingface checkpoint from {}".format(hf_path))
    # Load weights from Huggingface model
    ckpt = torch.load(hf_path, map_location=torch.device("cpu"))

    qkv = {}
    for name_str in ckpt.keys():
        array = ckpt[name_str].numpy()
        logger.info("Loading Huggingface weight {} with shape {}".format(name_str, array.shape))
        name = name_str.split(".")
        pointer = model
        key = None
        is_layer = False
        skipping = False
        for m_name in name:
            # Special in deepspeed.
            if name_str.find("intermediate.dense") >= 0 and m_name == "dense":
                pointer = getattr(pointer, "dense_act")
            elif name_str.find("pooler.dense") >= 0 and m_name == "dense":
                pointer = getattr(pointer, "dense_act")
            else:
                try:
                    pointer = getattr(pointer, m_name)
                except AttributeError:
                    logger.info("Skipping {}".format(".".join(name)))
                    skipping = True
                    break

        if skipping:
            continue

        # DeepSpeed BERT model has voc_size 8 aligned.
        if voc_size_diff > 0 and name_str.find("embeddings.word_embeddings") >= 0:
            z = np.zeros((voc_size_diff, array.shape[1]), dtype=array.dtype)
            array = np.concatenate((array, z), axis=0)

        set_data(pointer, array)
        logger.info("Initialize DeepSpeed weight {}".format(name))

    return model

def convert_ckpt_to_deepspeed(model, ckpt_type, ckpt_path, vocab_diff, kernel_enabled):

    # Load weights from checkpoint
    if ckpt_type == "HF":
        if kernel_enabled:
            load_hf_weights_in_bert_kernel(model, ckpt_path, vocab_diff)
        else:
            load_hf_weights_in_bert_torch(model, ckpt_path, vocab_diff)
    elif ckpt_type == "TF":
        if kernel_enabled:
            load_tf_weights_in_bert_kernel(model, ckpt_path, vocab_diff)
        else:
            raise ValueError("--deepspeed_transformer_kernel is required for loading TF checkpoint.")
    else:
        raise ValueError(f"Invalid ckpt_type.")
