#!/bin/bash

base_dir=`pwd`

# Where should we save checkpoints and tensorboard events?
JOB_NAME=adam_4k_seq128_progressive_layer_drop
OUTPUT_DIR=${base_dir}/bert_model_outputs

mkdir -p $OUTPUT_DIR

config="--progressive_layer_drop"

NCCL_TREE_THRESHOLD=0 deepspeed \
${base_dir}/deepspeed_train.py \
--cf ${base_dir}/bert_base_large_lr.json \
--max_seq_length 128 \
--output_dir $OUTPUT_DIR \
--deepspeed \
--print_steps 100 \
--lr_schedule "LE" \
--job_name $JOB_NAME \
--deepspeed_config ${base_dir}/deepspeed_bsz4k_progressive_layer_drop_config_seq128.json \
--data_path_prefix /data/bert \
${config} \
&> ${JOB_NAME}.log
