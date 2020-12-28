#!/bin/bash

base_dir=`pwd`

# Where should we save checkpoints and tensorboard events?
JOB_NAME=lamb_nvidia_data_64k_seq128
OUTPUT_DIR=${base_dir}/bert_model_nvidia_data_outputs

mkdir -p $OUTPUT_DIR

NCCL_TREE_THRESHOLD=0 deepspeed ${base_dir}/deepspeed_train.py \
--cf ${base_dir}/bert_large_lamb_nvidia_data.json \
--max_seq_length 128 \
--output_dir $OUTPUT_DIR \
--deepspeed \
--deepspeed_transformer_kernel \
--print_steps 100 \
--lr_schedule "EE" \
--lr_offset 10e-4 \
--job_name $JOB_NAME \
--deepspeed_config ${base_dir}/deepspeed_bsz64k_lamb_config_seq128.json \
--data_path_prefix /workspace/bert \
--use_nvidia_dataset \
&> ${JOB_NAME}.log
