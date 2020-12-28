#!/bin/bash

base_dir=`pwd`

# Where should we save checkpoints and tensorboard events?
JOB_NAME=onebit_adam_seq128
OUTPUT_DIR=${base_dir}/bert_model_outputs

mkdir -p $OUTPUT_DIR

NCCL_TREE_THRESHOLD=0 deepspeed --launcher=mvapich ${base_dir}/deepspeed_train.py \
--cf ${base_dir}/bert_large.json \
--max_seq_length 128 \
--output_dir $OUTPUT_DIR \
--deepspeed_mpi \
--deepspeed \
--print_steps 40 \
--lr_schedule "EE" \
--lr_offset 10e-4 \
--job_name $JOB_NAME \
--deepspeed_config ${base_dir}/deepspeed_bsz4k_onebit_config_seq128.json \
--data_path_prefix /data/bert 
#&> ${JOB_NAME}.log
