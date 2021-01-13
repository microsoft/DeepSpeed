#!/bin/bash

SSH_LIST=(35.208.92.231 35.206.107.250 35.208.115.196)

for ip in "${SSH_LIST[@]}"; do
  rsync -avz --progress multinode_ds_zero2_pretrain_gpt2XL_model_parallel_config.json yunmokoo@"$ip":/home/yunmokoo/workspace/DeepSpeed/DeepSpeedExamples/Megatron-LM/scripts
  rsync -avz --progress multinode_ds_zero2_pretrain_gpt2XL_model_parallel.sh yunmokoo@"$ip":/home/yunmokoo/workspace/DeepSpeed/DeepSpeedExamples/Megatron-LM/scripts
done

