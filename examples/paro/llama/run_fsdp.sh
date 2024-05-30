#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

output_dir=$1/`basename $0`/$(date "+%Y%m%d-%H%M%S")
echo ${output_dir}
num_of_model_param=7b
factor=4
gpu_bs=2
max_length=128
config_json="./ds_config.json"
gpt_options=" \
    --dataset /path/to/data/dataset/RedPajama/RedPajama-Data-1T-Sample.py \
    --num_of_model_param ${num_of_model_param}
    --weight_decay 0.1 \
    --strategy fsdp \
    --fsdp_sharding_strategy _HYBRID_SHARD_ZERO2 \
    --gradient_checkpointing false \
    --output_dir ${output_dir} \
    --max_length ${max_length} \
    --per_device_train_batch_size ${gpu_bs} \
    --gradient_accumulation_steps ${factor} \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --learning_rate 3e-6 \
    --warmup_ratio 0.06 \
    --ds_config ${config_json}
"
gpt_options="${gpt_options}
               \
"
DATESTR=$(date +"%m%d%H%M")

python train.py ${gpt_options} >> zero_logs/${num_of_model_param}_seq${max_length}_bs${gpu_bs}_factor${factor}_${DATESTR}_rank_${RANK}.txt 2>&1 &