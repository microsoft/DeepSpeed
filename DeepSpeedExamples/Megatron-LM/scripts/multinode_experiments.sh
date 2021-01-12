#!/bin/bash

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
TRAIN_MICRO_BATCH_SIZE_PER_GPU=${TRAIN_MICRO_BATCH_SIZE_PER_GPU:-16}

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

config_json="$script_dir/multinode_ds_zero2_pretrain_gpt2XL_model_parallel_config.json"
cat > ${config_json} <<-JSON
{
  "train_batch_size": ${TRAIN_BATCH_SIZE},
  "train_micro_batch_size_per_gpu": ${TRAIN_MICRO_BATCH_SIZE_PER_GPU},
  "gradient_accumulation_steps": 1,
  "steps_per_print": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.00015,
      "weight_decay": 1e-2
    }
  },
  "zero_optimization": {
    "stage": 2,
    "cpu_offload": false,
    "contiguous_gradients": false,
    "overlap_comm": false
  },
  "zero_allow_untested_optimizer": true,
  "gradient_clipping": 1.0,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "activation_checkpointing": {
    "partition_activations": true,
    "contiguous_memory_optimization": false
  },
  "wall_clock_breakdown": true
}
JSON

NUM_WORKERS=4
NUM_GPUS_PER_WORKER=8

MP_SIZE=${MP_SIZE:-8}
SEQ_LENGTH=${SEQ_LENGTH:-1024}

hostfile="/job/hostfile"
gpt_options=" \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 48 \
       --hidden-size 1536 \
       --num-attention-heads 24 \
       --seq-length ${SEQ_LENGTH} \
       --max-position-embeddings ${$SEQ_LENGTH} \
       --train-iters 20 \
       --resume-dataloader \
       --train-data webtext \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --no-load-optim \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --fp16 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --deepspeed \
       --deepspeed_config ${config_json}
    "


run_cmd="deepspeed --hostfile=${hostfile} --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} pretrain_gpt2.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
