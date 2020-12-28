#! /bin/bash

# Runs the "345M" parameter model

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export DLWS_NUM_WORKER=1
export DLWS_NUM_GPU_PER_WORKER=4
MP_SIZE=2

gpt_options=" \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 2 \
       --hidden-size 256 \
       --num-attention-heads 16 \
       --batch-size 8 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 1100 \
       --resume-dataloader \
       --train-data webtext \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --save ds_checkpoints
       --load ds_checkpoints
       --save-interval 100
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --checkpoint-activations \
       --fp16 \
       --deepspeed \
       --loss-scale 0 \
       --deepspeed_config ds_config_func_bs8.json
"

run_cmd="deepspeed --num_nodes ${DLWS_NUM_WORKER} --num_gpus ${DLWS_NUM_GPU_PER_WORKER} pretrain_gpt2.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
