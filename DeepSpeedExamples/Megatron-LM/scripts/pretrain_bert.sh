#!/bin/bash

RANK=0
WORLD_SIZE=1

python pretrain_bert.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --batch-size 4 \
       --seq-length 512 \
       --max-preds-per-seq 80 \
       --max-position-embeddings 512 \
       --train-iters 1000000 \
       --save checkpoints/bert_345m \
       --load checkpoints/bert_345m \
       --resume-dataloader \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type BertWordPieceTokenizer \
       --tokenizer-model-type bert-large-uncased \
       --presplit-sentences \
       --cache-dir cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --fp16 \
       --fp32-layernorm \
       --fp32-embedding
