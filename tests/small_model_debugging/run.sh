#! /bin/bash

GPUS_PER_NODE=1
# Change for multinode config
export MASTER_ADDR=localhost
export MASTER_PORT=6000
NNODES=1
export RANK=0
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

python test_model.py --flops-count true