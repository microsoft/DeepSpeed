export LOCAL_WORLD_SIZE=8
export NUM_NODES=1
export WORLD_SIZE=$((LOCAL_WORLD_SIZE * NUM_NODES))
export NCCL_DEBUG=INFO
MASTER_ADDR=$MASTER_ADDR
MASTER_PORT=$MASTER_PORT

echo $MASTER_ADDR
echo $MASTER_PORT
echo $PWD
#--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" --rdzv_id=1234 --rdzv_backend=c10d \
my_token=" "
HUGGING_FACE_HUB_TOKEN=$my_token torchrun --nnodes=$NUM_NODES --nproc_per_node=$LOCAL_WORLD_SIZE \
      simple_trainer_ulysses.py \
      --bf16 True \
      --max_steps 5 \
      --per_device_train_batch_size 2 \
      --per_device_eval_batch_size 2 \
      --gradient_checkpointing \
      --save_strategy "no" \
      --learning_rate 2e-5 \
      --logging_steps 1 \
      --report_to none \
      --output_dir "$PWD/logs" \
      --deepspeed zero3_hpz.json
