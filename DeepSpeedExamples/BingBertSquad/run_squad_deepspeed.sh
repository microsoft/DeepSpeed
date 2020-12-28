#~/bin/bash

#1: number of GPUs
#2: Model File Address
#3: BertSquad Data Directory Address
#4: Output Directory Address

NGPU_PER_NODE=$1
MODEL_FILE=$2
SQUAD_DIR=$3
OUTPUT_DIR=$4
LR=${5:-0.00003}
SEED=${6:-12345}
MASTER_PORT=${7:-29500}
DROPOUT=${8:-0.1}
echo "lr is ${LR}"
echo "seed is $SEED"
echo "master port is $MASTER_PORT"
echo "dropout is ${DROPOUT}"

# Force deepspeed to run with only local node
NUM_NODES=1
HOSTFILE=/dev/null

NGPU=$((NGPU_PER_NODE*NUM_NODES))
EFFECTIVE_BATCH_SIZE=24
MAX_GPU_BATCH_SIZE=3
PER_GPU_BATCH_SIZE=$((EFFECTIVE_BATCH_SIZE/NGPU))
if [[ $PER_GPU_BATCH_SIZE -lt $MAX_GPU_BATCH_SIZE ]]; then
       GRAD_ACCUM_STEPS=1
else
       GRAD_ACCUM_STEPS=$((PER_GPU_BATCH_SIZE/MAX_GPU_BATCH_SIZE))
fi
JOB_NAME="deepspeed_${NGPU}GPUs_${EFFECTIVE_BATCH_SIZE}batch_size"
config_json=onebit_deepspeed_bsz24_config.json
run_cmd="deepspeed --num_nodes ${NUM_NODES} --num_gpus ${NGPU_PER_NODE} \
       --master_port=${MASTER_PORT} \
       --hostfile ${HOSTFILE} \
       nvidia_run_squad_deepspeed.py \
       --bert_model bert-large-uncased \
       --do_train \
       --do_lower_case \
       --predict_batch_size 3 \
       --do_predict \
       --train_file $SQUAD_DIR/train-v1.1.json \
       --predict_file $SQUAD_DIR/dev-v1.1.json \
       --train_batch_size $PER_GPU_BATCH_SIZE \
       --learning_rate ${LR} \
       --num_train_epochs 2.0 \
       --max_seq_length 384 \
       --doc_stride 128 \
       --output_dir $OUTPUT_DIR \
       --job_name ${JOB_NAME} \
       --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
       --fp16 \
       --deepspeed \
       --deepspeed_config ${config_json} \
       --dropout ${DROPOUT} \
       --model_file $MODEL_FILE \
       --seed ${SEED} \
       --preln \
       "
echo ${run_cmd}
eval ${run_cmd}
