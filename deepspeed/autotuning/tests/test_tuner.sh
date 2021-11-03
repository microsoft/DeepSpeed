TASK_NAME=mnli
OUTPUT_DIR=./${TASK_NAME}/output
MODEL_NAME=distilbert-base-uncased
# MODEL_NAME=microsoft/deberta-v2-xxlarge
PER_DEVICE_TRAIN_BATCH_SIZE=1
HF_PATH=/home/chengli1/projects


NEPOCHS=1
NGPUS=8
NNODES=1
TEST_SAMPLES=100
MAX_SAMPLES=$(($TEST_SAMPLES * $PER_DEVICE_TRAIN_BATCH_SIZE * $NGPUS))

if [ "$MODEL_NAME" == "distilbert-base-uncased" ]; then
  deepspeed --autotuning tune --num_nodes=$NNODES --num_gpus=$NGPUS $HF_PATH/transformers/examples/pytorch/text-classification/run_glue.py --deepspeed ds_config.json \
    --model_name_or_path ${MODEL_NAME} \
    --task_name ${TASK_NAME} \
    --do_train \
    --do_eval \
    --max_seq_length 256 \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --learning_rate 13e-6 \
    --num_train_epochs 1 --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir \
    --save_steps 0

else
  deepspeed --autotuning tune --num_nodes=$NNODES --num_gpus=$NGPUS $HF_PATH/transformers/examples/pytorch/text-classification/run_glue.py --deepspeed ds_config.json \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --max_seq_length 256 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --learning_rate 13e-6 \
    --num_train_epochs $NEPOCHS \
    --output_dir ${OUTPUT_DIR}_z1 \
    --save_steps 0 \
    --overwrite_output_dir
fi
