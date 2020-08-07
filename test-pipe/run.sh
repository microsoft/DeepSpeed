#!/bin/bash

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

template_json="$script_dir/ds_config_TEMPLATE.json"
config_json="$script_dir/ds_config.json"

set +x

LOGDIR="$script_dir/logs-debug"
mkdir -p ${LOGDIR}

GPUS=4

SEED=1138

for PP in 4;
do
    for MBSIZE in 2 ;
    do
        #for GAS in 32 64 128 256 512 1024 ;
        for TRAIN_BATCH_SIZE in 64;
        do
            if [ $PP -eq 0 ]
            then
                DP=$GPUS
            else
                DP=$(( ${GPUS} / ${PP} ))
            fi

            GAS=$(( ${TRAIN_BATCH_SIZE} / (${DP} * ${MBSIZE}) ))

            LOGFILE="${LOGDIR}/log-batch_${TRAIN_BATCH_SIZE}-test-pp_${PP}-dp_${DP}-mp_${MP}-gas_${GAS}-mb_${MBSIZE}-ds_${USE_DEEPSPEED}-zero_${USE_ZERO}.txt"

            sed "s/CONFIG_BATCH_SIZE/${TRAIN_BATCH_SIZE}/" ${template_json} \
                | sed "s/CONFIG_MBSIZE/${MBSIZE}/" \
                | sed "s/CONFIG_DP/${DP}/" \
                | sed "s/CONFIG_PP/${PP}/" \
                | sed "s/CONFIG_GAS/${GAS}/" \
                | sed "s/CONFIG_SEED/${SEED}/" \
                > ${config_json}

                #--train-iters $(( 75 * ${GAS} )) \
                #--hidden-size 1600  \
            options=" \
                --seed=${SEED} \
                --pipeline-parallel-size=${PP} \
                --steps 15000 \
                --deepspeed_config=${config_json}
                "

            run_cmd="deepspeed -i worker-0 demo.py $@ ${options}"
            date > ${LOGFILE}
            echo ${run_cmd} >> ${LOGFILE}
            eval ${run_cmd} 2>&1 | tee -a ${LOGFILE}
            date >> ${LOGFILE}
        done
    done
done

echo "ALL DONE"
#deepspeed ~/train.py
