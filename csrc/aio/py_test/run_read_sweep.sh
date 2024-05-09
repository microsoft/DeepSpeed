#!/bin/bash
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <io_size> <output log dir> <target_gpu>"
    exit 1
fi


function validate_environment()
{
    validate_cmd="TORCH_EXTENSIONS_DIR=./torch_extentions python ./validate_async_io.py"
    eval ${validate_cmd}
    res=$?
    if [[ $res != 0 ]]; then
        echo "Failing because environment is not properly configured"
        echo "Possible fix: sudo apt-get install libaio-dev"
        exit 1
    fi
}


validate_environment

IO_SIZE=$1
LOG_DIR=$2/aio_perf_sweep
GPU_MEM=$3
RUN_SCRIPT=./test_ds_aio.py
READ_OPT="--read"

if [[ -d ${LOG_DIR} ]]; then
    rm -f ${LOG_DIR}/*
else
    mkdir -p ${LOG_DIR}
fi

if [[ ${GPU_MEM} == "gpu" ]]; then
    gpu_opt="--gpu"
else
    gpu_opt=""
fi

DISABLE_CACHE="sync; bash -c 'echo 1 > /proc/sys/vm/drop_caches' "
SYNC="sync"

for sub in single block; do
    ftd_map="--folder_to_device_mapping \
             /workspace/nvme01/aio:0 "
    if [[ $sub == "single" ]]; then
        sub_opt="--single_submit"
    else
        sub_opt=""
    fi
    for ov in overlap sequential; do
        if [[ $ov == "sequential" ]]; then
            ov_opt="--sequential_requests"
        else
            ov_opt=""
        fi
        for p in 1 ; do
            for t in 1 2 4 8; do
                for d in 8 16 32; do
                    for bs in 256K 512K 1M; do
                        SCHED_OPTS="${sub_opt} ${ov_opt} --handle ${gpu_opt} ${gds_opt} ${ftd_map}"
                        OPTS="--queue_depth ${d} --block_size ${bs} --io_size ${IO_SIZE}"
                        LOG="${LOG_DIR}/read_${sub}_${ov}_t${t}_p${p}_d${d}_bs${bs}.txt"
                        cmd="python ${RUN_SCRIPT} ${READ_OPT} ${OPTS} ${SCHED_OPTS} &> ${LOG}"
                        echo ${DISABLE_CACHE}
                        echo ${cmd}
                        echo ${SYNC}

                        eval ${cmd}
                        eval ${SYNC}
                        sleep 2
                    done
                done
            done
        done
    done
done
