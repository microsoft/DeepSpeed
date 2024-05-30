#!/bin/bash
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <io_size> <output log dir> <target_gpu>"
    exit 1
fi

function prep_folder()
{
    folder=$1
    if [[ -d ${folder} ]]; then
        rm -f ${folder}/*
    else
        mkdir -p ${folder}
    fi
}

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

function fileExists() {
    local file="$1"
    if [[ -f "$file" ]]; then
        return 0
    else
        return 1
    fi
}

validate_environment

IO_SIZE=$1
LOG_DIR=$2/aio_perf_sweep
MAP_DIR=$2/aio
GPU_MEM=$3
USE_GDS=$4
RUN_SCRIPT=./test_ds_aio.py
READ_OPT="--read"

prep_folder ${MAP_DIR}
prep_folder ${LOG_DIR}

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
if [[ ${USE_GDS} == "gds" ]]; then
    gds_opt="--use_gds"
else
    gds_opt=""
fi

DISABLE_CACHE="sync; bash -c 'echo 1 > /proc/sys/vm/drop_caches' "
SYNC="sync"
sub_opt=""
sub="block"
ov_opt=""
ov="overlap"
t=8
p=8

for d in 64 128; do
    for bs in 8M 16M; do
        SCHED_OPTS="${sub_opt} ${ov_opt} --handle ${gpu_opt} ${gds_opt} --folder_to_device_mapping /workspace/nvme03:0 /workspace/nvme03:1 /workspace/nvme03:2 /workspace/nvme03:3 /workspace/nvme47:4 /workspace/nvme47:5 /workspace/nvme47:6 /workspace/nvme47:7"
        OPTS="--queue_depth ${d} --block_size ${bs} --io_size ${IO_SIZE} --io_parallel ${t}"
        LOG="${LOG_DIR}/read_${sub}_${ov}_t${t}_p${p}_d${d}_bs${bs}.txt"
        cmd="python ${RUN_SCRIPT} ${READ_OPT} ${OPTS} ${SCHED_OPTS} &> ${LOG}"

        echo ${DISABLE_CACHE}
        echo ${cmd}
        echo ${SYNC}
        eval ${DISABLE_CACHE}
        eval ${cmd}
        eval ${SYNC}
        sleep 2
    done
done
