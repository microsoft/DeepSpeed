set -x

model=$1
branch1=$2
branch2=$3
dtype=$4
graph=$5
gpus=$6

version=0
log_path=results/${model}_${dtype}_${graph}_${gpus}gpus_v${version}
mkdir -p ${log_path}

params="--dtype $dtype "

echo "baseline $log_path"
deepspeed --num_gpus 1 hf-model-benchmark.py -m "${model}" $params &> ${log_path}/baseline.log

if [[ "$graph" == "true" ]]; then
    params+="--graph "
fi

cd ../../
git checkout ${branch1}
cd -
echo "ds ${branch1} $log_path"
deepspeed --num_gpus $gpus hf-model-benchmark.py --deepspeed -m "${model}" $params &> ${log_path}/ds-${branch1}.log

cd ../../
git checkout ${branch2}
cd -
echo "ds ${branch2} $log_path"
deepspeed --num_gpus $gpus hf-model-benchmark.py --deepspeed -m "${model}" $params&> ${log_path}/ds-${branch2}.log
