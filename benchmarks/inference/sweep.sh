set -x

export TRANSFORMERS_CACHE=/tmp/hf-cache

branch1=$1
branch2=$2

gptneo_models="EleutherAI/gpt-neo-2.7B EleutherAI/gpt-neo-1.3B EleutherAI/gpt-neo-125M"
gpt2_models="gpt2 gpt2-large gpt2-xl"
gptj_models="EleutherAI/gpt-j-6B"
opt_models=""
bloom_models=""

for gpus in `echo "1"`; do
    for dtype in `echo "fp16 fp32"`; do
        for graphs in `echo "true false"`; do
            for kernel in `echo "true false"`; do
                params="$dtype $graphs $kernel $gpus"
                for m in `echo "$gptneo_models"`; do
                  bash run_model.sh $m $branch1 $branch2 $params
                done

                for m in `echo "$gpt2_models"`; do
                  bash run_model.sh $m $branch1 $branch2 $params
                done

                for m in `echo "$gptj_models"`; do
                  bash run_model.sh $m $branch1 $branch2 $params
                done

                for m in `echo "$opt_models"`; do
                  bash run_model.sh $m $branch1 $branch2 $params
                done

                for m in `echo "$bloom_models"`; do
                  bash run_model.sh $m $branch1 $branch2 $params
                done
            done
        done
    done
done
