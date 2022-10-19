set -x

export TRANSFORMERS_CACHE=/tmp/hf-cache

branch1=$1
branch2=$2

gpus_params="1 2"
dtype_params="int8 fp16 fp32"
graph_params="true false"

gptneo_models="EleutherAI/gpt-neo-2.7B EleutherAI/gpt-neo-1.3B EleutherAI/gpt-neo-125M"
gpt2_models="gpt2 gpt2-large gpt2-xl"
gptj_models="EleutherAI/gpt-j-6B"
opt_models="facebook/opt-125m facebook/opt-1.3b facebook/opt-2.7b facebook/opt-6.7b facebook/opt-13b"
bloom_models="bigscience/bloom-560m bigscience/bloom-1b7 bigscience/bloom-3b bigscience/bloom-7b1"
bert_models="bert-base-cased bert-large-cased dslim/bert-base-NER distilbert-base-cased-distilled-squad"
roberta_models="roberta-base roberta-large Jean-Baptiste/roberta-large-ner-english"

for gpus in `echo "$gpus_params"`; do
    for dtype in `echo "$dtype_params"`; do
        for graph in `echo "$graph_params"`; do
            if [[ "$gpus" != "1" && "$graph" == "true" ]]; then
                # CUDA Graph is not supported for multi-gpu
                continue
            fi
            params="$dtype $graph $gpus"
            for m in `echo "$bert_models"`; do
              bash run_model.sh $m "bert" $branch1 $branch2 $params
            done

            for m in `echo "$roberta_models"`; do
              bash run_model.sh $m "roberta" $branch1 $branch2 $params
            done
        done
        # CUDA Graph is not supported for these models, so skip that for loop
        params="$dtype false $gpus"
        for m in `echo "$gptneo_models"`; do
          bash run_model.sh $m "gpt-neo" $branch1 $branch2 $params
        done

        for m in `echo "$gpt2_models"`; do
          bash run_model.sh $m "gpt2" $branch1 $branch2 $params
        done

        for m in `echo "$gptj_models"`; do
          bash run_model.sh $m "gpt-j" $branch1 $branch2 $params
        done

        for m in `echo "$opt_models"`; do
          bash run_model.sh $m "opt" $branch1 $branch2 $params
        done

        for m in `echo "$bloom_models"`; do
          bash run_model.sh $m "bloom" $branch1 $branch2 $params
        done
    done
done
