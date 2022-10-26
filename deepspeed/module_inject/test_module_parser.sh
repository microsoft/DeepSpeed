set -x

model_dir="./transformers/src/transformers/models"
#models=("electra" "roberta" "t5" "wav2vec2" "bert" "albert" "deberta" "deberta_v2" "bart")
#models=("albert" "bart" "beit" "bert" "bert_generation" "big_bird" "bigbird_pegasus" "blenderbot" "blenderbot_small" "camembert" "canine" "clip")
#models_no_modeling=("auto" "bartez" "bartpho" "bert_japanese" "bertweet" "bort" "byt5")

#length=${#models[@]}
#for ((i=0;i<$length;i++)); do
#	python get_injection_policy.py -f ./transformers/src/transformers/models/${models[$i]}/modeling_${models[$i]}.py
#done

for d in $model_dir/*/modeling_*.py; do
	python get_injection_policy.py -f $d
done
