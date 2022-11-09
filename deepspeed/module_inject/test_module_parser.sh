#Runs through all huggingface models and runs module_parser

set -x

model_dir="./transformers/src/transformers/models"

for d in $model_dir/*/modeling_*.py; do
	python module_parser.py -f $d
done
