#!/bin/bash
for i in 1 2 4 8
do
	MP_SIZE=$i bash scripts/ds_zero2_pretrain_gpt2XL_model_parallel.sh && python concat.py | gzip > mp${i}_dp$(expr 8 / $i)_trace.json.gz
done

