for SEQ_LENGTH in 256 512 1024 2048
do
	for MP_SIZE in 1 2 4 8 16
	do
		SEQ_LENGTH=$SEQ_LENGTH MP_SIZE=$MP_SIZE bash scripts/ds_zero2_pretrain_gpt2XL_model_parallel.sh && \
			./concat.py | gzip > trace_seq${SEQ_LENGTH}_mp${MP_SIZE}.json.gz
	done
done
