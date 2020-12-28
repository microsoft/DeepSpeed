#!/bin/bash

echo "processing gpt2 data ..."
DIR="/raid/mpatwary/redownload_v0/0-21"

for thread in {0..3}; do
    echo " launching thread "$thread && python make_gpt2_dataset.py $DIR $thread > $DIR/logs/shard_$thread.log 2>&1 &
done
