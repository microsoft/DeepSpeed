#!/bin/bash

name=${1-deepspeed}
image=deepspeed:0.1
echo "starting docker image named $name"
docker run -d -t --name $name \
        --network host \
        -v ${HOME}/workdir:/home/deepspeed/workdir \
        -v ${HOME}/.ssh:/home/deepspeed/.ssh \
        --gpus all $image bash
