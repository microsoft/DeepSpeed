#!/bin/bash

CUDA_VERSION=${1:-10.0}

BUILD_ID=`date +%m%d%Y`
docker build \
    -t deepspeed:testing-cuda${CUDA_VERSION}-${BUILD_ID} \
    -f Dockerfile \
    --build-arg CUDA_VERSION=$CUDA_VERSION \
    --build-arg BUILD_ID=${BUILD_ID} \
    .
