FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common pdsh ninja-build \
    llvm-9-dev cmake git && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get purge --auto-remove && \
    apt-get clean

RUN DS_BUILD_OPS=1 pip install deepspeed && \
    ds_report
