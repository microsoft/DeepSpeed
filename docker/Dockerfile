FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common pdsh ninja-build \
    llvm-9-dev cmake git && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get purge --auto-remove && \
    apt-get clean

ENV STAGE_DIR=/tmp

ARG DEEPSPEED_COMMIT

RUN git clone https://github.com/microsoft/DeepSpeed.git ${STAGE_DIR}/DeepSpeed && \
    cd ${STAGE_DIR}/DeepSpeed && \
    git checkout ${DEEPSPEED_COMMIT} && \
    pip install -r requirements/requirements.txt && \
    DS_BUILD_OPS=1 pip install -v . && \
    ds_report && \
    rm -r ${STAGE_DIR}/DeepSpeed
