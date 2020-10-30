ARG CUDA_VERSION

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu18.04

ARG DEEPSPEED_COMMIT
ARG TORCH_VERSION
ARG TORCH_VISION_VERSION

##############################################################################
# Installation/Basic Utilities and ninja for JIT compiling support
##############################################################################
RUN apt-get update && \
        apt-get install -y --no-install-recommends \
    software-properties-common sudo pdsh ninja-build

##############################################################################
# Install non-required but useful tools
##############################################################################
RUN apt-get install -y --no-install-recommends \
    wget vim tmux screen

##############################################################################
# Sparse attention dependencies (llvm forces py2, install before py3)
##############################################################################
RUN apt-get install -y --no-install-recommends \
    llvm-9-dev cmake

##############################################################################
# Python
##############################################################################
RUN apt-get install -y --no-install-recommends curl
ENV PYTHON_VERSION=3
RUN apt-get install -y python3 python3-dev && \
    rm -f /usr/bin/python && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
        python get-pip.py && \
        rm get-pip.py && \
    pip install --upgrade pip && \
    # Print python an pip version
    python -V && pip -V

##############################################################################
# Installation Latest Git
##############################################################################
RUN add-apt-repository ppa:git-core/ppa -y && \
    apt-get update && \
    apt-get install -y git && \
    git --version

##############################################################################
# PyTorch
##############################################################################
RUN echo ${CUDA_VERSION} ${TORCH_VERSION} ${TORCH_VISION_VERSION}
RUN pip install torch===${TORCH_VERSION} torchvision===${TORCH_VISION_VERSION} \
    -f https://download.pytorch.org/whl/torch_stable.html

##############################################################################
# Apex
##############################################################################
ENV APEX_COMMIT=494f8ab
RUN git clone https://github.com/NVIDIA/apex.git ${STAGE_DIR}/apex
RUN cd ${STAGE_DIR}/apex && \
    git checkout ${APEX_COMMIT} && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
RUN rm -rf ${STAGE_DIR}/apex

###############################################################################
## DeepSpeed
###############################################################################
RUN echo ${DEEPSPEED_COMMIT}
RUN git clone https://github.com/microsoft/DeepSpeed.git ${STAGE_DIR}/DeepSpeed
RUN cd ${STAGE_DIR}/DeepSpeed && \
    git checkout ${DEEPSPEED_COMMIT} && \
    pip install -r requirements/requirements.txt && \
    DS_BUILD_OPS=1 pip install -v .
RUN rm -rf ${STAGE_DIR}/DeepSpeed
RUN python -c "import deepspeed; print(deepspeed.__version__, deepspeed.__git_hash__, deepspeed.__git_branch__)"
