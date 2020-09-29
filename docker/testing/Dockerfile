ARG CUDA_VERSION

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu18.04

ARG BUILD_ID

##############################################################################
# Installation/Basic Utilities
##############################################################################
RUN apt-get update && \
	apt-get install -y --no-install-recommends \
    software-properties-common sudo pdsh

##############################################################################
# Install non-required but useful tools
##############################################################################
RUN apt-get install -y --no-install-recommends \
    wget vim tmux screen ninja-build

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
## Add deepspeed user
###############################################################################
RUN apt-get install -y sudo
RUN useradd --create-home --uid 1000 --shell /bin/bash deepspeed
RUN usermod -aG sudo deepspeed
RUN echo "deepspeed ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
# Change to non-root privilege
USER deepspeed
