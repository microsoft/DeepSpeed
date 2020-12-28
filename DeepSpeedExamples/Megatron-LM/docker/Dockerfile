# ===========
# base images
# ===========
FROM nvcr.io/nvidia/pytorch:19.05-py3


# ===============
# system packages
# ===============
RUN apt-get update && apt-get install -y \
    bash-completion \
    emacs \
    git \
    graphviz \
    htop \
    libopenexr-dev \
    rsync \
    wget \
&& rm -rf /var/lib/apt/lists/*


# ============
# pip packages
# ============
RUN pip install --upgrade pip && \
    pip install --upgrade setuptools
COPY requirements.txt /tmp/
RUN pip install --upgrade --ignore-installed -r /tmp/requirements.txt


# ===========
# latest apex
# ===========
RUN pip uninstall -y apex && \
git clone https://github.com/NVIDIA/apex.git ~/apex && \
cd ~/apex && \
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

