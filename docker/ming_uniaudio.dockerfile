FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list && \
    apt-get clean && \
    apt-get update

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        vim \
        libsndfile1 \
        build-essential \
        python3-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/pip3.11 /usr/bin/pip

RUN pip install --no-cache-dir --upgrade pip
RUN pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple/

RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchaudio==2.6.0 \
    torchvision==0.21.0


ENV FORCE_CUDA="1"
# A100: 8.0, RTX 30xx: 8.6, RTX 40xx/H100: 8.9 or 9.0
ENV TORCH_CUDA_ARCH_LIST="8.0"
ENV GROUPED_GEMM_CUTLASS="1"

ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_ROOT=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN pip install --no-cache-dir --no-build-isolation \
    tokenizers \
    # grouped_gemm==0.1.4 \
    peft==0.13.2 \
    diffusers==0.33.0 \
    decord==0.6.0 \
    hyperpyyaml==1.2.2 \
    soundfile==0.12.1 \
    transformers==4.52.4 \
    x_transformers \
    torchdiffeq \
    torchtune \
    torchao

ARG FLASH_ATTN_WHEEL=flash_attn-2.7.0.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
ARG FLASH_ATTN_URL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post1/${FLASH_ATTN_WHEEL}
RUN wget -q ${FLASH_ATTN_URL} -O /tmp/${FLASH_ATTN_WHEEL} && \
    pip install --no-cache-dir /tmp/${FLASH_ATTN_WHEEL} && \
    rm /tmp/${FLASH_ATTN_WHEEL}

WORKDIR /app

CMD ["/bin/bash"]
