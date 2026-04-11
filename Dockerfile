FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 系统依赖
RUN apt-get update && apt-get install -y \
    git git-lfs wget curl vim tmux \
    python3.10 python3.10-venv python3-pip \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /workspace

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# 安装 LLaMA-Factory
RUN git clone https://github.com/hiyouga/LLaMA-Factory.git /workspace/LLaMA-Factory && \
    cd /workspace/LLaMA-Factory && \
    pip install -e ".[torch,metrics]"

# 复制项目代码
COPY . /workspace/reframe-vlm
WORKDIR /workspace/reframe-vlm

# 数据和模型放 /workspace/datasets, /workspace/models（挂载 network volume）
ENV DATA_ROOT=/workspace/datasets
ENV MODEL_ROOT=/workspace/models

CMD ["/bin/bash"]
