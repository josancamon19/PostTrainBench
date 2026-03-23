FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:$PATH"

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# python/python3 symlinks
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv pip install --system --no-cache vllm==0.18.0 --torch-backend=auto
RUN uv pip install --system --no-cache \
    accelerate==1.13.0 \
    bitsandbytes==0.49.2 \
    datasets==4.8.3 \
    evaluate==0.4.6 \
    inspect-ai==0.3.200 \
    inspect-evals==0.5.2 \
    lm-eval==0.4.11 \
    matplotlib==3.10.8 \
    ninja \
    openai==2.29.0 \
    packaging \
    pandas==3.0.1 \
    peft==0.18.1 \
    scikit-learn==1.8.0 \
    transformers==5.3.0 \
    trl==0.29.1

# Setup workspace
RUN mkdir -p /app
WORKDIR /app
COPY . /app/
RUN chmod -R a+rw /app/ && chmod +x /app/timer.sh
