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
RUN uv pip install --system --no-cache vllm --torch-backend=auto
RUN uv pip install --system --no-cache \
    accelerate \
    bitsandbytes \
    datasets \
    evaluate \
    inspect-ai \
    inspect-evals \
    lm-eval \
    ninja \
    openai \
    packaging \
    pandas \
    peft \
    scikit-learn \
    transformers \
    trl \
    matplotlib

# Setup workspace
RUN mkdir -p /app
WORKDIR /app
COPY . /app/
RUN chmod -R a+rw /app/ && chmod +x /app/timer.sh
