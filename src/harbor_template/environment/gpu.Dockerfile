FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:$PATH"

# System dependencies (Python 3.11 via deadsnakes; if-verifiable needs >=3.11).
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# vLLM pulls a compatible CUDA Torch stack.
RUN uv pip install --system --no-cache vllm

RUN uv pip install --system --no-cache \
    accelerate \
    bitsandbytes \
    datasets \
    evaluate \
    inspect-ai \
    inspect-evals \
    lm-eval \
    matplotlib \
    ninja \
    "openai>=2.26.0" \
    anthropic \
    claude-agent-sdk \
    if-verifiable \
    packaging \
    pandas \
    peft \
    python-dotenv \
    requests \
    scikit-learn \
    shortuuid \
    tiktoken \
    transformers \
    tqdm \
    trl

RUN mkdir -p /app

COPY . /app/
RUN chmod -R a+rw /app/ && chmod +x /app/timer.sh
