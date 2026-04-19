FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Tinker SDK, dependencies, judge
RUN uv pip install --system --no-cache \
    tinker \
    tinker-cookbook \
    datasets \
    openai \
    anthropic \
    claude-agent-sdk \
    if-verifiable \
    boto3 \
    python-dotenv \
    requests \
    shortuuid \
    tiktoken \
    tqdm

# Setup workspace
RUN mkdir -p /app
WORKDIR /app
