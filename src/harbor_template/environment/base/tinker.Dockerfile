FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# codex CLI (for contamination judge)
RUN npm install -g @openai/codex

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Tinker SDK and dependencies
RUN uv pip install --system --no-cache \
    tinker \
    tinker-cookbook \
    datasets \
    openai \
    boto3 \
    python-dotenv \
    requests \
    shortuuid \
    tiktoken \
    tqdm

# Setup workspace
RUN mkdir -p /app
WORKDIR /app
