FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:$PATH"

# System dependencies + deadsnakes PPA for Python 3.12
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# python/python3 symlinks
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

# Install Node.js for npm (needed by agent CLI tools)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install prime CLI + SDK
RUN uv pip install --system --no-cache prime

# Install prime-rl from source (for reference configs and docs)
RUN git clone https://github.com/PrimeIntellect-ai/prime-rl.git /opt/prime-rl

# Utilities
RUN uv pip install --system --no-cache \
    datasets \
    openai \
    pandas \
    transformers

# Setup workspace
RUN mkdir -p /app
WORKDIR /app
COPY . /app/

# Configure prime CLI at build time from .env file, then remove it
RUN if [ -f /app/.env ]; then \
      set -a && . /app/.env && set +a && \
      [ -n "$PRIME_API_KEY" ] && prime config set-api-key "$PRIME_API_KEY"; \
      rm -f /app/.env; \
    fi

RUN chmod -R a+rw /app/ && chmod +x /app/timer.sh
