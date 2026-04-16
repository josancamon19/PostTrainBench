#!/bin/bash
set -e

export DEBIAN_FRONTEND=noninteractive
export PATH="/root/.local/bin:$PATH"

# Remove PEP 668 marker so pip/uv can install into system Python
# without --break-system-packages (which can corrupt running processes)
rm -f /usr/lib/python*/EXTERNALLY-MANAGED

# Node.js for agent CLI tools
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# vLLM (pulls compatible torch with CUDA automatically)
uv pip install --system --no-cache vllm

# ML packages
uv pip install --system --no-cache \
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

mkdir -p /app

# Start background GPU sampler — runs for container lifetime.
# Uses setsid to fully detach (nohup alone isn't reliable when the parent is
# an SSH session that harbor disconnects after setup). Silently skipped if
# nvidia-smi is unavailable (e.g. tinker mode, CPU runners).
SAMPLER="$(dirname "$0")/gpu_sampler.sh"
mkdir -p /logs
if command -v nvidia-smi >/dev/null 2>&1 && [ -f "$SAMPLER" ]; then
    chmod +x "$SAMPLER"
    setsid bash "$SAMPLER" < /dev/null > /logs/gpu_sampler.log 2>&1 &
    disown || true
    echo "started gpu_sampler.sh (pid=$!, log=/logs/gpu_sampler.log)"
else
    echo "gpu_sampler: skipped (nvidia-smi or script missing)"
fi
