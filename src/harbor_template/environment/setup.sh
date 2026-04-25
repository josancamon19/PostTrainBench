#!/bin/bash
# Verbose setup for gpu-runpod. Every step is timestamped and tee'd to
# /var/log/harbor-setup.log on the pod so that when a build times out we
# can still see which install stalled (harbor's exec also captures stdout,
# but that buffer can truncate on long-running installs).
set -e
mkdir -p /var/log
exec > >(tee -a /var/log/harbor-setup.log) 2>&1

ts() { echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*"; }
trap 'ts "FAILED at line $LINENO (last exit $?)"' ERR

export DEBIAN_FRONTEND=noninteractive
export PATH="/root/.local/bin:$PATH"

ts "setup.sh start  (host=$(hostname), gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a))"

# Remove PEP 668 marker so pip/uv can install into system Python
# without --break-system-packages (which can corrupt running processes).
rm -f /usr/lib/python*/EXTERNALLY-MANAGED
ts "removed EXTERNALLY-MANAGED marker"

# Node.js for agent CLI tools
ts "installing Node.js 22 via nodesource..."
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs
ts "Node.js ready ($(node --version))"

# uv
ts "installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
ts "uv ready ($(uv --version 2>/dev/null || echo unknown))"

# vLLM (pulls compatible torch with CUDA automatically)
ts "installing vllm (+ torch+cuda, usually the biggest single cost)..."
uv pip install --system --no-cache vllm
ts "vllm installed"

# ML packages
ts "installing ML + eval packages..."
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
ts "ML packages installed"

mkdir -p /app
ts "created /app"

# Background GPU sampler. Written to /usr/local/bin (NOT /app — the agent
# must not see the sampler script or be able to kill it by path). Output
# /logs/compute_samples.csv is outside the agent's /app cwd.
mkdir -p /logs
SAMPLER=/usr/local/bin/gpu_sampler.sh
cat > "$SAMPLER" <<'SAMPLER_EOF'
#!/bin/bash
set -u
SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL:-30}"
OUT_FILE="${GPU_SAMPLER_OUT:-/logs/compute_samples.csv}"
LOCK_FILE="${OUT_FILE}.lock"
OUT_DIR="$(dirname "$OUT_FILE")"
MARKER_FILE="${GPU_SAMPLER_MARKER:-${OUT_DIR}/gpu_sampler.started}"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >&2; }
cleanup() { log "sampler stopping (signal received)"; rm -f "$LOCK_FILE"; exit 0; }
trap cleanup TERM INT HUP
mkdir -p "$OUT_DIR" || { log "ERROR: cannot create $OUT_DIR"; exit 1; }
if [ -f "$LOCK_FILE" ]; then
    OLD_PID=$(cat "$LOCK_FILE" 2>/dev/null)
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        log "sampler already running (pid=$OLD_PID); exiting"; exit 0
    fi
fi
echo $$ > "$LOCK_FILE"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi not found; sampler skipping"; rm -f "$LOCK_FILE"; exit 0
fi
if ! nvidia-smi --query-gpu=index --format=csv,noheader >/dev/null 2>&1; then
    log "nvidia-smi present but failing; sampler skipping"; rm -f "$LOCK_FILE"; exit 0
fi
if [ ! -s "$OUT_FILE" ]; then
    echo "timestamp,gpu_index,utilization_gpu,utilization_memory,memory_used_mib,memory_total_mib,power_draw_w,temperature_c" > "$OUT_FILE"
fi
mkdir -p "$(dirname "$MARKER_FILE")"
date -u +%Y-%m-%dT%H:%M:%SZ > "$MARKER_FILE"
log "sampler started (interval=${SAMPLE_INTERVAL}s, out=${OUT_FILE})"
CONSECUTIVE_FAILURES=0
MAX_CONSECUTIVE_FAILURES=20
while true; do
    TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    TMP=$(mktemp 2>/dev/null) || TMP="/tmp/gpu_sampler.$$"
    if nvidia-smi \
        --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv,noheader,nounits \
        >"$TMP" 2>/dev/null; then
        CONSECUTIVE_FAILURES=0
        while IFS= read -r line; do
            [ -z "${line// /}" ] && continue
            case "$line" in [Nn]ot*[Ss]upported*) continue ;; esac
            printf '%s,%s\n' "$TS" "${line// /}" >> "$OUT_FILE"
        done <"$TMP"
    else
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        if [ "$CONSECUTIVE_FAILURES" -ge "$MAX_CONSECUTIVE_FAILURES" ]; then
            log "nvidia-smi failed ${CONSECUTIVE_FAILURES} times in a row; giving up"
            rm -f "$TMP" "$LOCK_FILE"; exit 0
        fi
    fi
    rm -f "$TMP"
    sleep "$SAMPLE_INTERVAL"
done
SAMPLER_EOF
chmod +x "$SAMPLER"
if command -v nvidia-smi >/dev/null 2>&1; then
    setsid bash "$SAMPLER" < /dev/null > /logs/gpu_sampler.log 2>&1 &
    disown || true
    ts "started gpu_sampler (pid=$!, script=$SAMPLER, log=/logs/gpu_sampler.log)"
else
    ts "gpu_sampler: skipped (no nvidia-smi)"
fi

ts "setup.sh complete"
