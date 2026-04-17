#!/bin/bash
# Background nvidia-smi sampler for the agent phase. Writes CSV rows to
# /logs/compute_samples.csv (read by the verifier's compute/parser.py).
#
# Runs for the container's lifetime. Designed to survive transient
# nvidia-smi failures, absent GPUs, and mid-run signals without silent
# data loss.

set -u

SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL:-30}"
OUT_FILE="${GPU_SAMPLER_OUT:-/logs/compute_samples.csv}"
LOCK_FILE="${OUT_FILE}.lock"
OUT_DIR="$(dirname "$OUT_FILE")"
MARKER_FILE="${GPU_SAMPLER_MARKER:-${OUT_DIR}/gpu_sampler.started}"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >&2; }

cleanup() {
    log "sampler stopping (signal received)"
    rm -f "$LOCK_FILE"
    exit 0
}
trap cleanup TERM INT HUP

mkdir -p "$OUT_DIR" || { log "ERROR: cannot create $OUT_DIR"; exit 1; }

# Prevent concurrent samplers — a second instance would double-log every row.
if [ -f "$LOCK_FILE" ]; then
    OLD_PID=$(cat "$LOCK_FILE" 2>/dev/null)
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        log "sampler already running (pid=$OLD_PID); exiting"
        exit 0
    fi
fi
echo $$ > "$LOCK_FILE"

# Detect nvidia-smi availability up front. Persistent failure → exit cleanly
# (compute.json will show status=no_samples; the verifier doesn't block on it).
if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi not found; sampler skipping"
    rm -f "$LOCK_FILE"
    exit 0
fi
if ! nvidia-smi --query-gpu=index --format=csv,noheader >/dev/null 2>&1; then
    log "nvidia-smi present but failing; sampler skipping"
    rm -f "$LOCK_FILE"
    exit 0
fi

# Header is written exactly once, before the first sample row.
if [ ! -s "$OUT_FILE" ]; then
    echo "timestamp,gpu_index,utilization_gpu,utilization_memory,memory_used_mib,memory_total_mib,power_draw_w,temperature_c" > "$OUT_FILE"
fi

# Touch a verifier-visible marker so compute/parser.py can distinguish
# "sampler never started" from "sampler started but produced 0 rows".
mkdir -p "$(dirname "$MARKER_FILE")"
date -u +%Y-%m-%dT%H:%M:%SZ > "$MARKER_FILE"
log "sampler started (interval=${SAMPLE_INTERVAL}s, out=${OUT_FILE})"

CONSECUTIVE_FAILURES=0
MAX_CONSECUTIVE_FAILURES=20

while true; do
    TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    # Capture into a tempfile first so a partial / malformed read doesn't
    # corrupt the CSV if nvidia-smi dies mid-output.
    TMP=$(mktemp 2>/dev/null) || TMP="/tmp/gpu_sampler.$$"
    if nvidia-smi \
        --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv,noheader,nounits \
        >"$TMP" 2>/dev/null; then
        CONSECUTIVE_FAILURES=0
        while IFS= read -r line; do
            # Skip blank / header-echo / obvious garbage rows.
            [ -z "${line// /}" ] && continue
            case "$line" in [Nn]ot*[Ss]upported*) continue ;; esac
            printf '%s,%s\n' "$TS" "${line// /}" >> "$OUT_FILE"
        done <"$TMP"
    else
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        if [ "$CONSECUTIVE_FAILURES" -ge "$MAX_CONSECUTIVE_FAILURES" ]; then
            log "nvidia-smi failed ${CONSECUTIVE_FAILURES} times in a row; giving up"
            rm -f "$TMP" "$LOCK_FILE"
            exit 0
        fi
    fi
    rm -f "$TMP"
    sleep "$SAMPLE_INTERVAL"
done
