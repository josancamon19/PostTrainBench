#!/bin/bash
# Background nvidia-smi sampler for the agent phase.
# Runs for the container's lifetime, dies naturally when container is torn down.
# Writes CSV rows to /logs/compute_samples.csv (accessible by the verifier).

set -u

SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL:-30}"
OUT_FILE="/logs/compute_samples.csv"

mkdir -p /logs
# Write header if file is new
if [ ! -f "$OUT_FILE" ]; then
    echo "timestamp,gpu_index,utilization_gpu,utilization_memory,memory_used_mib,memory_total_mib,power_draw_w,temperature_c" > "$OUT_FILE"
fi

while true; do
    # Query format mirrors columns above. Timestamp is added client-side so it's
    # stable even if nvidia-smi's own timestamp format changes.
    TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    if nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null \
        | while IFS= read -r line; do
            # nvidia-smi already returns comma-separated. Prepend timestamp.
            echo "${TS},${line// /}"
        done >> "$OUT_FILE"; then
        :
    fi
    sleep "$SAMPLE_INTERVAL"
done
