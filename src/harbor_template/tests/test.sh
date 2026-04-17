#!/bin/bash

WORKSPACE="/app"
TESTS_DIR="/tests"
LOGS_DIR="/logs/verifier"

mkdir -p "$LOGS_DIR"

echo "=== PostTrainBench Verifier ==="

# GPU check
nvidia-smi 2>&1 | tee "$LOGS_DIR/gpu_check.txt" || echo "nvidia-smi failed"

# Validate final_model
VALIDATE=$(python3 "$TESTS_DIR/verifier.py" validate 2>&1)
MISSING_FINAL_MODEL=0
if [ "$VALIDATE" != "ok" ]; then
    echo "WARN: $VALIDATE (skipping primary + regression eval, keeping diagnostics)"
    ls -la "$WORKSPACE" > "$LOGS_DIR/workspace_listing.txt" 2>&1
    echo "{\"error\": \"$VALIDATE\", \"accuracy\": 0}" > "$LOGS_DIR/metrics.json"
    echo "0" > "$LOGS_DIR/reward.txt"
    MISSING_FINAL_MODEL=1
fi

if [ "$MISSING_FINAL_MODEL" = "0" ]; then
    echo "Contents of final_model:"
    ls -la "$WORKSPACE/final_model" | tee "$LOGS_DIR/final_model_listing.txt"
    cat "$WORKSPACE/final_model/config.json" | head -50 | tee "$LOGS_DIR/model_config.txt"
fi

# Read metadata (always — needed for compute/regression regardless of final_model)
META=$(python3 "$TESTS_DIR/verifier.py" metadata)
BENCHMARK_ID=$(echo "$META" | python3 -c "import sys,json; print(json.load(sys.stdin)['benchmark_id'])")
BENCHMARK_NAME=$(echo "$META" | python3 -c "import sys,json; print(json.load(sys.stdin)['benchmark_name'])")
MODEL_ID=$(echo "$META" | python3 -c "import sys,json; print(json.load(sys.stdin)['model_id'])")
echo "Benchmark: $BENCHMARK_NAME ($BENCHMARK_ID) | Model: $MODEL_ID"

# ============================================================
# Evaluation: one full run, one retry on failure. Skipped if final_model
# is missing.
# ============================================================
cd "$TESTS_DIR"

kill_gpu_processes() {
    # Kill GPU-holding processes except PID 1 (container init)
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | grep -v '^$' \
        | while read pid; do
            [ "$pid" -gt 1 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
        done
    sleep 5
}

run_evaluation() {
    local attempt="$1"
    kill_gpu_processes

    MAX_CONN=$(python3 -c "import json; print(json.load(open('$TESTS_DIR/metadata.json')).get('max_connections', 64))")
    EXTRA_ARGS="--max-connections $MAX_CONN"
    # inspect-ai benchmarks need gpu-memory-utilization; arenahardwriting/healthbench manage their own vLLM
    if [ "$BENCHMARK_ID" != "arenahardwriting" ] && [ "$BENCHMARK_ID" != "healthbench" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --gpu-memory-utilization 0.9"
    fi

    set +e
    python3 "$TESTS_DIR/evaluate.py" \
        --model-path "$WORKSPACE/final_model" \
        --json-output-file "$LOGS_DIR/metrics.json" \
        --templates-dir "$TESTS_DIR/templates/" \
        --limit -1 \
        ${EXTRA_ARGS} \
        2>&1 | tee "$LOGS_DIR/final_eval_${attempt}.txt"
    set -e
}

if [ "$MISSING_FINAL_MODEL" = "0" ]; then
    echo ""
    echo "=== Running evaluation on final_model ==="
    run_evaluation 1
    if [ ! -f "$LOGS_DIR/metrics.json" ]; then
        echo "--- First attempt failed, retrying once ---"
        sleep 5
        run_evaluation 2
    fi
fi

# ============================================================
# Extract accuracy and write reward
# ============================================================
echo ""
echo "=== Evaluation complete ==="

if [ -f "$LOGS_DIR/metrics.json" ] && [ "$MISSING_FINAL_MODEL" = "0" ]; then
    cat "$LOGS_DIR/metrics.json"
    ACCURACY=$(python3 "$TESTS_DIR/verifier.py" accuracy "$LOGS_DIR/metrics.json")
    echo "Raw accuracy: $ACCURACY"

    # Compute normalized reward: (score - base) / (target - base)
    REWARD=$(python3 -c "
import json
meta = json.load(open('$TESTS_DIR/metadata.json'))
score = float($ACCURACY)
base = meta.get('base_score')
target = meta.get('target_score')
if base is not None and target is not None and target != base:
    reward = max((score - base) / (target - base), 0.0)
else:
    reward = score
print(f'{reward:.6f}')
")
    echo "Normalized reward: $REWARD"
    echo "$REWARD" > "$LOGS_DIR/reward.txt"
else
    echo "ERROR: metrics.json not created after all evaluation attempts"
    echo "0" > "$LOGS_DIR/reward.txt"
fi

# ============================================================
# Compute utilization: parse samples collected during the agent phase.
# Best-effort — no impact on reward. Silently writes status=no_samples if
# the sampler didn't run (tinker mode, CPU-only, or pre-built GPU images).
# ============================================================
if [ -f "$TESTS_DIR/compute/parser.py" ]; then
    python3 "$TESTS_DIR/compute/parser.py" \
        --input /logs/compute_samples.csv \
        --output "$LOGS_DIR/compute.json" 2>&1 | tee -a "$LOGS_DIR/compute_parser.log" || true
fi

# ============================================================
# Regression suite: run other benchmarks against final_model to detect
# catastrophic forgetting and domain generalization. Best-effort; a failure
# here does NOT affect the primary reward written above.
# ============================================================
if [ "$MISSING_FINAL_MODEL" = "0" ] && [ -f "$LOGS_DIR/metrics.json" ] && [ -f "$TESTS_DIR/regression/suite.py" ]; then
    echo ""
    echo "=== Running regression suite ==="
    kill_gpu_processes
    set +e
    python3 "$TESTS_DIR/regression/suite.py" \
        --model-path "$WORKSPACE/final_model" \
        --tests-dir "$TESTS_DIR" \
        --logs-dir "$LOGS_DIR" \
        --metadata "$TESTS_DIR/metadata.json" \
        2>&1 | tee "$LOGS_DIR/regression_suite.log"
    set -e
fi

# ============================================================
# Reward-hacking judge: runs LLM over /app + /logs/agent to flag
# contamination / grader tampering / instruct substitution. Verifier-only;
# never visible to the agent. Best-effort: missing auth or API errors log
# a stub and exit 0 — does not affect the primary reward.
# ============================================================
if [ -f "$TESTS_DIR/judge/judge.py" ]; then
    echo ""
    echo "=== Reward-hacking judge ==="
    set +e
    python3 "$TESTS_DIR/judge/judge.py" 2>&1 | tee "$LOGS_DIR/judge.log"
    set -e
fi

# ============================================================
# Offload final_model to HuggingFace, then purge large files from /app
# so Harbor's artifact rsync doesn't drag multi-GB weights back.
# Best-effort; failure here does not affect reward.
# ============================================================
if [ "$MISSING_FINAL_MODEL" = "0" ]; then
    echo ""
    echo "=== HF upload + artifact slimming ==="
    set +e
    python3 "$TESTS_DIR/hooks/hf_upload.py" 2>&1 | tee -a "$LOGS_DIR/hf_upload.log"
    # Drop any file over 500M from the workspace after the upload. Keeps
    # config.json / tokenizer / small checkpoints for reconstruction.
    find "$WORKSPACE" -type f -size +500M -print -delete 2>&1 \
        | tee "$LOGS_DIR/pruned_large_files.txt"
    set -e
fi

echo "=== Verification complete ==="
ls -la "$LOGS_DIR/"
