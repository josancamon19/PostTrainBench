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
if [ "$VALIDATE" != "ok" ]; then
    echo "ERROR: $VALIDATE"
    ls -la "$WORKSPACE" > "$LOGS_DIR/workspace_listing.txt" 2>&1
    echo "{\"error\": \"$VALIDATE\", \"accuracy\": 0}" > "$LOGS_DIR/metrics.json"
    echo "0" > "$LOGS_DIR/reward.txt"
    exit 0
fi

echo "Contents of final_model:"
ls -la "$WORKSPACE/final_model" | tee "$LOGS_DIR/final_model_listing.txt"
cat "$WORKSPACE/final_model/config.json" | head -50 | tee "$LOGS_DIR/model_config.txt"

# Read metadata
META=$(python3 "$TESTS_DIR/verifier.py" metadata)
BENCHMARK_ID=$(echo "$META" | python3 -c "import sys,json; print(json.load(sys.stdin)['benchmark_id'])")
BENCHMARK_NAME=$(echo "$META" | python3 -c "import sys,json; print(json.load(sys.stdin)['benchmark_name'])")
MODEL_ID=$(echo "$META" | python3 -c "import sys,json; print(json.load(sys.stdin)['model_id'])")
echo "Benchmark: $BENCHMARK_NAME ($BENCHMARK_ID) | Model: $MODEL_ID"

# Contamination judge (disabled — workspace artifacts alone are insufficient for
# long trajectories; need trajectory-level analysis to detect deliberate cheating)
# source "$TESTS_DIR/contamination.sh"

# ============================================================
# Evaluation with 3-phase retry logic
# ============================================================
echo ""
echo "=== Running evaluation on final_model ==="
cd "$TESTS_DIR"

EVAL_COUNTER=0

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
    local max_tokens_arg="$1"
    EVAL_COUNTER=$((EVAL_COUNTER + 1))
    echo "Evaluation attempt $EVAL_COUNTER"

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
        ${max_tokens_arg} \
        2>&1 | tee "$LOGS_DIR/final_eval_${EVAL_COUNTER}.txt"
    set -e
}

run_phase() {
    local max_retries="$1"
    local max_tokens_arg="$2"

    for ((attempt=1; attempt<=max_retries; attempt++)); do
        [ -f "$LOGS_DIR/metrics.json" ] && return 0
        sleep 5
        run_evaluation "$max_tokens_arg"
    done
}

PHASE2_TOKENS=$(python3 "$TESTS_DIR/verifier.py" token-args "$BENCHMARK_ID" 2)
PHASE3_TOKENS=$(python3 "$TESTS_DIR/verifier.py" token-args "$BENCHMARK_ID" 3)

echo "--- Phase 1: default token limits (up to 4 attempts) ---"
run_phase 4 ""

echo "--- Phase 2: reduced tokens [${PHASE2_TOKENS}] (up to 3 attempts) ---"
run_phase 3 "$PHASE2_TOKENS"

echo "--- Phase 3: further reduced tokens [${PHASE3_TOKENS}] (up to 2 attempts) ---"
run_phase 2 "$PHASE3_TOKENS"

# ============================================================
# Extract accuracy and write reward
# ============================================================
echo ""
echo "=== Evaluation complete (${EVAL_COUNTER} total attempts) ==="

if [ -f "$LOGS_DIR/metrics.json" ]; then
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
# Regression suite: run other benchmarks against final_model to detect
# catastrophic forgetting and domain generalization. Best-effort; a failure
# here does NOT affect the primary reward written above.
# ============================================================
if [ -f "$LOGS_DIR/metrics.json" ] && [ -f "$TESTS_DIR/regression_suite.py" ]; then
    echo ""
    echo "=== Running regression suite ==="
    kill_gpu_processes
    set +e
    python3 "$TESTS_DIR/regression_suite.py" \
        --model-path "$WORKSPACE/final_model" \
        --tests-dir "$TESTS_DIR" \
        --logs-dir "$LOGS_DIR" \
        --metadata "$TESTS_DIR/metadata.json" \
        2>&1 | tee "$LOGS_DIR/regression_suite.log"
    set -e
fi

echo "=== Verification complete ==="
ls -la "$LOGS_DIR/"
