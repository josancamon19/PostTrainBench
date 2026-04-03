#!/bin/bash
set -e

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

# ============================================================
# Contamination judge (codex CLI)
# ============================================================
echo ""
echo "=== Running Contamination Judge ==="

if [ -f "$TESTS_DIR/contamination_judge.py" ] && [ -n "$BENCHMARK_NAME" ]; then
    JUDGE_TASK=$(python3 "$TESTS_DIR/contamination_judge.py" \
        --model "$MODEL_ID" \
        --benchmark "$BENCHMARK_NAME" 2>/dev/null) || true

    if [ -n "$JUDGE_TASK" ] && [ -n "$OPENAI_API_KEY" ]; then
        echo "Running codex CLI contamination judge..."
        set +e
        cd "$WORKSPACE"
        codex --search -a never exec --json -c model_reasoning_summary=detailed \
            --skip-git-repo-check --yolo --model "gpt-5.1-codex" "$JUDGE_TASK" \
            2>&1 | tee "$LOGS_DIR/judge_output.json"
        set -e

        # Copy judgement files codex writes into workspace
        for f in contamination_judgement.txt disallowed_model_judgement.txt; do
            if [ -f "$WORKSPACE/$f" ]; then
                cp "$WORKSPACE/$f" "$LOGS_DIR/$f"
                echo "$f: $(cat "$LOGS_DIR/$f")"
            fi
        done
        [ -f "$LOGS_DIR/contamination_judgement.txt" ] || echo "no contamination detected (codex did not produce output)" > "$LOGS_DIR/contamination_judgement.txt"
        [ -f "$LOGS_DIR/disallowed_model_judgement.txt" ] || echo "only allowed use detected (codex did not produce output)" > "$LOGS_DIR/disallowed_model_judgement.txt"
    else
        echo "Warning: OPENAI_API_KEY not set or prompt generation failed, skipping judge"
        echo "no contamination detected (judge skipped - no API key)" > "$LOGS_DIR/contamination_judgement.txt"
        echo "only allowed use detected (judge skipped - no API key)" > "$LOGS_DIR/disallowed_model_judgement.txt"
    fi
else
    echo "Warning: contamination_judge.py or metadata not found, skipping judge"
    echo "no contamination detected (judge not available)" > "$LOGS_DIR/contamination_judgement.txt"
    echo "only allowed use detected (judge not available)" > "$LOGS_DIR/disallowed_model_judgement.txt"
fi

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

    # arenahardwriting/healthbench manage their own vLLM server
    EXTRA_ARGS=""
    if [ "$BENCHMARK_ID" != "arenahardwriting" ] && [ "$BENCHMARK_ID" != "healthbench" ]; then
        EXTRA_ARGS="--max-connections 64 --gpu-memory-utilization 0.9"
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
    echo "Accuracy: $ACCURACY"
    echo "$ACCURACY" > "$LOGS_DIR/reward.txt"
else
    echo "ERROR: metrics.json not created after all evaluation attempts"
    echo "0" > "$LOGS_DIR/reward.txt"
fi

echo "=== Verification complete ==="
ls -la "$LOGS_DIR/"
