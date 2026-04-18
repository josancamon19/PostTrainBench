#!/bin/bash

WORKSPACE="/app"
TESTS_DIR="/tests"
LOGS_DIR="/logs/verifier"

mkdir -p "$LOGS_DIR"

echo "=== PostTrainBench Verifier (Tinker) ==="

# ============================================================
# Validate best_checkpoint.txt
# ============================================================
MISSING_CHECKPOINT=0
if [ ! -f "$WORKSPACE/best_checkpoint.txt" ]; then
    echo "WARN: best_checkpoint.txt not found (skipping primary eval, keeping diagnostics)"
    ls -la "$WORKSPACE" > "$LOGS_DIR/workspace_listing.txt" 2>&1
    echo '{"error": "best_checkpoint.txt not found", "accuracy": 0}' > "$LOGS_DIR/metrics.json"
    echo "0" > "$LOGS_DIR/reward.txt"
    MISSING_CHECKPOINT=1
fi

CHECKPOINT=""
if [ "$MISSING_CHECKPOINT" = "0" ]; then
    CHECKPOINT=$(cat "$WORKSPACE/best_checkpoint.txt" | tr -d '[:space:]')
    if [ -z "$CHECKPOINT" ]; then
        echo "WARN: best_checkpoint.txt is empty"
        echo '{"error": "empty checkpoint path", "accuracy": 0}' > "$LOGS_DIR/metrics.json"
        echo "0" > "$LOGS_DIR/reward.txt"
        MISSING_CHECKPOINT=1
    else
        echo "Checkpoint: $CHECKPOINT"
    fi
fi

# ============================================================
# Read metadata
# ============================================================
META=$(python3 -c "import json; print(json.dumps(json.load(open('$TESTS_DIR/metadata.json'))))")
BENCHMARK_ID=$(echo "$META" | python3 -c "import sys,json; print(json.load(sys.stdin)['benchmark_id'])")
BENCHMARK_NAME=$(echo "$META" | python3 -c "import sys,json; print(json.load(sys.stdin)['benchmark_name'])")
MODEL_ID=$(echo "$META" | python3 -c "import sys,json; print(json.load(sys.stdin)['model_id'])")
echo "Benchmark: $BENCHMARK_NAME ($BENCHMARK_ID) | Model: $MODEL_ID"

# ============================================================
# Evaluation: one run, one retry on failure
# ============================================================
cd "$TESTS_DIR"

run_evaluation() {
    local attempt="$1"
    set +e
    python3 "$TESTS_DIR/evaluate.py" \
        --checkpoint "$CHECKPOINT" \
        --base-model "$MODEL_ID" \
        --json-output-file "$LOGS_DIR/metrics.json" \
        2>&1 | tee "$LOGS_DIR/final_eval_${attempt}.txt"
    set -e
}

if [ "$MISSING_CHECKPOINT" = "0" ]; then
    echo ""
    echo "=== Running evaluation on checkpoint ==="
    run_evaluation 1
    if [ ! -f "$LOGS_DIR/metrics.json" ]; then
        echo "--- First attempt failed, retrying once ---"
        sleep 5
        run_evaluation 2
    fi
fi

# ============================================================
# Extract accuracy + compute normalized reward
# ============================================================
echo ""
echo "=== Evaluation complete ==="

if [ -f "$LOGS_DIR/metrics.json" ] && [ "$MISSING_CHECKPOINT" = "0" ]; then
    cat "$LOGS_DIR/metrics.json"
    ACCURACY=$(python3 -c "import json; m=json.load(open('$LOGS_DIR/metrics.json')); print(m.get('accuracy', 0))")
    echo "Raw accuracy: $ACCURACY"

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
    echo "ERROR: metrics.json not created"
    echo "0" > "$LOGS_DIR/reward.txt"
fi

# ============================================================
# Regression suite: same runner as GPU mode. suite.py detects mode from
# metadata and invokes each eval with --checkpoint <URI> --base-model.
# Best-effort; no effect on the primary reward.
# ============================================================
if [ "$MISSING_CHECKPOINT" = "0" ] && [ -f "$LOGS_DIR/metrics.json" ] && [ -f "$TESTS_DIR/regression/suite.py" ]; then
    echo ""
    echo "=== Running regression suite ==="
    set +e
    python3 "$TESTS_DIR/regression/suite.py" \
        --model-path "$CHECKPOINT" \
        --tests-dir "$TESTS_DIR" \
        --logs-dir "$LOGS_DIR" \
        --metadata "$TESTS_DIR/metadata.json" \
        2>&1 | tee "$LOGS_DIR/regression_suite.log"
    set -e
fi

# ============================================================
# Reward-hacking judge: same as GPU mode — verifier-only, zero-gates
# reward.txt on hack.
# ============================================================
if [ -f "$TESTS_DIR/judge/judge.py" ]; then
    echo ""
    echo "=== Reward-hacking judge ==="
    set +e
    python3 "$TESTS_DIR/judge/judge.py" 2>&1 | tee "$LOGS_DIR/judge.log"
    set -e
fi

# Note: no HF upload for Tinker mode — final_model is a Tinker checkpoint
# URI, not on-disk weights to upload.

echo "=== Verification complete ==="
ls -la "$LOGS_DIR/"
