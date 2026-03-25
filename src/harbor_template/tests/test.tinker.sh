#!/bin/bash
set -e

WORKSPACE="/app"
TESTS_DIR="/tests"
LOGS_DIR="/logs/verifier"

mkdir -p "$LOGS_DIR"

echo "=== PostTrainBench Verifier (Tinker) ==="

# Validate best_checkpoint.txt exists
if [ ! -f "$WORKSPACE/best_checkpoint.txt" ]; then
    echo "ERROR: best_checkpoint.txt not found"
    ls -la "$WORKSPACE" > "$LOGS_DIR/workspace_listing.txt" 2>&1
    echo '{"error": "best_checkpoint.txt not found", "accuracy": 0}' > "$LOGS_DIR/metrics.json"
    echo "0" > "$LOGS_DIR/reward.txt"
    exit 0
fi

CHECKPOINT=$(cat "$WORKSPACE/best_checkpoint.txt" | tr -d '[:space:]')
echo "Checkpoint: $CHECKPOINT"

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: best_checkpoint.txt is empty"
    echo '{"error": "empty checkpoint path", "accuracy": 0}' > "$LOGS_DIR/metrics.json"
    echo "0" > "$LOGS_DIR/reward.txt"
    exit 0
fi

# Read metadata
META=$(python3 -c "import json; print(json.dumps(json.load(open('$WORKSPACE/metadata.json'))))")
BENCHMARK_ID=$(echo "$META" | python3 -c "import sys,json; print(json.load(sys.stdin)['benchmark_id'])")
MODEL_ID=$(echo "$META" | python3 -c "import sys,json; print(json.load(sys.stdin)['model_id'])")
echo "Benchmark: $BENCHMARK_ID | Model: $MODEL_ID"

# Run evaluation
echo ""
echo "=== Running evaluation on checkpoint ==="
cd "$TESTS_DIR"

set +e
python3 "$TESTS_DIR/evaluate.py" \
    --checkpoint "$CHECKPOINT" \
    --base-model "$MODEL_ID" \
    --json-output-file "$LOGS_DIR/metrics.json" \
    2>&1 | tee "$LOGS_DIR/final_eval.txt"
set -e

# Extract accuracy and write reward
echo ""
echo "=== Evaluation complete ==="

if [ -f "$LOGS_DIR/metrics.json" ]; then
    cat "$LOGS_DIR/metrics.json"
    ACCURACY=$(python3 -c "import json; m=json.load(open('$LOGS_DIR/metrics.json')); print(m.get('accuracy', 0))")
    echo "Accuracy: $ACCURACY"
    echo "$ACCURACY" > "$LOGS_DIR/reward.txt"
else
    echo "ERROR: metrics.json not created"
    echo "0" > "$LOGS_DIR/reward.txt"
fi

echo "=== Verification complete ==="
ls -la "$LOGS_DIR/"
