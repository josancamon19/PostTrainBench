#!/bin/bash
set -e

REPO="ghcr.io/josancamon19/posttrainbench"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Building per-task GPU images ==="
# Export tasks first to get the environment directories
cd "$ROOT_DIR"
python "$SCRIPT_DIR/adapter.py" mode=gpu all=true include_target=false 2>/dev/null

PIDS=()
for task_dir in "$ROOT_DIR/datasets/posttrainbench/gpu"/*/; do
    task_id=$(basename "$task_dir")
    tag="$REPO-gpu:$task_id"
    echo "Building $tag..."
    docker build --platform linux/amd64 -t "$tag" -f "$task_dir/environment/Dockerfile" "$task_dir/environment" &
    PIDS+=($!)
done

# Wait for all builds
for pid in "${PIDS[@]}"; do
    wait $pid || echo "WARNING: build $pid failed"
done

echo "Pushing per-task images..."
PIDS=()
for task_dir in "$ROOT_DIR/datasets/posttrainbench/gpu"/*/; do
    task_id=$(basename "$task_dir")
    tag="$REPO-gpu:$task_id"
    docker push "$tag" &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
    wait $pid || echo "WARNING: push $pid failed"
done

echo "=== Done ==="
