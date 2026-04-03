#!/bin/bash
set -e

REPO="ghcr.io/josancamon19/posttrainbench"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR="$SCRIPT_DIR/harbor_template/environment"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Building base images ==="
docker build --platform linux/amd64 -t "$REPO-gpu:latest" -f "$ENV_DIR/base.Dockerfile" "$ENV_DIR" &
PID_GPU=$!
docker build --platform linux/amd64 -t "$REPO-tinker:latest" -f "$ENV_DIR/base.tinker.Dockerfile" "$ENV_DIR" &
PID_TINKER=$!
wait $PID_GPU && echo "GPU base built." || { echo "GPU base build failed"; exit 1; }
wait $PID_TINKER && echo "Tinker base built." || { echo "Tinker base build failed"; exit 1; }

echo "Pushing base images..."
docker push "$REPO-gpu:latest" &
docker push "$REPO-tinker:latest" &
wait

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
