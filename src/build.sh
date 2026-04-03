#!/bin/bash
set -e

REPO="ghcr.io/josancamon19/posttrainbench"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR="$SCRIPT_DIR/harbor_template/environment"

echo "Building GPU and Tinker base images in parallel..."

docker build --platform linux/amd64 -t "$REPO-gpu:latest" -f "$ENV_DIR/base.Dockerfile" "$ENV_DIR" &
PID_GPU=$!

docker build --platform linux/amd64 -t "$REPO-tinker:latest" -f "$ENV_DIR/base.tinker.Dockerfile" "$ENV_DIR" &
PID_TINKER=$!

wait $PID_GPU && echo "GPU image built." || { echo "GPU build failed"; exit 1; }
wait $PID_TINKER && echo "Tinker image built." || { echo "Tinker build failed"; exit 1; }

echo "Pushing images..."
docker push "$REPO-gpu:latest" &
docker push "$REPO-tinker:latest" &
wait

echo "Done."
