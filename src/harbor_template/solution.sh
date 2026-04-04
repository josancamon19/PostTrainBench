#!/bin/bash
set -e

# Oracle solution: download instruct model to final_model/ for baseline scoring.
# The verifier evaluates this as-is, giving us the instruct baseline score.

cd /app

INSTRUCT_ID="{instruct_id}"

python3 <<PYEOF
from huggingface_hub import snapshot_download

instruct_id = "$INSTRUCT_ID"
print(f"Downloading instruct model: {instruct_id}")
snapshot_download(instruct_id, local_dir="/app/final_model")
print("Saved to /app/final_model")
PYEOF

echo "Done. final_model contents:"
ls /app/final_model/
