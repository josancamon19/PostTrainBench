#!/bin/bash
set -e

# Oracle solution. When the instruct variant is served on Tinker (e.g.
# Llama-3.1-8B-Instruct, Qwen3-8B, Qwen3-30B-A3B), save its sampling
# weights as the checkpoint so the verifier scores the instruct baseline.
# Otherwise (Llama-3.2-{1B,3B}-Instruct aren't on Tinker), fall back to a
# dummy "save base" checkpoint — a smoke test that just verifies the
# pipeline end-to-end.

cd /app

INSTRUCT_ORACLE_ID="{instruct_oracle_id}"

python3 <<PYEOF
import json
import tinker

instruct_oracle_id = "$INSTRUCT_ORACLE_ID"
meta = json.load(open("metadata.json"))
base_model_id = meta["model_id"]

if instruct_oracle_id:
    print(f"Oracle: saving instruct model {instruct_oracle_id} as checkpoint")
    model_for_sampler = instruct_oracle_id
    name = "instruct_oracle"
else:
    print(f"Oracle unavailable on Tinker; saving base {base_model_id} (smoke test)")
    model_for_sampler = base_model_id
    name = "base_dummy"

service_client = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(base_model=model_for_sampler)
response = training_client.save_weights_for_sampler(name=name).result()
checkpoint_path = response.path
print(f"Checkpoint: {checkpoint_path}")

with open("best_checkpoint.txt", "w") as f:
    f.write(checkpoint_path)
print("Saved best_checkpoint.txt")
PYEOF

echo "Done. best_checkpoint.txt:"
cat best_checkpoint.txt
