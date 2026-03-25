"""Quick smoke test: start a short prime-rl training run on Qwen3-4B with GSM8K."""

import subprocess
import sys
import tempfile
from pathlib import Path

CONFIG = """
model = "Qwen/Qwen3-4B-Instruct-2507"
max_steps = 5

batch_size = 32
rollouts_per_example = 4

[sampling]
max_tokens = 512

[[env]]
id = "primeintellect/math-env"
args = { dataset_name = "openai/gsm8k", dataset_subset = "main", max_examples = 50 }

[eval]
interval = 5
num_examples = 10
eval_base_model = true

[[eval.env]]
id = "primeintellect/math-env"
args = { dataset_name = "openai/gsm8k", dataset_subset = "main" }
""".strip()


def main():
    # 1. Check API key
    result = subprocess.run(["prime", "whoami", "--plain"], capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: prime CLI not authenticated. Run: prime config set-api-key <your-key>")
        sys.exit(1)
    print(f"Authenticated: {result.stdout.strip()}")

    # 2. Verify model is available
    result = subprocess.run(["prime", "rl", "models", "--plain"], capture_output=True, text=True)
    if "Qwen3-4B-Instruct-2507" not in result.stdout:
        print("ERROR: Qwen/Qwen3-4B-Instruct-2507 not available")
        print(result.stdout)
        sys.exit(1)
    print("Model available: Qwen/Qwen3-4B-Instruct-2507")

    # 3. Write config
    config_path = Path("test_rl_config.toml")
    config_path.write_text(CONFIG)
    print(f"Config written to {config_path}")

    # 4. Start training run
    print("\nStarting training run (5 steps, should take ~2-5 minutes)...")
    result = subprocess.run(
        ["prime", "rl", "run", str(config_path), "--plain"],
        text=True,
    )

    if result.returncode != 0:
        print(f"Training failed with exit code {result.returncode}")
        sys.exit(1)

    print("\nTraining complete! Checking runs...")

    # 5. List runs to verify
    subprocess.run(["prime", "rl", "list", "--plain"])

    # Cleanup
    config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
