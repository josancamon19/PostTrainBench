"""Test script to build and validate the PostTrainBench Docker image on Modal."""

import modal

modal.enable_output()

# Build from our actual Dockerfile
image = modal.Image.from_dockerfile(
    "src/harbor_template/environment/Dockerfile",
    context_dir="src/harbor_template/environment",
)

app = modal.App.lookup("__ptb_image_test__", create_if_missing=True)

print("Creating sandbox with H100 GPU...")
sb = modal.Sandbox.create(app=app, image=image, timeout=120, gpu="H100")

checks = [
    ("GPU check", "nvidia-smi | head -5"),
    (
        "torch CUDA",
        'python3 -c \'import torch; print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), "device:", torch.cuda.get_device_name(0))\'',
    ),
    ("vllm import", "python3 -c 'from vllm import LLM; print(\"vllm OK\")'"),
    ("transformers", "python3 -c 'import transformers; print(\"transformers:\", transformers.__version__)'"),
    ("datasets", "python3 -c 'import datasets; print(\"datasets:\", datasets.__version__)'"),
    ("trl", "python3 -c 'import trl; print(\"trl:\", trl.__version__)'"),
    ("peft", "python3 -c 'import peft; print(\"peft:\", peft.__version__)'"),
    ("inspect_ai", "python3 -c 'import inspect_ai; print(\"inspect_ai OK\")'"),
    ("inspect_evals gsm8k", "python3 -c 'import inspect_evals.gsm8k; print(\"gsm8k OK\")'"),
    ("workspace files", "ls /app/"),
    ("timer", "bash /app/timer.sh"),
]

all_ok = True
for name, cmd in checks:
    proc = sb.exec("bash", "-c", cmd)
    stdout = proc.stdout.read().strip()
    stderr = proc.stderr.read().strip()
    rc = proc.returncode
    status = "OK" if rc == 0 else "FAIL"
    if rc != 0:
        all_ok = False
    print(f"[{status}] {name}: {stdout}")
    if rc != 0 and stderr:
        print(f"  STDERR: {stderr[:300]}")

sb.terminate()

if all_ok:
    print("\nAll checks passed!")
else:
    print("\nSome checks FAILED")
    exit(1)
