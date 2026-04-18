#!/usr/bin/env python3
"""Measure MMLU accuracy on Modal GPU.

Use for instruct models Tinker doesn't serve (Llama-3.2-{1B,3B}-Instruct).
Full eval by default; writes /tmp/baselines/mmlu-<slug>.json so results merge
with Tinker runs.

Usage:
    python scripts/mmlu_modal.py meta-llama/Llama-3.2-1B-Instruct
    python scripts/mmlu_modal.py m1,m2,m3   # sequential
"""

import json
import os
import sys
from pathlib import Path

import modal

IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install(
    "vllm==0.8.5",
    "transformers==4.51.3",
    "datasets>=2.0",
    "huggingface-hub>=0.25",
)

app = modal.App("mmlu-baselines")
_secrets = [modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})] if os.environ.get("HF_TOKEN") else []


@app.function(image=IMAGE, gpu="H100", timeout=7200, secrets=_secrets)
def evaluate_mmlu(model_id: str, limit: int | None = None) -> dict:
    import re

    from vllm import LLM, SamplingParams

    from datasets import load_dataset

    dataset = load_dataset("cais/mmlu", "all", split="test")
    if limit is not None and limit > 0:
        dataset = dataset.shuffle(seed=42).select(range(min(limit, len(dataset))))
    print(f"MMLU: {len(dataset)} samples", flush=True)

    llm = LLM(model=model_id, gpu_memory_utilization=0.9, trust_remote_code=True)
    tok = llm.get_tokenizer()

    system = (
        "Answer the following multiple choice question with the single letter "
        "of the correct option (A, B, C, or D). No explanation."
    )
    if "qwen" in model_id.lower() and "base" not in model_id.lower():
        system = "/no_think\n" + system

    letters = ["A", "B", "C", "D"]
    prompts = []
    for ex in dataset:
        choices = "\n".join(f"{letters[i]}) {c}" for i, c in enumerate(ex["choices"]))
        user = f"{ex['question']}\n\n{choices}\n\nAnswer:"
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    outputs = llm.generate(prompts, SamplingParams(max_tokens=64, temperature=0.0))

    correct = 0
    for out, ex in zip(outputs, dataset, strict=True):
        text = out.outputs[0].text
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        m = re.search(r"\b([A-D])\b", text.upper())
        if m and (ord(m.group(1)) - ord("A")) == ex["answer"]:
            correct += 1

    return {"accuracy": correct / len(dataset), "correct": correct, "total": len(dataset)}


@app.local_entrypoint()
def main(models: str):
    out_dir = Path("/tmp/baselines")
    out_dir.mkdir(exist_ok=True)
    for model_id in models.split(","):
        slug = model_id.replace("/", "_")
        out = out_dir / f"mmlu-{slug}.json"
        print(f"=== {model_id} ===")
        result = evaluate_mmlu.remote(model_id)
        out.write_text(json.dumps(result, indent=2))
        print(f"  accuracy={result['accuracy']:.4f} → {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    with modal.enable_output(), app.run():
        main(",".join(sys.argv[1:]))
