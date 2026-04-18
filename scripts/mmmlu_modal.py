#!/usr/bin/env python3
"""Measure MMMLU accuracy on Modal GPU.

Use for instruct models Tinker doesn't serve (Llama-3.2 instruct) or
serves without /no_think (Qwen3 instruct). Writes
/tmp/mmmlu_baselines/<slug>.json so results merge with Tinker runs.

Usage:
    python scripts/mmmlu_modal.py meta-llama/Llama-3.2-1B-Instruct
    python scripts/mmmlu_modal.py <m1> <m2> <m3>  # sequential
"""

import json
import os
import sys
from pathlib import Path

import modal

LANG_CONFIGS = [
    "AR_XY",
    "BN_BD",
    "DE_DE",
    "ES_LA",
    "FR_FR",
    "HI_IN",
    "ID_ID",
    "IT_IT",
    "JA_JP",
    "KO_KR",
    "PT_BR",
    "SW_KE",
    "YO_NG",
    "ZH_CN",
]

# Pin vllm + transformers to a known-good pair; newer transformers breaks vllm.
IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install(
    "vllm==0.6.6.post1",
    "transformers==4.46.3",
    "datasets>=2.0",
    "huggingface-hub>=0.25",
)

app = modal.App("mmmlu-baselines")
_secrets = [modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})] if os.environ.get("HF_TOKEN") else []


@app.function(image=IMAGE, gpu="H100", timeout=3600, secrets=_secrets)
def evaluate_mmmlu(model_id: str, per_lang: int = 100) -> dict:
    import re

    from vllm import LLM, SamplingParams

    from datasets import concatenate_datasets, load_dataset

    parts = []
    for cfg in LANG_CONFIGS:
        ds = load_dataset("openai/MMMLU", cfg, split="test").shuffle(seed=42)
        ds = ds.select(range(min(per_lang, len(ds))))
        parts.append(ds)
    dataset = concatenate_datasets(parts)
    print(f"MMMLU: {len(dataset)} samples", flush=True)

    llm = LLM(model=model_id, gpu_memory_utilization=0.9, trust_remote_code=True)
    tok = llm.get_tokenizer()

    system = (
        "Answer the following multiple-choice question with the single letter "
        "of the correct option (A, B, C, or D). No explanation."
    )
    if "qwen" in model_id.lower() and "base" not in model_id.lower():
        system = "/no_think\n" + system

    prompts = []
    for ex in dataset:
        choices = "\n".join(f"{c}) {ex[c]}" for c in "ABCD")
        user = f"{ex['Question']}\n\n{choices}\n\nAnswer:"
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    outputs = llm.generate(prompts, SamplingParams(max_tokens=64, temperature=0.0))

    correct = 0
    for out, ex in zip(outputs, dataset, strict=True):
        text = out.outputs[0].text
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        m = re.search(r"\b([A-D])\b", text.upper())
        if m and m.group(1) == ex["Answer"].upper():
            correct += 1

    return {"accuracy": correct / len(dataset), "correct": correct, "total": len(dataset)}


@app.local_entrypoint()
def main(models: str):
    out_dir = Path("/tmp/mmmlu_baselines")
    out_dir.mkdir(exist_ok=True)
    for model_id in models.split(","):
        slug = model_id.replace("/", "_")
        out = out_dir / f"{slug}.json"
        print(f"=== {model_id} ===")
        result = evaluate_mmmlu.remote(model_id)
        out.write_text(json.dumps(result, indent=2))
        print(f"  accuracy={result['accuracy']:.3f} → {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    with modal.enable_output(), app.run():
        main(",".join(sys.argv[1:]))
