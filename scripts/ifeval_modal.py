#!/usr/bin/env python3
"""Measure IFEval accuracy on Modal GPU.

Use for instruct models Tinker doesn't serve (Llama-3.2-{1B,3B}-Instruct).
Full eval (541 prompts) by default; writes /tmp/baselines/ifeval-<slug>.json.

Usage:
    python scripts/ifeval_modal.py meta-llama/Llama-3.2-1B-Instruct
    python scripts/ifeval_modal.py m1,m2,m3   # sequential
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
    "if-verifiable",
)

app = modal.App("ifeval-baselines")
_secrets = [modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})] if os.environ.get("HF_TOKEN") else []


@app.function(image=IMAGE, gpu="H100", timeout=3600, secrets=_secrets)
def evaluate_ifeval(model_id: str, limit: int | None = None) -> dict:
    from if_verifiable import evaluate_output_for_sample, get_eval_data
    from vllm import LLM, SamplingParams

    samples = list(get_eval_data("ifeval"))
    if limit is not None and limit > 0:
        samples = samples[: min(limit, len(samples))]
    print(f"IFEval: {len(samples)} prompts", flush=True)

    llm = LLM(model=model_id, gpu_memory_utilization=0.9, trust_remote_code=True)
    tok = llm.get_tokenizer()

    prompts = []
    for s in samples:
        msgs = [{"role": "user", "content": s.prompt}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    outputs = llm.generate(prompts, SamplingParams(max_tokens=1024, temperature=0.0))

    agg = {"binary_strict": 0.0, "binary_loose": 0.0, "partial_strict": 0.0, "partial_loose": 0.0}
    total, errors = 0, 0
    for out, s in zip(outputs, samples, strict=True):
        try:
            text = out.outputs[0].text
            if "</think>" in text:
                text = text.split("</think>", 1)[1]
            _, scores = evaluate_output_for_sample("ifeval", s, text.strip())
            agg["binary_strict"] += scores.binary_strict
            agg["binary_loose"] += scores.binary_loose
            agg["partial_strict"] += scores.partial_strict
            agg["partial_loose"] += scores.partial_loose
            total += 1
        except Exception as e:
            errors += 1
            print(f"  error: {e}", file=sys.stderr)

    denom = max(total, 1)
    metrics = {k: round(v / denom, 6) for k, v in agg.items()}
    metrics["accuracy"] = metrics["binary_strict"]
    metrics["samples_scored"] = total
    metrics["samples_errored"] = errors
    return metrics


@app.local_entrypoint()
def main(models: str):
    out_dir = Path("/tmp/baselines")
    out_dir.mkdir(exist_ok=True)
    for model_id in models.split(","):
        slug = model_id.replace("/", "_")
        out = out_dir / f"ifeval-{slug}.json"
        print(f"=== {model_id} ===")
        result = evaluate_ifeval.remote(model_id)
        out.write_text(json.dumps(result, indent=2))
        print(f"  accuracy={result['accuracy']:.4f} → {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    with modal.enable_output(), app.run():
        main(",".join(sys.argv[1:]))
