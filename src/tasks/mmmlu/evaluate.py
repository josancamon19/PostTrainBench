#!/usr/bin/env python3
"""OpenAI MMMLU eval for the verifier (GPU / vLLM path).

Direct vLLM offline inference + per-sample MCQ letter match across all 14
MMMLU languages. Runs the full test set by default; pass --limit N for a
balanced N-sample dev slice.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

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
SYSTEM = "Answer the following multiple-choice question with the single letter of the correct option (A, B, C, or D). No explanation."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run OpenAI MMMLU via vLLM.")
    p.add_argument("--model-path", type=str, default="final_model")
    p.add_argument(
        "--limit", type=int, default=-1, help="Total samples (balanced across languages). -1 = full test set."
    )
    p.add_argument("--json-output-file", type=str, default=None)
    p.add_argument("--templates-dir", type=str, default="templates/")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return p.parse_args()


def _model_type(path: str) -> str:
    low = path.lower()
    for fam in ("qwen", "llama", "gemma", "smollm"):
        if fam in low:
            return fam
    with open(os.path.join(path, "config.json")) as f:
        arch = json.load(f)["architectures"][0].lower()
    for fam in ("gemma", "llama", "qwen", "smollm"):
        if fam in arch:
            return fam
    raise ValueError(arch)


def _chat_template(templates_dir: str, path: str) -> str:
    fam = _model_type(path)
    tpl = {"qwen": "qwen3.jinja", "llama": "llama3.jinja", "gemma": "gemma3.jinja", "smollm": "smollm.jinja"}[fam]
    return (Path(templates_dir) / tpl).read_text()


def main() -> None:
    args = parse_args()

    from vllm import LLM, SamplingParams

    from datasets import concatenate_datasets, load_dataset

    per_lang = max(1, args.limit // len(LANG_CONFIGS)) if args.limit and args.limit > 0 else None
    parts = []
    for cfg in LANG_CONFIGS:
        ds = load_dataset("openai/MMMLU", cfg, split="test")
        if per_lang:
            ds = ds.shuffle(seed=42).select(range(min(per_lang, len(ds))))
        parts.append(ds)
    dataset = concatenate_datasets(parts)
    print(f"MMMLU: {len(dataset)} samples across {len(LANG_CONFIGS)} languages", flush=True)

    llm = LLM(model=args.model_path, gpu_memory_utilization=args.gpu_memory_utilization)
    tokenizer = llm.get_tokenizer()
    chat_template = _chat_template(args.templates_dir, args.model_path)

    prompts = []
    for ex in dataset:
        choices = "\n".join(f"{c}) {ex[c]}" for c in "ABCD")
        user = f"{ex['Question']}\n\n{choices}\n\nAnswer:"
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        prompts.append(
            tokenizer.apply_chat_template(msgs, chat_template=chat_template, add_generation_prompt=True, tokenize=False)
        )

    outputs = llm.generate(prompts, SamplingParams(max_tokens=args.max_tokens, temperature=0.0))

    correct = 0
    for out, ex in zip(outputs, dataset, strict=True):
        text = out.outputs[0].text
        m = re.search(r"\b([A-D])\b", text.upper())
        if m and m.group(1) == ex["Answer"].upper():
            correct += 1

    accuracy = correct / len(dataset)
    metrics = {"accuracy": accuracy, "correct": correct, "total": len(dataset)}
    print(json.dumps(metrics, indent=2))
    if args.json_output_file:
        Path(args.json_output_file).write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
