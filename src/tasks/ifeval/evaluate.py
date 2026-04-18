#!/usr/bin/env python3
"""IFEval regression eval for the verifier (GPU path).

Drops inspect-ai in favor of direct vLLM + if-verifiable scoring. Writes
metrics.json with `accuracy` = binary_strict (prompt-level all-pass).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run IFEval via vLLM + if-verifiable.")
    p.add_argument("--model-path", type=str, default="final_model")
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--json-output-file", type=str, default=None)
    p.add_argument("--templates-dir", type=str, default="templates/")
    p.add_argument("--max-connections", type=int, default=64)  # unused; vLLM offline
    p.add_argument("--max-tokens", type=int, default=2048)
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

    from if_verifiable import evaluate_output_for_sample, get_eval_data
    from vllm import LLM, SamplingParams

    samples = list(get_eval_data("ifeval"))
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    llm = LLM(model=args.model_path, gpu_memory_utilization=args.gpu_memory_utilization)
    tokenizer = llm.get_tokenizer()
    chat_template = _chat_template(args.templates_dir, args.model_path)

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": s.prompt}],
            chat_template=chat_template,
            add_generation_prompt=True,
            tokenize=False,
        )
        for s in samples
    ]
    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    outputs = llm.generate(prompts, params)

    agg = {"binary_strict": 0.0, "binary_loose": 0.0, "partial_strict": 0.0, "partial_loose": 0.0}
    for out, s in zip(outputs, samples, strict=True):
        response = out.outputs[0].text
        _, scores = evaluate_output_for_sample("ifeval", s, response)
        agg["binary_strict"] += scores.binary_strict
        agg["binary_loose"] += scores.binary_loose
        agg["partial_strict"] += scores.partial_strict
        agg["partial_loose"] += scores.partial_loose

    n = len(samples)
    metrics = {k: v / n for k, v in agg.items()}
    metrics["accuracy"] = metrics["binary_strict"]

    if args.json_output_file:
        Path(args.json_output_file).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
