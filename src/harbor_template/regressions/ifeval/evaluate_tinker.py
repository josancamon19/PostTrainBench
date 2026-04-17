#!/usr/bin/env python3
"""Evaluate a Tinker checkpoint (or base model) on google/IFEval.

Uses the scoring from Google's original instruction_following_eval source
(vendored under this directory). We report the standard 4 IFEval numbers
and write the most-conservative — prompt_strict — as `accuracy` so the
regression suite can slot it in without special-casing.

Usage:
    python evaluate_tinker.py --base-model "meta-llama/Llama-3.2-1B"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # src/
sys.path.insert(0, str(Path(__file__).parent.parent))  # for the relative ifeval package
import tinker

from datasets import load_dataset
from ifeval.simple_eval import evaluate_output, strip_thinking
from tinker_util import parse_args, setup_tinker

MAX_TOKENS = 1024


def main() -> None:
    args = parse_args("Evaluate a Tinker checkpoint on google/IFEval.")
    ctx = setup_tinker(args)

    dataset = load_dataset("google/IFEval", split="train")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    sampling_params = tinker.SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        stop=ctx.renderer.get_stop_sequences(),
    )

    # Fire all requests concurrently.
    futures = []
    prompts: list[dict] = []
    for ex in dataset:
        messages = [{"role": "user", "content": ex["prompt"]}]
        prompt = ctx.renderer.build_generation_prompt(messages)
        futures.append(ctx.sampling_client.sample(prompt=prompt, sampling_params=sampling_params, num_samples=1))
        prompts.append(ex)
    print(f"Fired {len(futures)} requests", flush=True)

    agg = {"prompt_strict": 0.0, "prompt_loose": 0.0, "instruction_strict": 0.0, "instruction_loose": 0.0}
    errors = 0
    total = 0
    for i, (fut, ex) in enumerate(zip(futures, prompts, strict=True)):
        try:
            result = fut.result()
            parsed, _ = ctx.renderer.parse_response(result.sequences[0].tokens)
            content = strip_thinking(parsed["content"]).replace("<|im_end|>", "").strip()
            _, scores = evaluate_output(content, ex["instruction_id_list"], ex["kwargs"], ex["prompt"])
            for k, v in scores.items():
                agg[k] += v
            total += 1
        except Exception as e:
            errors += 1
            print(f"  sample {i}: ERROR {e}", file=sys.stderr)
        if (i + 1) % 100 == 0 or (i + 1) == len(futures):
            print(
                f"  {i + 1}/{len(futures)} | prompt_strict={agg['prompt_strict'] / max(total, 1):.3f}",
                flush=True,
            )

    denom = max(total, 1)
    metrics = {k: round(v / denom, 6) for k, v in agg.items()}
    metrics["accuracy"] = metrics["prompt_strict"]  # used by regression_suite
    metrics["samples_scored"] = total
    metrics["samples_errored"] = errors
    print(json.dumps(metrics, indent=2))

    if args.json_output_file:
        with open(args.json_output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.json_output_file}")


if __name__ == "__main__":
    main()
