#!/usr/bin/env python3
"""Evaluate a Tinker checkpoint (or base model) on HealthBench.

Usage:
    python evaluate.py --checkpoint "tinker://<run_id>/sampler_weights/final"
    python evaluate.py --base-model "Qwen/Qwen3-1.7B-Base"
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, sys.path[0])
sys.path.insert(0, sys.path[0] + "/../..")
from tinker_util import parse_args, setup_tinker, save_metrics

from tinker import types
from tinker_cookbook.renderers import get_text_content

from evaluation_code.data_loader import load_healthbench
from evaluation_code.grader import grade_examples_parallel
from evaluation_code.scoring import aggregate_scores
from evaluation_code.text_utils import limit_repetitions

MAX_TOKENS = 16384


def generate_answers_tinker(ctx, examples, max_tokens=MAX_TOKENS):
    """Generate responses for all examples using Tinker sampling."""
    sampling_params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        stop=ctx.renderer.get_stop_sequences(),
    )

    print(f"[generate] Generating answers for {len(examples)} examples via Tinker")

    # Fire all requests concurrently
    futures = []
    for example in examples:
        messages = example.conversation.copy()
        prompt = ctx.renderer.build_generation_prompt(messages)
        futures.append(ctx.sampling_client.sample(prompt=prompt, sampling_params=sampling_params, num_samples=1))

    print(f"[generate] Fired {len(futures)} requests...")

    responses = []
    for i, (future, example) in enumerate(zip(futures, examples)):
        try:
            result = future.result()
            parsed, _ = ctx.renderer.parse_response(result.sequences[0].tokens)
            text = get_text_content(parsed)

            # Strip thinking tags if present
            if text.startswith("<think>") and "</think>" in text:
                text = text.split("</think>", maxsplit=1)[-1].strip()

            text = limit_repetitions(text)
            responses.append(text)
        except Exception as e:
            print(f"[generate] Example {i} ({example.example_id}): ERROR - {e}", file=sys.stderr)
            responses.append("")

        if (i + 1) % 10 == 0 or (i + 1) == len(examples):
            print(f"[generate] Progress: {i + 1}/{len(examples)}", flush=True)

    return responses


def main() -> None:
    args = parse_args("Evaluate a Tinker checkpoint on HealthBench.")

    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    ctx = setup_tinker(args)

    # Load data
    examples = load_healthbench()
    random.Random(42).shuffle(examples)
    if args.limit:
        examples = examples[: min(args.limit, len(examples))]
    print(f"[data] Loaded {len(examples)} HealthBench examples")

    # Generate
    responses = generate_answers_tinker(ctx, examples)
    print(f"[generate] Generated {len(responses)} responses")

    # Grade
    print("[judge] Grading responses...")
    results = grade_examples_parallel(
        examples=examples,
        responses=responses,
        grader_model="gpt-4.1-mini",
        example_workers=4,
        criteria_workers=8,
        max_concurrent_requests=64,
    )

    # Score
    metrics = aggregate_scores(results, examples)
    print(f"\n[done] Accuracy: {metrics.accuracy:.4f} (±{metrics.stderr:.4f})")
    print(f"  Examples: {metrics.n_examples} | Grader calls: {metrics.total_grader_calls}")

    output = {
        "accuracy": metrics.accuracy,
        "stderr": metrics.stderr,
        "n_examples": metrics.n_examples,
        "total_grader_calls": metrics.total_grader_calls,
    }
    save_metrics(output, args.json_output_file)


if __name__ == "__main__":
    main()
