#!/usr/bin/env python3
"""Evaluate a Tinker checkpoint (or base model) on HumanEval."""

from __future__ import annotations

import subprocess
import sys
import tempfile

sys.path.insert(0, sys.path[0])
sys.path.insert(0, sys.path[0] + "/../..")
from tinker_util import parse_args, setup_tinker, batch_evaluate, save_metrics

from datasets import load_dataset

MAX_TOKENS = 4000
SYSTEM = (
    "Complete the following Python function. Return ONLY the function body code, "
    "no explanation, no markdown fences. Continue directly from where the code left off."
)


def build_messages(example):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": example["prompt"]},
    ]


def run_tests(code: str, test: str, entry_point: str, timeout: int = 10) -> bool:
    """Execute generated code against test cases."""
    full_code = f"{code}\n{test}\ncheck({entry_point})"
    try:
        result = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


def make_score_fn(dataset_list):
    """Create a score function with access to test cases."""

    def score(content: str, example: dict) -> bool:
        # Combine prompt + completion
        full_code = example["prompt"] + content
        return run_tests(full_code, example["test"], example["entry_point"])

    return score


def main() -> None:
    args = parse_args("Evaluate a Tinker checkpoint on HumanEval.")
    ctx = setup_tinker(args)
    dataset = load_dataset("openai/openai_humaneval", split="test")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    dataset_list = list(dataset)
    metrics = batch_evaluate(ctx, dataset_list, build_messages, make_score_fn(dataset_list), max_tokens=MAX_TOKENS)
    save_metrics(metrics, args.json_output_file)


if __name__ == "__main__":
    main()
