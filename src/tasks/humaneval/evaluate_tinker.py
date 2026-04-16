#!/usr/bin/env python3
"""Evaluate a Tinker checkpoint (or base model) on HumanEval."""

from __future__ import annotations

import subprocess
import sys

sys.path.insert(0, sys.path[0])
sys.path.insert(0, sys.path[0] + "/../..")
from datasets import load_dataset
from tinker_util import batch_evaluate, parse_args, save_metrics, setup_tinker

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


def fix_indent(content: str, prompt: str) -> str:
    """Auto-indent response to match function body level if needed."""
    # Find first non-empty code line (skip thinking content)
    first_code = ""
    for line in content.split("\n"):
        if line.strip() and not line.strip().startswith(("```", "#")):
            first_code = line
            break

    # Already indented — leave as-is
    if not first_code or first_code.startswith("    "):
        return content

    # Needs indentation — add 4 spaces to each non-empty line
    return "\n".join("    " + line if line.strip() else line for line in content.split("\n"))


def make_score_fn(dataset_list):
    """Create a score function with access to test cases."""

    def score(content: str, example: dict) -> bool:
        full_code = example["prompt"] + content
        # If syntax error from bad indent, try with fix
        if not run_tests(full_code, example["test"], example["entry_point"]):
            fixed = example["prompt"] + fix_indent(content, example["prompt"])
            return run_tests(fixed, example["test"], example["entry_point"])
        return True

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
