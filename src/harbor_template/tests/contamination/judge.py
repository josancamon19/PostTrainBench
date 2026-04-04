#!/usr/bin/env python3
"""
Generate the contamination judge prompt for use with codex CLI.

The actual judging is done by codex CLI in test.sh.
"""

import argparse
import sys
from pathlib import Path

JUDGE_PROMPT_TEMPLATE = (Path(__file__).parent / "prompt.txt").read_text()


def get_other_allowed_data(benchmark: str) -> str:
    """Get benchmark-specific allowed data clauses."""
    if "gsm8k" in benchmark.lower():
        return "- Usage of the training subset of GSM8K for training.\n"
    return ""


def build_prompt(model: str, benchmark: str) -> str:
    """Build the full judge prompt with placeholders filled in."""
    other_allowed = get_other_allowed_data(benchmark)
    return JUDGE_PROMPT_TEMPLATE.format(
        model=model,
        benchmark=benchmark,
        other_allowed_data=other_allowed,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate the contamination judge prompt for codex CLI")
    parser.add_argument("--model", type=str, required=True, help="Expected base model ID")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name")
    args = parser.parse_args()

    prompt = build_prompt(args.model, args.benchmark)
    print(prompt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
