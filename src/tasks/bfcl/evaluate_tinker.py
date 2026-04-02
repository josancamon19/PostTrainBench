#!/usr/bin/env python3
"""Evaluate a Tinker checkpoint (or base model) on BFCL (Berkeley Function Calling Leaderboard).

NOTE: BFCL requires structured tool calling which is hard to evaluate with raw generation.
This implementation uses a simplified approach — prompting the model with function schemas
and checking if the generated function call matches the expected one.
"""

from __future__ import annotations

import json
import re
import sys

sys.path.insert(0, sys.path[0])
sys.path.insert(0, sys.path[0] + "/../..")
from tinker_util import parse_args, setup_tinker, batch_evaluate, save_metrics

from datasets import load_dataset

MAX_TOKENS = 16000
SYSTEM = (
    "You are a helpful assistant with access to functions. When the user asks you to do something, "
    'respond with a function call in JSON format: {"name": "function_name", "arguments": {...}}. '
    "Only output the JSON function call, nothing else."
)


def prepare_dataset(limit: int | None = None):
    """Load BFCL simple function calling subset."""
    ds = load_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard", split="test")
    examples = []
    for row in ds:
        # BFCL has multiple categories; focus on simple function calls
        try:
            question = json.loads(row["question"]) if isinstance(row["question"], str) else row["question"]
            functions = json.loads(row["function"]) if isinstance(row["function"], str) else row["function"]
            ground_truth = (
                json.loads(row["ground_truth"]) if isinstance(row["ground_truth"], str) else row["ground_truth"]
            )
        except (json.JSONDecodeError, TypeError):
            continue

        if not question or not functions or not ground_truth:
            continue

        # Build a user message with function schemas
        func_schemas = (
            json.dumps(functions, indent=2) if isinstance(functions, list) else json.dumps([functions], indent=2)
        )
        user_msg = question[0]["content"] if isinstance(question, list) and question else str(question)

        examples.append(
            {
                "user_message": user_msg,
                "function_schemas": func_schemas,
                "ground_truth": ground_truth,
            }
        )
    if limit:
        examples = examples[:limit]
    return examples


def build_messages(example):
    return [
        {"role": "system", "content": SYSTEM + f"\n\nAvailable functions:\n{example['function_schemas']}"},
        {"role": "user", "content": example["user_message"]},
    ]


def score(content: str, example: dict) -> bool:
    """Check if generated function call matches ground truth (name match)."""
    try:
        # Try to parse the model output as JSON
        # Strip markdown fences if present
        content = re.sub(r"```json?\s*", "", content)
        content = re.sub(r"```\s*", "", content)
        generated = json.loads(content.strip())
        gt = example["ground_truth"]
        if isinstance(gt, list):
            gt = gt[0] if gt else {}
        if isinstance(generated, list):
            generated = generated[0] if generated else {}
        # Check function name match at minimum
        gen_name = generated.get("name", "")
        gt_name = gt.get("name", "") if isinstance(gt, dict) else ""
        return gen_name == gt_name
    except (json.JSONDecodeError, TypeError, KeyError):
        return False


def main() -> None:
    args = parse_args("Evaluate a Tinker checkpoint on BFCL.")
    ctx = setup_tinker(args)
    dataset = prepare_dataset(limit=args.limit)
    metrics = batch_evaluate(ctx, dataset, build_messages, score, max_tokens=MAX_TOKENS)
    save_metrics(metrics, args.json_output_file)


if __name__ == "__main__":
    main()
