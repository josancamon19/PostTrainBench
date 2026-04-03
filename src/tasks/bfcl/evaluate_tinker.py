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
    """Load full BFCL v3 dataset (all subsets with ground truth answers)."""
    from huggingface_hub import hf_hub_download, HfApi

    api = HfApi()
    all_files = api.list_repo_files("gorilla-llm/Berkeley-Function-Calling-Leaderboard", repo_type="dataset")
    # Find all BFCL_v3 files that have corresponding ground truth
    q_files = sorted(
        f for f in all_files
        if f.startswith("BFCL_v3") and "possible_answer" not in f and f"possible_answer/{f}" in all_files
    )

    examples = []
    for q_file in q_files:
        q_path = hf_hub_download(
            repo_id="gorilla-llm/Berkeley-Function-Calling-Leaderboard",
            filename=q_file,
            repo_type="dataset",
        )
        gt_path = hf_hub_download(
            repo_id="gorilla-llm/Berkeley-Function-Calling-Leaderboard",
            filename=f"possible_answer/{q_file}",
            repo_type="dataset",
        )
        with open(q_path) as f:
            questions = {r["id"]: r for r in (json.loads(line) for line in f)}
        with open(gt_path) as f:
            answers = {r["id"]: r for r in (json.loads(line) for line in f)}

        for qid, row in questions.items():
            gt = answers.get(qid)
            if not gt:
                continue
            question = row.get("question", [])
            functions = row.get("function", [])
            ground_truth = gt.get("ground_truth", [])
            if not question or not functions or not ground_truth:
                continue

            func_schemas = (
                json.dumps(functions, indent=2) if isinstance(functions, list) else json.dumps([functions], indent=2)
            )
            q = question[0] if isinstance(question, list) else question
            if isinstance(q, list):
                q = q[0]
            user_msg = q["content"] if isinstance(q, dict) else str(q)

            examples.append(
                {
                    "user_message": user_msg,
                    "function_schemas": func_schemas,
                    "ground_truth": ground_truth,
                }
            )

    print(f"[data] Loaded {len(examples)} BFCL examples from {len(q_files)} subsets")
    if limit:
        examples = examples[:limit]
    return examples


def build_messages(example):
    return [
        {"role": "system", "content": SYSTEM + f"\n\nAvailable functions:\n{example['function_schemas']}"},
        {"role": "user", "content": example["user_message"]},
    ]


def score(content: str, example: dict) -> bool:
    """Check if generated function call matches ground truth (function name match).

    Ground truth format: [{"func_name": {"param": [possible_values], ...}}]
    Model output format: {"name": "func_name", "arguments": {...}}
    """
    try:
        # Strip markdown fences if present
        content = re.sub(r"```json?\s*", "", content)
        content = re.sub(r"```\s*", "", content)
        generated = json.loads(content.strip())
        if isinstance(generated, list):
            generated = generated[0] if generated else {}

        # Extract function name from model output ({"name": "...", "arguments": {...}})
        gen_name = generated.get("name", "")

        # Extract function name from ground truth ({"func_name": {"param": [...]}})
        gt = example["ground_truth"]
        if isinstance(gt, list):
            gt = gt[0] if gt else {}
        # GT keys are the function names
        gt_name = list(gt.keys())[0] if isinstance(gt, dict) and gt else ""

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
