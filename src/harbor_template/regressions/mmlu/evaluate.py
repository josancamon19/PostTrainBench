#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

import inspect_evals.mmlu  # noqa: F401  (registers task definitions)
from inspect_ai import eval as inspect_eval  # type: ignore
from inspect_ai.util._display import init_display_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MMLU eval (0-shot) via Inspect AI.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="final_model",
        help="Path to the Hugging Face model (directory or model identifier).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Optional limit for number of samples to evaluate (-1 for all).",
    )
    parser.add_argument(
        "--json-output-file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--templates-dir",
        type=str,
        default="templates/",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    init_display_type("plain")

    other_kwargs = {}
    if (args.limit is not None) and (args.limit != -1):
        other_kwargs["limit"] = args.limit

    task = "inspect_evals/mmlu_0_shot"
    model_args = {
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    model_args.update(template_kwargs(args))

    eval_out = inspect_eval(
        task,
        model=f"vllm/{args.model_path}",
        model_args=model_args,
        score_display=False,
        log_realtime=False,
        log_format="json",
        timeout=18000000,
        attempt_timeout=18000000,
        max_tokens=args.max_tokens,
        max_connections=args.max_connections,
        **other_kwargs,
    )

    if args.json_output_file is not None:
        assert len(eval_out) == 1, eval_out
        assert len(eval_out[0].results.scores) == 1, eval_out[0].results.scores
        metrics = {}
        for k, v in eval_out[0].results.scores[0].metrics.items():
            metrics[k] = v.value

        with open(args.json_output_file, "w") as f:
            json.dump(metrics, f, indent=2)


def model_type(args) -> str:
    if "qwen" in args.model_path.lower():
        return "qwen"
    if "llama" in args.model_path.lower():
        return "llama"
    if "gemma" in args.model_path.lower():
        return "gemma"
    if "smollm" in args.model_path.lower():
        return "smollm"

    with open(os.path.join(args.model_path, "config.json")) as f:
        config = json.load(f)
    architecture = config["architectures"][0].lower()
    if "gemma" in architecture:
        return "gemma"
    if "llama" in architecture:
        return "llama"
    if "qwen" in architecture:
        return "qwen"
    if "smollm" in architecture:
        return "smollm"
    raise ValueError(architecture)


def template_kwargs(args) -> dict:
    model_type_str = model_type(args)
    if model_type_str == "qwen":
        template = "qwen3.jinja"
    elif model_type_str == "llama":
        template = "llama3.jinja"
    elif model_type_str == "gemma":
        template = "gemma3.jinja"
    elif model_type_str == "smollm":
        template = "smollm.jinja"
    else:
        raise ValueError(model_type_str)
    return {"chat_template": os.path.join(args.templates_dir, template)}


if __name__ == "__main__":
    main()
