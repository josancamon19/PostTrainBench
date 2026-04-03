#!/usr/bin/env python3
"""Run baseline evaluations for all base + instruct models on all benchmarks using Modal GPUs.

Usage:
    python scripts/run_baselines.py                    # Run all missing baselines
    python scripts/run_baselines.py --model llama3.2-1b --benchmark gsm8k  # Run one pair
    python scripts/run_baselines.py --type instruct    # Only instruct models
    python scripts/run_baselines.py --dry-run          # Show what would run
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import modal

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
from constants import BENCHMARKS, BASE_SCORES, INSTRUCT_BASELINES

EVAL_MODELS = {
    "llama3.1-8b": {
        "base": "meta-llama/Llama-3.1-8B",
        "instruct": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "llama3.2-3b": {
        "base": "meta-llama/Llama-3.2-3B",
        "instruct": "meta-llama/Llama-3.2-3B-Instruct",
    },
    "llama3.2-1b": {
        "base": "meta-llama/Llama-3.2-1B",
        "instruct": "meta-llama/Llama-3.2-1B-Instruct",
    },
}

GPU_BENCHMARKS = list(BENCHMARKS.keys())
BASE_IMAGE = "ghcr.io/josancamon19/posttrainbench-gpu:latest"


def has_score(model_id: str, benchmark_id: str, model_type: str) -> bool:
    base_model_id = model_id.replace("-Instruct", "")
    if model_type == "base":
        return (base_model_id, benchmark_id) in BASE_SCORES
    return (base_model_id, benchmark_id) in INSTRUCT_BASELINES


def run_eval(model_key: str, model_type: str, benchmark_id: str, dry_run: bool = False) -> dict | None:
    model_id = EVAL_MODELS[model_key][model_type]
    label = f"{benchmark_id}/{model_id}"
    base_id = EVAL_MODELS[model_key]["base"]

    if has_score(base_id, benchmark_id, model_type):
        print(f"  SKIP {label} (already have score)")
        return None

    if dry_run:
        print(f"  WOULD RUN {label}")
        return None

    print(f"  RUNNING {label}...")

    task_dir = SRC_DIR / "tasks" / benchmark_id
    eval_script = task_dir / "evaluate.py"
    if not eval_script.exists():
        print(f"  SKIP {label} (no evaluate.py)")
        return None

    # Build the image with task files baked in
    image = modal.Image.from_registry(BASE_IMAGE)

    # HF_TOKEN for gated models
    hf_token = os.environ.get("HF_TOKEN", "")
    env_vars = {}
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token

    # OPENAI_API_KEY for judge-based benchmarks
    if benchmark_id in ("arenahardwriting", "healthbench"):
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if openai_key:
            env_vars["OPENAI_API_KEY"] = openai_key

    if env_vars:
        image = image.env(env_vars)

    image = image.add_local_file(str(eval_script), "/app/evaluate.py")

    # Copy evaluation_code if exists
    eval_code = task_dir / "evaluation_code"
    if eval_code.is_dir():
        image = image.add_local_dir(str(eval_code), "/app/evaluation_code")

    # Copy templates
    templates = SRC_DIR / "harbor_template" / "environment" / "templates"
    if templates.is_dir():
        image = image.add_local_dir(str(templates), "/app/templates")

    # Build command
    limit = "-1"
    if benchmark_id in ("arenahardwriting", "healthbench"):
        limit = "32"

    cmd = f"cd /app && python3 evaluate.py --model-path {model_id} --limit {limit} --json-output-file /tmp/metrics.json"

    try:
        app = modal.App.lookup("posttrainbench-baselines", create_if_missing=True)
        sb = modal.Sandbox.create(
            image=image,
            gpu="H100",
            timeout=3600,
            app=app,
        )

        process = sb.exec("bash", "-c", cmd)
        for line in process.stdout:
            print(f"    {line}", end="")
        process.wait()

        # Read metrics
        metrics_process = sb.exec("cat", "/tmp/metrics.json")
        output = ""
        for line in metrics_process.stdout:
            output += line
        metrics_process.wait()
        sb.terminate()

        metrics = json.loads(output)
        score = metrics.get("accuracy", 0)
        print(f"  DONE {label}: {score:.4f} ({score*100:.1f}%)")
        return {
            "model_key": model_key,
            "model_type": model_type,
            "model_id": model_id,
            "benchmark_id": benchmark_id,
            "score": score,
            "metrics": metrics,
        }

    except Exception as e:
        print(f"  ERROR {label}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluations on Modal GPUs")
    parser.add_argument("--model", type=str, default=None, help="Model key (e.g. llama3.2-1b)")
    parser.add_argument("--benchmark", type=str, default=None, help="Benchmark ID (e.g. gsm8k)")
    parser.add_argument("--type", type=str, default=None, choices=["base", "instruct"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=3, help="Max concurrent Modal sandboxes")
    parser.add_argument("--output", type=str, default="scripts/baseline_results.json")
    args = parser.parse_args()

    models = [args.model] if args.model else list(EVAL_MODELS.keys())
    benchmarks = [args.benchmark] if args.benchmark else GPU_BENCHMARKS
    types = [args.type] if args.type else ["base", "instruct"]

    jobs = []
    for model_key in models:
        for model_type in types:
            for benchmark_id in benchmarks:
                jobs.append((model_key, model_type, benchmark_id))

    # Filter out skips before parallelizing
    actual_jobs = []
    for model_key, model_type, benchmark_id in jobs:
        model_id = EVAL_MODELS[model_key][model_type]
        base_id = EVAL_MODELS[model_key]["base"]
        if has_score(base_id, benchmark_id, model_type):
            print(f"  SKIP {benchmark_id}/{model_id} (already have score)")
        else:
            actual_jobs.append((model_key, model_type, benchmark_id))

    print(f"\nRunning {len(actual_jobs)} evaluations (dry_run={args.dry_run}, workers={args.workers})...\n")

    results = []
    if args.dry_run:
        for model_key, model_type, benchmark_id in actual_jobs:
            model_id = EVAL_MODELS[model_key][model_type]
            print(f"  WOULD RUN {benchmark_id}/{model_id}")
    elif args.workers <= 1:
        for model_key, model_type, benchmark_id in actual_jobs:
            result = run_eval(model_key, model_type, benchmark_id)
            if result:
                results.append(result)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_eval, mk, mt, bid): (mk, mt, bid)
                for mk, mt, bid in actual_jobs
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

    if results:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        existing = []
        if output_path.exists():
            with open(output_path) as f:
                existing = json.load(f)

        existing.extend(results)
        with open(output_path, "w") as f:
            json.dump(existing, f, indent=2)

        print(f"\nResults saved to {output_path}")
        print(f"\n{'Model':<40} {'Benchmark':<20} {'Score':>10}")
        print("-" * 72)
        for r in results:
            print(f"{r['model_id']:<40} {r['benchmark_id']:<20} {r['score']*100:>9.1f}%")

        print("\nTo update constants.py, add these entries:")
        for r in results:
            base_id = r["model_id"].replace("-Instruct", "")
            dict_name = "INSTRUCT_BASELINES" if r["model_type"] == "instruct" else "BASE_SCORES"
            print(f'    ("{base_id}", "{r["benchmark_id"]}"): {r["score"]:.3f},')


if __name__ == "__main__":
    main()
