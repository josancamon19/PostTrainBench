#!/usr/bin/env python3
"""Measure full-eval ifeval + mmlu baselines on Tinker for Llama models.

Covers 4 models × 2 benchmarks = 8 runs. Skips pairs whose output JSON already
exists, so it's resumable.

Usage:
    python scripts/baselines_tinker.py            # run all missing pairs
    python scripts/baselines_tinker.py --force    # re-run even if output exists
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT_DIR = Path("/tmp/baselines")

# (model_id, [benchmark_ids])
PAIRS = [
    ("meta-llama/Llama-3.1-8B", ["mmlu", "ifeval"]),
    ("meta-llama/Llama-3.1-8B-Instruct", ["mmlu", "ifeval"]),
    ("meta-llama/Llama-3.2-3B", ["mmlu", "ifeval"]),
    ("meta-llama/Llama-3.2-1B", ["mmlu", "ifeval"]),
]


def run_pair(model_id: str, bench: str, force: bool) -> dict | None:
    slug = model_id.replace("/", "_")
    out = OUT_DIR / f"{bench}-{slug}.json"
    eval_py = ROOT / "src" / "tasks" / bench / "evaluate_tinker.py"

    if out.exists() and not force:
        data = json.load(open(out))
        print(f"SKIP (exists): {out.name}  accuracy={data.get('accuracy', 0):.4f}")
        return data

    print(f"=== {bench} × {model_id} ===", flush=True)
    cmd = [
        sys.executable,
        str(eval_py),
        "--base-model",
        model_id,
        "--json-output-file",
        str(out),
    ]
    rc = subprocess.run(cmd).returncode
    if rc != 0 or not out.exists():
        print(f"  FAILED (rc={rc})")
        return None
    data = json.load(open(out))
    print(f"  accuracy={data.get('accuracy', 0):.4f}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-run even if output exists.")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    results: dict[tuple[str, str], float] = {}

    for model_id, benchmarks in PAIRS:
        for bench in benchmarks:
            data = run_pair(model_id, bench, args.force)
            if data is not None and "accuracy" in data:
                results[(model_id, bench)] = float(data["accuracy"])

    print("\n=== Summary (paste into constants.py SCORES) ===")
    for model_id, benchmarks in PAIRS:
        for bench in benchmarks:
            score = results.get((model_id, bench))
            tag = f"{score:.3f}" if score is not None else "MISSING"
            print(f"  ({model_id!r}, {bench!r}): {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
