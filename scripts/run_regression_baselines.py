#!/usr/bin/env python3
"""Measure regression-suite baselines (MMLU, TruthfulQA) for every Tinker-
compatible base model in constants.MODELS, via Tinker locally.

Each (model, eval) pair runs in a separate subprocess so we don't have to
juggle multiple training clients in one process. Results are written to
scripts/regression_baselines.json and also printed as a ready-to-paste
constants.py block.

Usage:
    python scripts/run_regression_baselines.py --limit 200
    python scripts/run_regression_baselines.py --model llama3.2-1b --eval mmlu
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
from constants import MODELS  # noqa: E402

EVAL_SCRIPTS: dict[str, Path] = {
    "mmlu": REPO_ROOT / "src/harbor_template/regressions/mmlu/evaluate_tinker.py",
    "truthfulqa": REPO_ROOT / "src/harbor_template/regressions/truthfulqa/evaluate_tinker.py",
}

OUT_JSON = REPO_ROOT / "scripts" / "regression_baselines.json"


def run_one(eval_id: str, base_model: str, limit: int) -> dict | None:
    script = EVAL_SCRIPTS[eval_id]
    tmp = Path(f"/tmp/ptb_baseline_{eval_id}_{base_model.replace('/', '_')}.json")
    tmp.unlink(missing_ok=True)
    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        str(script),
        "--base-model",
        base_model,
        "--json-output-file",
        str(tmp),
    ]
    if limit > 0:
        cmd.extend(["--limit", str(limit)])
    print(f"→ {base_model} / {eval_id} (limit={'full' if limit <= 0 else limit})", flush=True)
    try:
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: exit {e.returncode}", file=sys.stderr)
        return None
    if not tmp.exists():
        print("  ERROR: no metrics written", file=sys.stderr)
        return None
    return json.loads(tmp.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="0 or negative = full dataset")
    parser.add_argument("--model", type=str, default=None, help="One of MODELS keys; default = all tinker-compatible")
    parser.add_argument("--eval", type=str, default=None, help="One of 'mmlu'/'truthfulqa'; default = all")
    args = parser.parse_args()

    model_keys = [args.model] if args.model else [k for k, m in MODELS.items() if m.tinker]
    eval_ids = [args.eval] if args.eval else list(EVAL_SCRIPTS.keys())
    for eid in eval_ids:
        if eid not in EVAL_SCRIPTS:
            print(f"unknown eval: {eid}", file=sys.stderr)
            return 1

    existing: dict[str, dict[str, float]] = {}
    if OUT_JSON.exists():
        existing = json.loads(OUT_JSON.read_text())

    results: dict[str, dict[str, float]] = dict(existing)
    for mk in model_keys:
        info = MODELS.get(mk)
        if info is None:
            print(f"skip unknown model key: {mk}", file=sys.stderr)
            continue
        results.setdefault(info.model_id, {})
        for eid in eval_ids:
            metrics = run_one(eid, info.model_id, args.limit)
            if metrics is None:
                continue
            acc = metrics.get("accuracy")
            print(f"  {info.model_id} / {eid}: accuracy={acc}")
            results[info.model_id][eid] = acc
            OUT_JSON.write_text(json.dumps(results, indent=2))

    print("\n=== Paste into constants.py ===\n")
    print("REGRESSION_BASE_SCORES: dict[tuple[str, str], float] = {")
    for model_id in sorted(results):
        for eval_id in sorted(results[model_id]):
            v = results[model_id][eval_id]
            if v is None:
                continue
            print(f'    ("{model_id}", "{eval_id}"): {v:.3f},')
    print("}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
