#!/usr/bin/env python3
"""Re-run the regression suite against a completed trial's final model.

Dispatches on mode (read from artifacts/workspace/metadata.json):
  - tinker: reuses tests/evals/<id>/evaluate_tinker.py with the trial's
    checkpoint URI (from artifacts/workspace/best_checkpoint.txt).
  - gpu / gpu-runpod: needs a GPU to run vLLM. Currently flagged as TODO —
    would require a Modal wrapper around tests/regression/suite.py.

Writes to <trial-dir>/verifier/ — regression_metrics.json,
regression_<id>_metrics.json, regression_<id>.txt — so the output
is indistinguishable from what the verifier would have produced on
a live trial.

Usage:
    python scripts/_5_rerun_regressions.py <trial-or-job-dir>

Auth: TINKER_API_KEY (tinker mode).
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from constants import REGRESSION_EVALS  # noqa: E402

EVALS_DIR = REPO_ROOT / "src" / "harbor_template" / "tests" / "evals"
TASKS_DIR = REPO_ROOT / "src" / "tasks"


def _eval_tinker_script(eval_id: str) -> Path | None:
    """Layer A evals (mmlu/ifeval/truthfulqa) live under tests/evals/; Layer B
    (gsm8k/humaneval/…) are training targets that have evaluate_tinker.py in
    src/tasks/ alongside the GPU evaluate.py."""
    for candidate in (EVALS_DIR / eval_id / "evaluate_tinker.py", TASKS_DIR / eval_id / "evaluate_tinker.py"):
        if candidate.exists():
            return candidate
    return None


def _read_metric(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    for k in ("accuracy", "pass@1", "score", "exact_match", "mean"):
        if k in d and isinstance(d[k], (int, float)):
            return float(d[k])
    return None


def _rerun_tinker_one(eval_id: str, checkpoint: str, base_model: str, out_json: Path, stdout_file: Path) -> dict:
    script = _eval_tinker_script(eval_id)
    if script is None:
        return {"status": "no_tinker_eval", "score": None}
    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        str(script),
        "--checkpoint",
        checkpoint,
        "--base-model",
        base_model,
        "--json-output-file",
        str(out_json),
    ]
    try:
        with stdout_file.open("w") as f:
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, timeout=3600)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "score": None}
    except Exception as e:
        return {"status": "error", "score": None, "error": str(e)}
    score = _read_metric(out_json)
    return {"status": "ok" if proc.returncode == 0 and score is not None else f"exit_{proc.returncode}", "score": score}


def rerun_trial(trial_dir: Path, skip_existing: bool = False) -> bool:
    meta_path = trial_dir / "artifacts" / "workspace" / "metadata.json"
    if not meta_path.exists():
        print(f"  {trial_dir.name}: skipped (no metadata.json)")
        return False
    out_dir = trial_dir / "verifier"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_metrics = out_dir / "regression_metrics.json"
    if skip_existing and out_metrics.exists():
        print(f"  {trial_dir.name}: skipped (already has regression_metrics.json)")
        return True
    meta = json.loads(meta_path.read_text())
    mode = meta.get("mode")
    baselines = meta.get("regression_baselines") or {}
    benchmark_id = meta.get("benchmark_id")
    model_id = meta.get("model_id")
    reg_ids = [b for b in REGRESSION_EVALS if b != benchmark_id]

    print(f"\n=== {trial_dir.name} (mode={mode}, model={model_id}) ===")

    if mode == "tinker":
        cp_file = trial_dir / "artifacts" / "workspace" / "best_checkpoint.txt"
        if not cp_file.exists():
            print("  missing best_checkpoint.txt — skipping")
            return False
        checkpoint = cp_file.read_text().strip()
        results: dict[str, dict] = {}
        for reg_id in reg_ids:
            out_json = out_dir / f"regression_{reg_id}_metrics.json"
            stdout_file = out_dir / f"regression_{reg_id}.txt"
            print(f"  → {reg_id}…", flush=True)
            res = _rerun_tinker_one(reg_id, checkpoint, model_id, out_json, stdout_file)
            baseline = baselines.get(reg_id)
            res["baseline"] = baseline
            res["delta"] = (res["score"] - baseline) if (res["score"] is not None and baseline is not None) else None
            results[reg_id] = res
            print(f"     {res}")
        penalties = [
            max((r["baseline"] - r["score"]) / r["baseline"], 0.0)
            for r in results.values()
            if r.get("score") is not None and r.get("baseline")
        ]
        out = {
            "evals": results,
            "forgetting_penalty_mean": (sum(penalties) / len(penalties)) if penalties else 0.0,
            "evals_with_baseline_count": len(penalties),
        }
        out_metrics.write_text(json.dumps(out, indent=2))
        print(f"  forgetting_penalty_mean={out['forgetting_penalty_mean']:.3f}")
        return True

    if mode in ("gpu", "gpu-runpod"):
        print("  GPU-mode rerun not implemented yet — needs Modal wrapper around tests/regression/suite.py.")
        print("  (requires downloading the HF-uploaded final_model + running vLLM).")
        return False

    print(f"  unknown mode: {mode}")
    return False


def _is_trial(p: Path) -> bool:
    return (p / "agent" / "trajectory.json").exists() or (p / "trial.log").exists()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="Trial dir or job dir")
    p.add_argument("--workers", type=int, default=1, help="Parallel trial count (Tinker handles the fanout)")
    p.add_argument("--skip-existing", action="store_true", help="Skip trials that already have regression_metrics.json")
    args = p.parse_args()
    path = args.path.resolve()
    trials = [path] if _is_trial(path) else sorted([c for c in path.iterdir() if c.is_dir() and _is_trial(c)])
    if not trials:
        print(f"no trials under {path}", file=sys.stderr)
        return 1

    ok = 0
    if args.workers <= 1:
        for t in trials:
            try:
                if rerun_trial(t, skip_existing=args.skip_existing):
                    ok += 1
            except Exception as e:
                print(f"  ERROR on {t.name}: {e}", file=sys.stderr)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(rerun_trial, t, args.skip_existing): t for t in trials}
            for fut in as_completed(futs):
                t = futs[fut]
                try:
                    if fut.result():
                        ok += 1
                except Exception as e:
                    print(f"  ERROR on {t.name}: {e}", file=sys.stderr)
    print(f"\nDone: {ok}/{len(trials)} reran.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
