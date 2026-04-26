#!/usr/bin/env python3
"""Re-run the regression suite against a completed trial's final model.

Dispatches on mode (read from artifacts/workspace/metadata.json):
  - tinker: reuses tests/evals/<id>/evaluate_tinker.py with the trial's
    checkpoint URI (from artifacts/workspace/best_checkpoint.txt).
  - gpu / gpu-runpod: pulls the model from HuggingFace (URL from
    verifier/final_model_hf.txt) and runs evaluate.py inside a Modal H100
    container, one eval per container call for fault isolation.

Writes to <trial-dir>/verifier/ — regression_metrics.json,
regression_<id>_metrics.json, regression_<id>.txt — so the output
is indistinguishable from what the verifier would have produced on
a live trial.

By default re-runs only evals whose status != "ok". Use --force to redo
all of them.

Usage:
    python scripts/_1_rerun_regressions.py <trial-or-job-dir>

Auth: TINKER_API_KEY (tinker), HF_TOKEN/HF_TOKEN_WRITE (gpu Modal pull).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from constants import BASE_SCORES, REGRESSION_BASE_SCORES, REGRESSION_EVALS  # noqa: E402

# ---------------------------------------------------------------------------
# Modal app / image / function definitions for GPU-mode regression rerun.
# Defined at module level so we don't need serialized=True (which would force
# a Python version match between local and image). Using the image's native
# Python 3.10 means all eval deps (vllm, inspect-evals, if-verifiable, etc.)
# are already installed via the GHCR base image.
# ---------------------------------------------------------------------------
try:
    import modal as _modal

    _HAS_MODAL = True
except ImportError:
    _modal = None
    _HAS_MODAL = False

if _HAS_MODAL:
    # Modal's `from_registry` runs orchestration in Python 3.10 (Ubuntu base),
    # but the GHCR image installs vllm/inspect-evals/if-verifiable etc. into
    # Python 3.11 (via deadsnakes). We invoke /usr/bin/python3.11 explicitly in
    # the eval subprocess to use those preinstalled deps. No pip_install needed.
    _GPU_IMAGE = (
        _modal.Image.from_registry("ghcr.io/josancamon19/posttrainbench-gpu:latest")
        .add_local_dir(str(REPO_ROOT / "src" / "tasks"), remote_path="/eval_src")
        .add_local_dir(
            str(REPO_ROOT / "src" / "harbor_template" / "environment" / "templates"),
            remote_path="/templates",
        )
    )
    _aux_evals = REPO_ROOT / "src" / "harbor_template" / "tests" / "evals"
    if _aux_evals.exists():
        _GPU_IMAGE = _GPU_IMAGE.add_local_dir(str(_aux_evals), remote_path="/regression_only_evals")

    _GPU_APP = _modal.App("ptb-regression-rerun")

    _GPU_SECRETS = []
    for _k in ("HF_TOKEN", "HF_TOKEN_WRITE", "OPENAI_API_KEY"):
        if os.environ.get(_k):
            _GPU_SECRETS.append(_modal.Secret.from_dict({_k: os.environ[_k]}))

    @_GPU_APP.function(image=_GPU_IMAGE, gpu="H100", timeout=2400, secrets=_GPU_SECRETS)
    def _run_regression_eval_remote(eval_id: str, hf_repo: str, limit: int | None) -> dict:
        """Inside Modal: pull model from HF, run evaluate.py, return result."""
        import json as _j
        import os as _o
        import subprocess as _sp
        import time as _t
        from pathlib import Path as _P

        from huggingface_hub import snapshot_download

        eval_dir = _P("/regression_only_evals") / eval_id
        if not (eval_dir / "evaluate.py").exists():
            eval_dir = _P("/eval_src") / eval_id
        if not (eval_dir / "evaluate.py").exists():
            return {"status": "missing_evaluate", "score": None}

        # The GHCR image installs all eval deps into /usr/bin/python3.11, so
        # invoke that explicitly. Modal's orchestration Python is 3.10 (Ubuntu
        # base) and doesn't see those packages.
        eval_python = "/usr/bin/python3.11"

        token = _o.environ.get("HF_TOKEN_WRITE") or _o.environ.get("HF_TOKEN")
        print(f"[modal] downloading {hf_repo} ...", flush=True)
        model_path = snapshot_download(hf_repo, local_dir="/tmp/model", token=token)

        out_metrics = _P("/tmp/metrics.json")
        cmd = [
            eval_python,
            str(eval_dir / "evaluate.py"),
            "--model-path",
            model_path,
            "--json-output-file",
            str(out_metrics),
            "--templates-dir",
            "/templates",
        ]
        if limit is not None:
            cmd.extend(["--limit", str(limit)])

        print(f"[modal] running: {' '.join(cmd)}", flush=True)
        start = _t.time()
        proc = _sp.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=2200,
            env={**_o.environ, "PYTHONPATH": str(eval_dir) + ":" + _o.environ.get("PYTHONPATH", "")},
        )
        elapsed = round(_t.time() - start, 1)

        metrics_obj = None
        if out_metrics.exists():
            try:
                metrics_obj = _j.loads(out_metrics.read_text())
            except Exception:
                metrics_obj = None

        score = None
        if metrics_obj:
            for k in ("accuracy", "pass@1", "score", "exact_match", "mean", "pct"):
                v = metrics_obj.get(k)
                if isinstance(v, (int, float)):
                    score = float(v)
                    break

        status = "ok" if proc.returncode == 0 and score is not None else f"exit_{proc.returncode}"
        return {
            "status": status,
            "score": score,
            "seconds": elapsed,
            "stdout": (proc.stdout + proc.stderr)[-50_000:],
            "metrics": metrics_obj,
        }


def _baselines_for(model_id: str, reg_ids: list[str]) -> dict[str, float | None]:
    """Fallback when metadata doesn't carry regression_baselines (older trials
    predate the regression suite). REGRESSION_BASE_SCORES first, BASE_SCORES
    fallback — same precedence the adapter uses at export time."""
    out: dict[str, float | None] = {}
    for reg_id in reg_ids:
        v = REGRESSION_BASE_SCORES.get((model_id, reg_id))
        if v is None:
            v = BASE_SCORES.get((model_id, reg_id))
        out[reg_id] = v
    return out


EVALS_DIR = REPO_ROOT / "src" / "harbor_template" / "tests" / "evals"
TASKS_DIR = REPO_ROOT / "src" / "tasks"


def _eval_tinker_script(eval_id: str) -> Path | None:
    """Regression-only evals (mmlu/ifeval/truthfulqa) live under tests/evals/;
    training-target evals (gsm8k/humaneval/…) have evaluate_tinker.py in
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


_LIMIT_OVERRIDES: dict[str, int | None] = {
    "mmlu": 500,
    "ifeval": 300,
    "truthfulqa": 300,
    "gsm8k": 150,
    "humaneval": 150,
    "gpqamain": 150,
}


def _rerun_gpu_one(eval_id: str, hf_repo: str, out_json: Path, stdout_file: Path) -> dict:
    """Dispatch a single GPU regression eval via the module-level Modal app."""
    if not _HAS_MODAL:
        return {"status": "error", "score": None, "error": "modal not installed", "seconds": 0.0}
    limit = _LIMIT_OVERRIDES.get(eval_id)
    try:
        with _modal.enable_output(), _GPU_APP.run():
            result = _run_regression_eval_remote.remote(eval_id, hf_repo, limit)
    except Exception as e:
        return {"status": "error", "score": None, "error": str(e), "seconds": 0.0}

    stdout_file.write_text(result.get("stdout") or "")
    if result.get("metrics") is not None:
        out_json.write_text(json.dumps(result["metrics"], indent=2))
    return {"status": result.get("status"), "score": result.get("score"), "seconds": result.get("seconds")}


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
    start = time.time()
    try:
        with stdout_file.open("w") as f:
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, timeout=3600)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "score": None, "seconds": round(time.time() - start, 1)}
    except Exception as e:
        return {"status": "error", "score": None, "error": str(e), "seconds": round(time.time() - start, 1)}
    elapsed = round(time.time() - start, 1)
    score = _read_metric(out_json)
    status = "ok" if proc.returncode == 0 and score is not None else f"exit_{proc.returncode}"
    return {"status": status, "score": score, "seconds": elapsed}


def _evals_to_retry(out_metrics: Path, reg_ids: list[str], force: bool) -> list[str]:
    """If --force or no prior metrics: retry all. Otherwise only ones that errored."""
    if force or not out_metrics.exists():
        return list(reg_ids)
    try:
        prior = json.loads(out_metrics.read_text())
    except Exception:
        return list(reg_ids)
    prior_evals = prior.get("evals", {})
    return [r for r in reg_ids if (prior_evals.get(r) or {}).get("status") != "ok"]


def _merge_and_write_metrics(out_metrics: Path, results: dict[str, dict], baselines: dict[str, float | None]) -> None:
    prior = {}
    if out_metrics.exists():
        try:
            prior = json.loads(out_metrics.read_text()).get("evals", {}) or {}
        except Exception:
            prior = {}
    merged = {**prior, **results}
    penalties = [
        max((r["baseline"] - r["score"]) / r["baseline"], 0.0)
        for r in merged.values()
        if r.get("score") is not None and r.get("baseline")
    ]
    out = {
        "evals": merged,
        "forgetting_penalty_mean": (sum(penalties) / len(penalties)) if penalties else 0.0,
        "evals_with_baseline_count": len(penalties),
    }
    out_metrics.write_text(json.dumps(out, indent=2))
    print(f"  forgetting_penalty_mean={out['forgetting_penalty_mean']:.3f} ({len(penalties)} baseline-evals)")


def _resolve_metadata(trial_dir: Path) -> dict | None:
    """metadata.json lives at /tests/metadata.json in the pod (verifier-only),
    so it's not in artifacts/workspace/. Resolve from the task export instead.
    """
    # Preferred: artifacts/workspace/ (older trials wrote it there)
    p = trial_dir / "artifacts" / "workspace" / "metadata.json"
    if p.exists():
        return json.loads(p.read_text())
    # Fallback: read result.json's config.task.path → task_export/tests/metadata.json
    result_path = trial_dir / "result.json"
    if not result_path.exists():
        return None
    try:
        result = json.loads(result_path.read_text())
        task_path = result.get("config", {}).get("task", {}).get("path")
        if task_path:
            md = REPO_ROOT / task_path / "tests" / "metadata.json"
            if md.exists():
                return json.loads(md.read_text())
    except Exception:
        pass
    return None


def rerun_trial(trial_dir: Path, force: bool = False) -> bool:
    meta = _resolve_metadata(trial_dir)
    if meta is None:
        print(f"  {trial_dir.name}: skipped (could not resolve metadata.json — task export may be missing)")
        return False
    out_dir = trial_dir / "verifier"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_metrics = out_dir / "regression_metrics.json"
    mode = meta.get("mode")
    benchmark_id = meta.get("benchmark_id")
    model_id = meta.get("model_id")
    reg_ids = [b for b in REGRESSION_EVALS if b != benchmark_id]
    baselines = meta.get("regression_baselines") or _baselines_for(model_id, reg_ids)
    todo = _evals_to_retry(out_metrics, reg_ids, force)
    if not todo:
        print(f"  {trial_dir.name}: all evals already ok (use --force to redo)")
        return True

    print(f"\n=== {trial_dir.name} (mode={mode}, model={model_id}, retrying {len(todo)}/{len(reg_ids)}) ===")

    if mode == "tinker":
        cp_file = trial_dir / "artifacts" / "workspace" / "best_checkpoint.txt"
        if not cp_file.exists():
            print("  missing best_checkpoint.txt — skipping")
            return False
        checkpoint = cp_file.read_text().strip()
        results: dict[str, dict] = {}
        for reg_id in todo:
            out_json = out_dir / f"regression_{reg_id}_metrics.json"
            stdout_file = out_dir / f"regression_{reg_id}.txt"
            print(f"  → {reg_id}…", flush=True)
            res = _rerun_tinker_one(reg_id, checkpoint, model_id, out_json, stdout_file)
            baseline = baselines.get(reg_id)
            res["baseline"] = baseline
            res["delta"] = (res["score"] - baseline) if (res["score"] is not None and baseline is not None) else None
            results[reg_id] = res
            print(f"     {res}")
        _merge_and_write_metrics(out_metrics, results, baselines)
        return True

    if mode in ("gpu", "gpu-runpod"):
        hf_log = trial_dir / "verifier" / "final_model_hf.txt"
        if not hf_log.exists() or not hf_log.read_text().startswith("ok:"):
            print(
                f"  {trial_dir.name}: cannot rerun GPU regressions — verifier/final_model_hf.txt missing or not 'ok:'"
            )
            print("  (model needs to be on HF; either re-run hf_upload.py or pass a local final_model/.)")
            return False
        hf_url = hf_log.read_text().strip().removeprefix("ok: ").strip()
        # URL form: https://huggingface.co/<user>/<repo>
        hf_repo = hf_url.removeprefix("https://huggingface.co/").strip("/")
        if not _HAS_MODAL:
            print("  modal not installed — `uv pip install modal` then `modal token new` to set up.")
            return False
        results = {}
        for reg_id in todo:
            out_json = out_dir / f"regression_{reg_id}_metrics.json"
            stdout_file = out_dir / f"regression_{reg_id}.txt"
            print(f"  → {reg_id} (Modal H100, hf={hf_repo})…", flush=True)
            res = _rerun_gpu_one(reg_id, hf_repo, out_json, stdout_file)
            baseline = baselines.get(reg_id)
            res["baseline"] = baseline
            res["delta"] = (res["score"] - baseline) if (res["score"] is not None and baseline is not None) else None
            results[reg_id] = res
            print(f"     {res}")
        _merge_and_write_metrics(out_metrics, results, baselines)
        return True

    print(f"  unknown mode: {mode}")
    return False


def _is_trial(p: Path) -> bool:
    return (p / "agent" / "trajectory.json").exists() or (p / "trial.log").exists()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="Trial dir or job dir")
    p.add_argument("--workers", type=int, default=1, help="Parallel trial count (Tinker handles the fanout)")
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run all evals even if their prior status was ok. Default: only retry failed evals.",
    )
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
                if rerun_trial(t, force=args.force):
                    ok += 1
            except Exception as e:
                print(f"  ERROR on {t.name}: {e}", file=sys.stderr)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(rerun_trial, t, args.force): t for t in trials}
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
