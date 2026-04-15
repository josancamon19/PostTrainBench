#!/usr/bin/env python3
"""Reconstruct an agent's experiment history from a trial trajectory.

Two passes:
  1. Programmatic regex/keyword scan of agent/trajectory.json — fast, deterministic,
     emits extracted_events.json + progress.json (best-score-over-time).
  2. LLM reconstruction (Claude Agent SDK) — reads the trial dir, infers
     experiment_history.json + method_summary.json. Skipped with --no-llm
     or when ANTHROPIC_API_KEY is unset.

Usage:
    python scripts/mine_trajectory.py <trial-or-job-dir> [--no-llm] [--model haiku]

Outputs land in <trial-dir>/reconstructed/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Programmatic event extraction
# ---------------------------------------------------------------------------

# Regex bank — keep simple and obvious. Matches against Bash command strings
# and contents written via Write tool. Order matters: more-specific first.
_TRAIN_PATTERNS = [
    re.compile(r"\b(accelerate|torchrun|deepspeed)\s+launch\b", re.I),
    re.compile(r"python3?\s+\S*train[\w_-]*\.py\b", re.I),
    re.compile(r"python3?\s+\S*sft[\w_-]*\.py\b", re.I),
    re.compile(r"python3?\s+\S*dpo[\w_-]*\.py\b", re.I),
    re.compile(r"python3?\s+\S*finetune[\w_-]*\.py\b", re.I),
    re.compile(r"\b(SFTTrainer|DPOTrainer|GRPOTrainer|PPOTrainer|RLOOTrainer)\b"),
    re.compile(r"\bunsloth\b|\baxolotl\b"),
    re.compile(r"\btransformers\.Trainer\b"),
    re.compile(r"trainer\.train\(\)"),
]

_EVAL_PATTERNS = [
    re.compile(r"python3?\s+\S*evaluate[\w_-]*\.py\b", re.I),
    re.compile(r"\binspect\s+eval\b", re.I),
    re.compile(r"\blm[_-]eval\b", re.I),
]

_DATA_LOAD_PATTERNS = [
    re.compile(r'load_dataset\(\s*["\']([^"\']+)["\']'),
    re.compile(r"huggingface\.co/datasets/([\w./-]+)"),
    re.compile(r"\bdatasets/([\w./-]+)"),
    re.compile(r"\bhuggingface-cli\s+download\s+(\S+)"),
]

_PIP_INSTALL_PATTERN = re.compile(r"\b(?:uv\s+)?pip\s+install\b", re.I)

_METHOD_KEYWORDS = {
    "sft": [r"\bSFTTrainer\b", r"\bsft\.py\b", r"supervised[_-]?fine[_-]?tun", r"--method\s+sft"],
    "dpo": [r"\bDPOTrainer\b", r"\bdpo\.py\b", r"\bdirect[_-]?preference"],
    "ppo": [r"\bPPOTrainer\b", r"\bppo\.py\b"],
    "grpo": [r"\bGRPOTrainer\b", r"\bgrpo\.py\b"],
    "rloo": [r"\bRLOOTrainer\b"],
    "rejection_sampling": [r"reject(ion)?[_-]?sampl", r"best[_-]?of[_-]?n"],
    "lora": [r"\bLoraConfig\b", r"\bpeft\b", r"\bLoRA\b"],
    "qlora": [r"\bQLoRA\b|\bqlora\b", r"\b4bit\b|\b8bit\b.*bitsandbytes"],
    "distillation": [r"\bdistill", r"\bteacher[_-]?model"],
}

_HYPERPARAM_PATTERNS = {
    "learning_rate": re.compile(r"--learning[_-]?rate[\s=]+([\d.eE+-]+)|learning_rate\s*=\s*([\d.eE+-]+)"),
    "batch_size": re.compile(
        r"--(?:per[_-]device[_-]train[_-])?batch[_-]?size[\s=]+(\d+)|"
        r"per_device_train_batch_size\s*=\s*(\d+)|batch_size\s*=\s*(\d+)"
    ),
    "epochs": re.compile(r"--num[_-]?train[_-]?epochs[\s=]+([\d.]+)|num_train_epochs\s*=\s*([\d.]+)"),
    "max_steps": re.compile(r"--max[_-]?steps[\s=]+(\d+)|max_steps\s*=\s*(\d+)"),
    "lr_scheduler": re.compile(r"--lr[_-]?scheduler[_-]?type[\s=]+(\w+)|lr_scheduler_type\s*=\s*[\"']?(\w+)"),
}

# Score extraction from eval output — try common formats.
_SCORE_PATTERNS = [
    re.compile(r'"accuracy"\s*:\s*([\d.]+)'),
    re.compile(r'"pass@1"\s*:\s*([\d.]+)'),
    re.compile(r'"exact_match"\s*:\s*([\d.]+)'),
    re.compile(r'"score"\s*:\s*([\d.]+)'),
    re.compile(r"\baccuracy[:=\s]+([\d.]+)", re.I),
    re.compile(r"\bpass@1[:=\s]+([\d.]+)", re.I),
]


def _classify_bash(cmd: str) -> str:
    if any(p.search(cmd) for p in _TRAIN_PATTERNS):
        return "train"
    if any(p.search(cmd) for p in _EVAL_PATTERNS):
        return "eval"
    if any(p.search(cmd) for p in _DATA_LOAD_PATTERNS):
        return "data_load"
    if _PIP_INSTALL_PATTERN.search(cmd):
        return "pip_install"
    return "shell"


def _extract_methods(text: str) -> list[str]:
    found = []
    for method, patterns in _METHOD_KEYWORDS.items():
        if any(re.search(p, text, re.I) for p in patterns):
            found.append(method)
    return found


def _extract_hyperparams(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, pat in _HYPERPARAM_PATTERNS.items():
        m = pat.search(text)
        if m:
            val = next((g for g in m.groups() if g is not None), None)
            if val is not None:
                out[name] = val
    return out


def _extract_datasets(text: str) -> list[str]:
    found = set()
    for pat in _DATA_LOAD_PATTERNS:
        for m in pat.finditer(text):
            if m.groups():
                found.add(m.group(1))
    return sorted(found)


def _extract_scores(text: str) -> list[float]:
    scores = []
    for pat in _SCORE_PATTERNS:
        for m in pat.finditer(text):
            try:
                v = float(m.group(1))
                if 0 <= v <= 1.0:
                    scores.append(v)
            except (ValueError, IndexError):
                continue
    return scores


def _observation_text(obs: Any) -> str:
    if not isinstance(obs, dict):
        return ""
    parts: list[str] = []
    for r in obs.get("results", []) or []:
        c = r.get("content")
        if isinstance(c, str):
            parts.append(c)
    return "\n".join(parts)


def extract_events(trajectory: dict) -> list[dict]:
    """Walk the trajectory and emit a list of structured tool-call events."""
    events: list[dict] = []
    for step in trajectory.get("steps", []):
        tool_calls = step.get("tool_calls") or []
        if not tool_calls:
            continue
        ts = step.get("timestamp")
        step_id = step.get("step_id")
        obs_text = _observation_text(step.get("observation"))

        for tc in tool_calls:
            tool = tc.get("function_name") or "unknown"
            args = tc.get("arguments") or {}
            event: dict[str, Any] = {
                "step_id": step_id,
                "ts": ts,
                "tool": tool,
            }

            if tool == "Bash":
                cmd = args.get("command") or ""
                kind = _classify_bash(cmd)
                event["kind"] = kind
                event["command"] = cmd[:500]
                if kind == "train":
                    event["methods"] = _extract_methods(cmd)
                    hp = _extract_hyperparams(cmd)
                    if hp:
                        event["hyperparams"] = hp
                    ds = _extract_datasets(cmd)
                    if ds:
                        event["datasets"] = ds
                elif kind == "eval":
                    scores = _extract_scores(obs_text)
                    if scores:
                        event["score"] = max(scores)
                elif kind == "data_load":
                    ds = _extract_datasets(cmd + "\n" + obs_text)
                    if ds:
                        event["datasets"] = ds

            elif tool == "Write":
                path = args.get("file_path") or ""
                content = args.get("content") or ""
                event["kind"] = "script_write"
                event["path"] = path
                event["preview"] = content[:300]
                methods = _extract_methods(content[:5000])
                if methods:
                    event["methods"] = methods
                hp = _extract_hyperparams(content[:5000])
                if hp:
                    event["hyperparams"] = hp
                ds = _extract_datasets(content[:5000])
                if ds:
                    event["datasets"] = ds

            elif tool == "Edit":
                event["kind"] = "edit"
                event["path"] = args.get("file_path", "")

            else:
                event["kind"] = "other"

            events.append(event)
    return events


def build_progress(events: list[dict]) -> dict:
    """Best-score-over-time series from eval events."""
    points: list[dict] = []
    for e in events:
        if e.get("kind") != "eval" or "score" not in e:
            continue
        points.append({"ts": e["ts"], "step_id": e["step_id"], "score": e["score"]})
    points.sort(key=lambda p: p["ts"] or "")
    best = 0.0
    for p in points:
        best = max(best, p["score"])
        p["best_so_far"] = best
    return {"points": points, "final_best": best, "num_evals": len(points)}


def render_progress_plot(progress: dict, out_path: Path) -> None:
    """Render progress.png. Silently skips if matplotlib is unavailable or no points."""
    points = progress.get("points") or []
    if not points:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot", file=sys.stderr)
        return

    base = datetime.fromisoformat(points[0]["ts"].replace("Z", "+00:00"))
    xs = []
    ys_score = []
    ys_best = []
    for p in points:
        t = datetime.fromisoformat(p["ts"].replace("Z", "+00:00"))
        xs.append((t - base).total_seconds() / 3600)
        ys_score.append(p["score"])
        ys_best.append(p["best_so_far"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, ys_score, "o-", label="eval score", alpha=0.5, linewidth=1)
    ax.plot(xs, ys_best, "-", label="best so far", linewidth=2)
    ax.set_xlabel("Hours since first eval")
    ax.set_ylabel("Score")
    ax.set_title(f"Experiment progress ({len(points)} evals, final best {progress['final_best']:.3f})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# LLM reconstruction (optional)
# ---------------------------------------------------------------------------

_RECONSTRUCT_PROMPT = """\
You are reconstructing an agent's experiment history from its trajectory.

Working dir is the trial directory. Read these files using the Read/Glob/Grep tools:
  agent/trajectory.json — full agent action history
  reconstructed/extracted_events.json — pre-classified tool calls (use this first; it's small)
  artifacts/manifest.json — what artifacts the agent produced
  verifier/metrics.json, verifier/regression_metrics.json — final eval scores
  result.json — trial outcome

Goal: infer the agent's experiment journal — every distinct training attempt, what method
and dataset it used, hyperparameters, the eval score it achieved, and which checkpoint
was kept. Also infer overall method signatures.

The agent was NOT instructed to log experiments. Reconstruct from observed tool calls.
Prefer extracted_events.json over the raw trajectory — it has the relevant Bash/Write
events already parsed. Read trajectory.json only to disambiguate.

Return STRICTLY this JSON (no surrounding prose):

{
  "experiments": [
    {
      "ts": "ISO8601 timestamp of training kickoff",
      "method": "sft|dpo|grpo|ppo|rejection_sampling|distillation|other",
      "dataset": "HF dataset name or local path used for training",
      "hyperparams": {"learning_rate": "...", "batch_size": "...", ...},
      "eval_score": float-or-null,
      "checkpoint": "directory under /app/ where this run's weights landed, or null"
    }
  ],
  "method_summary": {
    "methods_used": ["sft", "dpo", ...],
    "num_experiments": int,
    "num_evals": int,
    "num_checkpoints": int,
    "final_method": "method that produced the submitted final_model",
    "datasets_used": ["..."],
    "narrative": "2-3 sentence summary of the agent's overall strategy and how it iterated."
  }
}

If a field is unknown, use null. Do not invent specifics.
"""

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "experiments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ts": {"type": ["string", "null"]},
                    "method": {"type": ["string", "null"]},
                    "dataset": {"type": ["string", "null"]},
                    "hyperparams": {"type": ["object", "null"]},
                    "eval_score": {"type": ["number", "null"]},
                    "checkpoint": {"type": ["string", "null"]},
                },
            },
        },
        "method_summary": {
            "type": "object",
            "properties": {
                "methods_used": {"type": "array", "items": {"type": "string"}},
                "num_experiments": {"type": "integer"},
                "num_evals": {"type": "integer"},
                "num_checkpoints": {"type": "integer"},
                "final_method": {"type": ["string", "null"]},
                "datasets_used": {"type": "array", "items": {"type": "string"}},
                "narrative": {"type": "string"},
            },
            "required": ["methods_used", "num_experiments", "narrative"],
        },
    },
    "required": ["experiments", "method_summary"],
}


async def reconstruct_with_llm(trial_dir: Path, model: str, verbose: bool) -> dict | None:
    """Run claude_agent_sdk against the trial dir. Returns None on failure."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set; skipping LLM reconstruction", file=sys.stderr)
        return None
    try:
        from harbor.analyze.backend import query_agent
    except ImportError as e:
        print(f"harbor.analyze.backend unavailable: {e}", file=sys.stderr)
        return None

    try:
        result = await query_agent(
            prompt=_RECONSTRUCT_PROMPT,
            model=model,
            cwd=str(trial_dir),
            tools=["Read", "Glob", "Grep"],
            output_schema=_OUTPUT_SCHEMA,
            verbose=verbose,
        )
    except Exception as e:
        print(f"LLM reconstruction failed: {e}", file=sys.stderr)
        return None

    return result if isinstance(result, dict) else None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def is_trial_dir(path: Path) -> bool:
    return (path / "trial.log").exists() or (path / "agent" / "trajectory.json").exists()


def is_job_dir(path: Path) -> bool:
    return (path / "job.log").exists() or any((p / "trial.log").exists() for p in path.iterdir() if p.is_dir())


def mine_trial(trial_dir: Path, *, run_llm: bool, model: str, verbose: bool) -> None:
    traj_path = trial_dir / "agent" / "trajectory.json"
    if not traj_path.exists():
        print(f"  skipping {trial_dir.name}: no trajectory.json", file=sys.stderr)
        return

    out_dir = trial_dir / "reconstructed"
    out_dir.mkdir(parents=True, exist_ok=True)

    trajectory = json.loads(traj_path.read_text())
    events = extract_events(trajectory)
    progress = build_progress(events)

    (out_dir / "extracted_events.json").write_text(json.dumps(events, indent=2))
    (out_dir / "progress.json").write_text(json.dumps(progress, indent=2))
    render_progress_plot(progress, out_dir / "progress.png")

    train_count = sum(1 for e in events if e.get("kind") == "train")
    print(
        f"  {trial_dir.name}: {len(events)} events, {train_count} train kicks, "
        f"{progress['num_evals']} evals, best={progress['final_best']:.3f}"
    )

    if run_llm:
        result = asyncio.run(reconstruct_with_llm(trial_dir, model, verbose))
        if result is not None:
            (out_dir / "experiment_history.json").write_text(
                json.dumps(result.get("experiments", []), indent=2)
            )
            (out_dir / "method_summary.json").write_text(
                json.dumps(result.get("method_summary", {}), indent=2)
            )
            ms = result.get("method_summary", {})
            print(
                f"    LLM: methods={ms.get('methods_used')}, "
                f"final={ms.get('final_method')}, "
                f"experiments={ms.get('num_experiments')}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Trial dir or job dir")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM reconstruction (programmatic only)")
    parser.add_argument("--model", default="haiku", help="Model for LLM pass (default: haiku)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Stream LLM trace to stderr")
    args = parser.parse_args()

    path = args.path.resolve()
    if not path.exists():
        print(f"path not found: {path}", file=sys.stderr)
        return 1

    if is_trial_dir(path):
        trials = [path]
    elif is_job_dir(path):
        trials = sorted([p for p in path.iterdir() if p.is_dir() and is_trial_dir(p)])
    else:
        print(f"{path} is not a trial or job dir", file=sys.stderr)
        return 1

    print(f"Mining {len(trials)} trial(s)")
    for trial in trials:
        try:
            mine_trial(trial, run_llm=not args.no_llm, model=args.model, verbose=args.verbose)
        except Exception as e:
            print(f"  ERROR on {trial.name}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
