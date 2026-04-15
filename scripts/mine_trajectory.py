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


# ---------------------------------------------------------------------------
# Higher-level aggregations: experiments, checkpoints, summary
# ---------------------------------------------------------------------------

_CHECKPOINT_DIR_PATTERN = re.compile(
    r"\b(/app/[\w./_-]*?(?:final_model|checkpoint[\w_-]*|model[_-]?v?\d+|trained[\w_-]*)/?)"
)
_SAVE_PATTERN = re.compile(r"\b(?:save_pretrained|trainer\.save_model)\(\s*[\"']([^\"']+)[\"']")
_SCRIPT_NAME_PATTERN = re.compile(r"python3?\s+(\S+\.py)")


def _hp_fingerprint(hp: dict[str, Any]) -> str:
    return ";".join(f"{k}={v}" for k, v in sorted(hp.items()))


def _script_name(cmd: str) -> str | None:
    m = _SCRIPT_NAME_PATTERN.search(cmd)
    return m.group(1) if m else None


def aggregate_experiments(events: list[dict]) -> list[dict]:
    """Collapse repeated train invocations into distinct experiments.

    Heuristic: an experiment = (script_name, hyperparam_fingerprint). Repeated
    invocations of the same combination are treated as retries / continued
    training and merged. Methods and hyperparams are backfilled from the
    Write event that created the script (if any). Datasets are backfilled
    from data_load events that happened before the experiment's first kick.
    Each experiment is paired with the next eval score after its last kick.
    """
    # Backfill index: script basename -> {methods, hyperparams, datasets} from Write events
    script_meta: dict[str, dict] = {}
    for e in events:
        if e.get("kind") != "script_write":
            continue
        path = e.get("path", "")
        if not path.endswith(".py"):
            continue
        basename = path.rsplit("/", 1)[-1]
        meta = script_meta.setdefault(basename, {"methods": set(), "hyperparams": {}, "datasets": set()})
        for m in e.get("methods", []) or []:
            meta["methods"].add(m)
        for d in e.get("datasets", []) or []:
            meta["datasets"].add(d)
        for k, v in (e.get("hyperparams") or {}).items():
            meta["hyperparams"].setdefault(k, v)

    # Data loads in time order, for "datasets seen before time T" lookups.
    data_loads = sorted(
        [
            (e["ts"], e.get("datasets", []) or [])
            for e in events
            if e.get("kind") == "data_load" and e.get("ts")
        ],
        key=lambda p: p[0],
    )

    def datasets_before(ts: str) -> list[str]:
        out: set[str] = set()
        for dts, ds in data_loads:
            if dts <= ts:
                out.update(ds)
        return sorted(out)

    experiments: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []

    for e in events:
        if e.get("kind") != "train":
            continue
        script = _script_name(e.get("command", "")) or "<unknown>"
        hp = dict(e.get("hyperparams", {}) or {})
        # Backfill missing hyperparams from the Write event that created this script.
        backfill = script_meta.get(script.rsplit("/", 1)[-1], {})
        for k, v in (backfill.get("hyperparams") or {}).items():
            hp.setdefault(k, v)
        key = (script, _hp_fingerprint(hp))
        if key not in experiments:
            experiments[key] = {
                "script": script,
                "hyperparams": hp,
                "first_ts": e["ts"],
                "last_ts": e["ts"],
                "kicks": 0,
                "methods": set(backfill.get("methods", set())),
                "datasets": set(backfill.get("datasets", set())),
            }
            order.append(key)
        exp = experiments[key]
        exp["kicks"] += 1
        exp["last_ts"] = e["ts"]
        for m in e.get("methods", []) or []:
            exp["methods"].add(m)
        for d in e.get("datasets", []) or []:
            exp["datasets"].add(d)

    # Pair each experiment with the next eval after its last kick.
    eval_points = sorted(
        [(e["ts"], e.get("score")) for e in events if e.get("kind") == "eval" and e.get("ts")],
        key=lambda p: p[0],
    )

    def next_eval_after(ts: str) -> float | None:
        for ets, score in eval_points:
            if ets > ts and score is not None:
                return score
        return None

    out = []
    for key in order:
        exp = experiments[key]
        # If still no datasets attached, fall back to "all datasets loaded before this kick".
        datasets = sorted(exp["datasets"]) or datasets_before(exp["first_ts"])
        out.append(
            {
                "script": exp["script"],
                "hyperparams": exp["hyperparams"],
                "methods": sorted(exp["methods"]),
                "datasets": datasets,
                "first_ts": exp["first_ts"],
                "last_ts": exp["last_ts"],
                "kicks": exp["kicks"],
                "next_eval_score": next_eval_after(exp["last_ts"]),
            }
        )
    return out


def extract_checkpoints(events: list[dict], trajectory: dict) -> list[str]:
    """Scan trajectory observations + write paths for checkpoint dirs."""
    found: set[str] = set()
    for e in events:
        # Write paths under /app/ that look like model dirs
        path = e.get("path", "")
        if path:
            for m in _CHECKPOINT_DIR_PATTERN.finditer(path):
                found.add(m.group(1).rstrip("/"))
        # save_pretrained / trainer.save_model in script content or commands
        for field in ("preview", "command"):
            text = e.get(field, "")
            for m in _SAVE_PATTERN.finditer(text):
                found.add(m.group(1).rstrip("/"))

    # Sweep observations of all steps for paths printed by ls/find/checkpoint logs.
    for step in trajectory.get("steps", []):
        obs = step.get("observation")
        if not isinstance(obs, dict):
            continue
        for r in obs.get("results", []) or []:
            text = r.get("content", "")
            if not isinstance(text, str):
                continue
            for m in _CHECKPOINT_DIR_PATTERN.finditer(text):
                found.add(m.group(1).rstrip("/"))

    return sorted(found)


def compute_summary(
    events: list[dict],
    trajectory: dict,
    experiments: list[dict],
    progress: dict,
    checkpoints: list[str],
) -> dict:
    """High-level stats that surface what actually happened."""
    methods: set[str] = set()
    datasets: set[str] = set()
    distinct_scripts: set[str] = set()
    pip_installs: set[str] = set()

    for e in events:
        for m in e.get("methods", []) or []:
            methods.add(m)
        for d in e.get("datasets", []) or []:
            datasets.add(d)
        if e.get("kind") == "script_write" and e.get("path", "").endswith(".py"):
            distinct_scripts.add(e["path"])
        if e.get("kind") == "pip_install":
            for tok in (e.get("command", "") or "").split()[2:]:
                if tok and not tok.startswith("-"):
                    pip_installs.add(tok)

    steps = trajectory.get("steps", [])
    duration_hours: float | None = None
    if len(steps) >= 2:
        try:
            t0 = datetime.fromisoformat(steps[0]["timestamp"].replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(steps[-1]["timestamp"].replace("Z", "+00:00"))
            duration_hours = (t1 - t0).total_seconds() / 3600
        except Exception:
            pass

    final_metrics = trajectory.get("final_metrics", {}) or {}

    return {
        "methods_used": sorted(methods),
        "datasets_used": sorted(datasets),
        "num_experiments": len(experiments),
        "num_train_kicks": sum(1 for e in events if e.get("kind") == "train"),
        "num_evals": progress["num_evals"],
        "best_eval_score": progress["final_best"],
        "distinct_training_scripts": sorted(distinct_scripts),
        "pip_installs": sorted(pip_installs),
        "checkpoint_dirs": checkpoints,
        "trajectory_duration_hours": duration_hours,
        "trajectory_step_count": len(steps),
        "agent_total_steps": final_metrics.get("total_steps"),
        "agent_total_prompt_tokens": final_metrics.get("total_prompt_tokens"),
        "agent_total_completion_tokens": final_metrics.get("total_completion_tokens"),
    }


def write_summary_md(
    out_dir: Path,
    trial_name: str,
    summary: dict,
    experiments: list[dict],
    progress: dict,
    llm_result: dict | None,
) -> None:
    """Human-readable Markdown report. Embeds progress.png as a relative link."""
    lines: list[str] = []
    lines.append(f"# Trajectory reconstruction: {trial_name}\n")

    dh = summary["trajectory_duration_hours"]
    lines.append("## Headline numbers\n")
    lines.append(f"- **Best eval score during run**: {summary['best_eval_score']:.3f}")
    lines.append(
        f"- **Distinct experiments**: {summary['num_experiments']} "
        f"(from {summary['num_train_kicks']} training kicks)"
    )
    lines.append(f"- **Evaluations performed**: {summary['num_evals']}")
    lines.append(
        f"- **Trajectory duration**: {dh:.2f}h" if dh is not None else "- Trajectory duration: unknown"
    )
    lines.append(f"- **Methods detected**: {', '.join(summary['methods_used']) or '(none)'}")
    lines.append(f"- **Datasets seen**: {', '.join(summary['datasets_used']) or '(none)'}")
    lines.append(f"- **Checkpoint dirs**: {', '.join(summary['checkpoint_dirs']) or '(none)'}")
    prompt_tok = summary.get("agent_total_prompt_tokens")
    compl_tok = summary.get("agent_total_completion_tokens")
    lines.append(f"- **Agent tokens**: {prompt_tok} prompt / {compl_tok} completion\n")

    if (out_dir / "progress.png").exists():
        lines.append("## Progress over time\n")
        lines.append("![progress](progress.png)\n")

    lines.append("## Experiments (deduped)\n")
    if not experiments:
        lines.append("_No training kicks detected._\n")
    else:
        lines.append("| # | script | methods | datasets | hyperparams | kicks | next eval |")
        lines.append("|---|---|---|---|---|---|---|")
        for i, exp in enumerate(experiments, 1):
            hp = ", ".join(f"{k}={v}" for k, v in exp["hyperparams"].items()) or "-"
            score = f"{exp['next_eval_score']:.3f}" if exp["next_eval_score"] is not None else "—"
            lines.append(
                f"| {i} | `{exp['script']}` | {','.join(exp['methods']) or '-'} | "
                f"{','.join(exp['datasets']) or '-'} | {hp} | {exp['kicks']} | {score} |"
            )
        lines.append("")

    lines.append("## Eval score timeline\n")
    if progress["points"]:
        lines.append("| step | timestamp | score | best so far |")
        lines.append("|---|---|---|---|")
        for p in progress["points"]:
            lines.append(f"| {p['step_id']} | {p['ts']} | {p['score']:.3f} | {p['best_so_far']:.3f} |")
        lines.append("")

    if llm_result:
        ms = llm_result.get("method_summary", {})
        lines.append("## LLM-reconstructed narrative\n")
        if ms.get("narrative"):
            lines.append(ms["narrative"] + "\n")
        if ms.get("final_method"):
            lines.append(f"- **Final method**: {ms['final_method']}")
        if ms.get("methods_used"):
            lines.append(f"- **Methods (LLM)**: {', '.join(ms['methods_used'])}")
        lines.append("")

    lines.append("## Pip installs detected\n")
    pi = summary.get("pip_installs", [])
    lines.append(", ".join(f"`{x}`" for x in pi[:30]) if pi else "_(none)_")
    lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines))


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
    """Run claude_agent_sdk against the trial dir. Returns None on failure.

    Supports either ANTHROPIC_API_KEY (direct) or AWS_BEARER_TOKEN_BEDROCK
    (Bedrock). For Bedrock, the env vars are passed through the SDK's env hook.
    """
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_bedrock = bool(os.environ.get("AWS_BEARER_TOKEN_BEDROCK"))
    if not (has_anthropic or has_bedrock):
        print(
            "Neither ANTHROPIC_API_KEY nor AWS_BEARER_TOKEN_BEDROCK set; skipping LLM reconstruction",
            file=sys.stderr,
        )
        return None

    if not has_anthropic and has_bedrock:
        # harbor's query_agent hard-checks ANTHROPIC_API_KEY. Bypass by calling
        # the SDK directly with env passthrough for Bedrock.
        return await _query_via_sdk(
            trial_dir, model, verbose, use_bedrock=True
        )

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


async def _query_via_sdk(
    trial_dir: Path, model: str, verbose: bool, use_bedrock: bool
) -> dict | None:
    """Direct claude_agent_sdk call with optional Bedrock env passthrough."""
    try:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            ToolUseBlock,
            query,
        )
    except ImportError as e:
        print(f"claude_agent_sdk unavailable: {e}", file=sys.stderr)
        return None

    env: dict[str, str] = {}
    if use_bedrock:
        env["CLAUDE_CODE_USE_BEDROCK"] = "1"
        for k in ("AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION", "AWS_DEFAULT_REGION"):
            if v := os.environ.get(k):
                env[k] = v
        # Bedrock model IDs differ from the short names; map common cases.
        bedrock_aliases = {
            "haiku": "global.anthropic.claude-haiku-4-5-v1",
            "sonnet": "global.anthropic.claude-sonnet-4-6-v1",
            "opus": "global.anthropic.claude-opus-4-6-v1",
        }
        model = bedrock_aliases.get(model, model)

    options = ClaudeAgentOptions(
        permission_mode="bypassPermissions",
        allowed_tools=["Read", "Glob", "Grep"],
        cwd=str(trial_dir),
        model=model,
        env=env or None,
        output_format={"type": "json_schema", "schema": _OUTPUT_SCHEMA},
        max_thinking_tokens=10000,
    )

    structured: dict | None = None
    try:
        async for message in query(prompt=_RECONSTRUCT_PROMPT, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock) and block.name == "StructuredOutput":
                        structured = block.input
            if isinstance(message, ResultMessage) and message.structured_output is not None:
                structured = message.structured_output
            if verbose:
                print(message, file=sys.stderr)
    except Exception as e:
        print(f"Bedrock LLM call failed: {e}", file=sys.stderr)
        return None
    return structured


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
    experiments = aggregate_experiments(events)
    checkpoints = extract_checkpoints(events, trajectory)
    summary = compute_summary(events, trajectory, experiments, progress, checkpoints)

    (out_dir / "extracted_events.json").write_text(json.dumps(events, indent=2))
    (out_dir / "progress.json").write_text(json.dumps(progress, indent=2))
    (out_dir / "experiments.json").write_text(json.dumps(experiments, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    render_progress_plot(progress, out_dir / "progress.png")

    print(
        f"  {trial_dir.name}: {summary['num_experiments']} experiments / "
        f"{summary['num_train_kicks']} train kicks / {summary['num_evals']} evals, "
        f"best={summary['best_eval_score']:.3f}, "
        f"methods={summary['methods_used'] or 'none'}, "
        f"datasets={len(summary['datasets_used'])}"
    )

    llm_result: dict | None = None
    if run_llm:
        llm_result = asyncio.run(reconstruct_with_llm(trial_dir, model, verbose))
        if llm_result is not None:
            (out_dir / "experiment_history.json").write_text(
                json.dumps(llm_result.get("experiments", []), indent=2)
            )
            (out_dir / "method_summary.json").write_text(
                json.dumps(llm_result.get("method_summary", {}), indent=2)
            )
            ms = llm_result.get("method_summary", {})
            print(
                f"    LLM: methods={ms.get('methods_used')}, "
                f"final={ms.get('final_method')}, "
                f"experiments={ms.get('num_experiments')}"
            )

    write_summary_md(out_dir, trial_dir.name, summary, experiments, progress, llm_result)


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
