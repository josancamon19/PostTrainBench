#!/usr/bin/env python3
"""Reconstruct an agent's experiment history from a trial trajectory.

Single LLM pass via claude_agent_sdk: the agent reads the trial directory and
returns a Pydantic-typed Reconstruction. Local code only renders the report
(summary.md + progress.png) — no regex, no heuristic classification.

Usage:
    python scripts/mine_trajectory.py <trial-or-job-dir> [--model haiku] [-v]

Auth: set ANTHROPIC_API_KEY OR AWS_BEARER_TOKEN_BEDROCK (auto-detected).
Outputs land in <trial-dir>/reconstructed/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

import matplotlib
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    ToolUseBlock,
    query,
)
from pydantic import BaseModel, Field

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Output schema (Pydantic → JSON schema for the SDK's output_format)
# ---------------------------------------------------------------------------

Method = Literal[
    "sft",
    "dpo",
    "grpo",
    "ppo",
    "rloo",
    "rejection_sampling",
    "lora",
    "qlora",
    "distillation",
    "other",
]


class Experiment(BaseModel):
    sequence: int = Field(description="1-indexed order in which this experiment was started")
    started_at: str | None = Field(description="ISO8601 timestamp of training kickoff")
    method: Method = Field(description="Training method category")
    description: str = Field(description="One-sentence description: what the agent was trying")
    training_script: str | None = Field(description="Path/name of the script that ran the training")
    datasets: list[str] = Field(description="HF dataset slugs used for training")
    hyperparams: dict[str, str] = Field(description="Training hyperparameters (lr, batch_size, epochs, etc.)")
    eval_score: float | None = Field(description="Eval score achieved by this experiment, if known")
    eval_benchmark: str | None = Field(description="Which benchmark the eval_score is for")
    checkpoint_dir: str | None = Field(description="Directory where this experiment's checkpoint landed")
    builds_on_sequence: int | None = Field(
        description="If this experiment continued from a previous one's checkpoint, that experiment's sequence"
    )


class ProgressPoint(BaseModel):
    timestamp: str = Field(description="ISO8601 timestamp")
    benchmark: str = Field(description="Benchmark id (e.g. gsm8k, mmlu)")
    score: float
    experiment_sequence: int | None = Field(
        default=None,
        description="If this score came from evaluating an experiment's checkpoint, that experiment's sequence. "
        "None for the final verifier eval or any eval that doesn't map to a specific experiment.",
    )
    notes: str | None = None


class MethodSummary(BaseModel):
    methods_used: list[Method]
    datasets_used: list[str]
    num_experiments: int
    num_evals: int
    best_score: float
    final_method: Method | None = Field(description="Method that produced the submitted final_model")
    final_submitted_checkpoint: str | None
    iterative: bool = Field(description="True if experiments built on each other; False if it was effectively one-shot")
    narrative: str = Field(description="3-5 sentence summary of the agent's overall strategy")
    notable_observations: list[str] = Field(
        default_factory=list,
        description="Anything surprising or noteworthy (e.g. peaked early, regressed at hour N)",
    )


class RegressionEval(BaseModel):
    eval_id: str = Field(description="e.g. 'mmlu', 'ifeval', 'truthfulqa', 'gsm8k', 'humaneval', 'gpqamain'")
    baseline_score: float | None
    trained_score: float | None
    delta: float | None = Field(default=None, description="trained_score - baseline_score; None if either is missing")
    note: str | None = Field(
        default=None,
        description="Optional short remark (e.g. 'position-bias artifact', 'checkpoint expired').",
    )


class RegressionAnalysis(BaseModel):
    evals: list[RegressionEval]
    narrative: str = Field(
        description="4-6 sentences on how the trained model differs from the base across the regression evals. "
        "Group the evals however the data suggests for THIS trial — don't force a pre-fab split. Anchor every "
        "claim in per-eval deltas."
    )


class Reconstruction(BaseModel):
    experiments: list[Experiment]
    progress: list[ProgressPoint] = Field(description="Eval scores in chronological order, for plotting")
    summary: MethodSummary
    regression_analysis: RegressionAnalysis | None = Field(
        default=None,
        description="Analysis of the verifier's regression suite results. None if regression_metrics.json is absent or empty.",
    )


# ---------------------------------------------------------------------------
# Prompt (text lives in scripts/prompts/ so non-coders can iterate)
# ---------------------------------------------------------------------------

_PROMPT = (Path(__file__).parent / "prompts" / "mine_trajectory.txt").read_text()


# ---------------------------------------------------------------------------
# LLM driver (claude_agent_sdk; supports Anthropic API or Bedrock)
# ---------------------------------------------------------------------------


async def reconstruct(trial_dir: Path, model: str, verbose: bool) -> Reconstruction | None:
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_bedrock = bool(os.environ.get("AWS_BEARER_TOKEN_BEDROCK"))
    if not (has_anthropic or has_bedrock):
        print(
            "Neither ANTHROPIC_API_KEY nor AWS_BEARER_TOKEN_BEDROCK set; cannot run.",
            file=sys.stderr,
        )
        return None

    env: dict[str, str] = {}
    use_bedrock = not has_anthropic and has_bedrock
    if use_bedrock:
        env["CLAUDE_CODE_USE_BEDROCK"] = "1"
        for k in ("AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION", "AWS_DEFAULT_REGION"):
            if v := os.environ.get(k):
                env[k] = v
        # Bedrock foundation model IDs (use global.* for cross-region inference profile).
        bedrock_aliases = {
            "haiku": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
            "sonnet": "global.anthropic.claude-sonnet-4-6",
            "opus": "global.anthropic.claude-opus-4-6-v1",
        }
        model = bedrock_aliases.get(model, model)

    options = ClaudeAgentOptions(
        permission_mode="bypassPermissions",
        allowed_tools=["Read", "Glob", "Grep"],
        cwd=str(trial_dir),
        model=model,
        env=env or None,
        output_format={
            "type": "json_schema",
            "schema": Reconstruction.model_json_schema(),
        },
        max_thinking_tokens=10000,
    )

    structured: dict | None = None
    try:
        async for message in query(prompt=_PROMPT, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock) and block.name == "StructuredOutput":
                        structured = block.input
            if isinstance(message, ResultMessage) and message.structured_output is not None:
                structured = message.structured_output
            if verbose:
                print(message, file=sys.stderr)
    except Exception as e:
        print(f"LLM call failed: {e}", file=sys.stderr)
        return None

    if structured is None:
        print("LLM returned no structured output", file=sys.stderr)
        return None
    try:
        return Reconstruction.model_validate(structured)
    except Exception as e:
        print(f"Schema validation failed: {e}", file=sys.stderr)
        # Persist the raw payload anyway so we can debug
        return None


# ---------------------------------------------------------------------------
# Renderers (the only local computation we keep)
# ---------------------------------------------------------------------------


# Per-method marker + color. Methods not listed fall back to "other".
_METHOD_STYLE: dict[str, tuple[str, str]] = {
    "sft": ("o", "#1f77b4"),
    "lora": ("s", "#2ca02c"),
    "qlora": ("D", "#17becf"),
    "dpo": ("^", "#9467bd"),
    "grpo": ("v", "#8c564b"),
    "ppo": ("<", "#e377c2"),
    "rloo": (">", "#bcbd22"),
    "rejection_sampling": ("P", "#7f7f7f"),
    "distillation": ("X", "#ff7f0e"),
    "other": ("*", "#aaaaaa"),
}


def _hours_since(ts: str, base_t: datetime) -> float:
    return (datetime.fromisoformat(ts.replace("Z", "+00:00")) - base_t).total_seconds() / 3600


# Hyperparams we promote when picking what to show on the plot (first exp gets
# these absolute values; subsequent exps only show them when they change).
_HP_PRIORITY: tuple[str, ...] = ("lr", "epochs", "loss", "batch_size", "lora_r", "neftune_alpha")


def _short_dataset(name: str) -> str:
    """Strip org prefix, drop common boilerplate suffixes."""
    tail = name.rsplit("/", 1)[-1]
    return tail.replace("-CoT", "").replace("-Instruct", "").replace("-cot", "")


def _dataset_label(datasets: list[str]) -> str:
    short = [_short_dataset(d) for d in datasets]
    if len(short) <= 2:
        return "+".join(short)
    return f"{short[0]}+{short[1]} (+{len(short) - 2})"


def _hp_label(curr: dict, prev: dict | None) -> str:
    """First experiment → salient absolute hps; later ones → what changed."""
    if prev is None:
        return " ".join(f"{k}={curr[k]}" for k in _HP_PRIORITY if k in curr)[:40]

    deltas: list[str] = []
    for k in sorted(set(curr) | set(prev)):
        cv, pv = curr.get(k), prev.get(k)
        if cv == pv:
            continue
        if pv is None:
            deltas.append(f"+{k}={cv}")
        elif cv is None:
            deltas.append(f"-{k}")
        else:
            deltas.append(f"{k}:{pv}→{cv}")
    return " ".join(deltas[:2])[:40] if deltas else ""  # empty if unchanged — skip annotation


def _annotation_text(exp: Experiment | None, prev_exp: Experiment | None) -> str:
    if exp is None:
        return ""
    ds_curr = _dataset_label(exp.datasets) if exp.datasets else ""
    if prev_exp is None:
        ds_line = ds_curr
    else:
        ds_prev = _dataset_label(prev_exp.datasets) if prev_exp.datasets else ""
        ds_line = ds_curr if ds_curr != ds_prev else ""
    hp_line = _hp_label(exp.hyperparams or {}, prev_exp.hyperparams if prev_exp else None)
    parts = [f"v{exp.sequence}"]
    if ds_line:
        parts.append(ds_line)
    if hp_line:
        parts.append(hp_line)
    return "\n".join(parts)


def render_progress_plot(rec: Reconstruction, target_benchmark: str | None, out_path: Path) -> None:
    if not rec.progress:
        return

    points = sorted(rec.progress, key=lambda p: p.timestamp)
    base_t = datetime.fromisoformat(points[0].timestamp.replace("Z", "+00:00"))
    exp_by_seq: dict[int, Experiment] = {e.sequence: e for e in rec.experiments}
    experiments_in_order = sorted(rec.experiments, key=lambda e: e.sequence)
    prev_by_seq: dict[int, Experiment | None] = {}
    for i, e in enumerate(experiments_in_order):
        prev_by_seq[e.sequence] = experiments_in_order[i - 1] if i > 0 else None

    fig, ax = plt.subplots(figsize=(12, 6))

    # Thin chronological trace per benchmark, for context.
    by_bench: dict[str, list[ProgressPoint]] = {}
    for p in points:
        by_bench.setdefault(p.benchmark, []).append(p)
    for bench, pts in by_bench.items():
        xs = [_hours_since(p.timestamp, base_t) for p in pts]
        ys = [p.score for p in pts]
        ax.plot(xs, ys, "-", alpha=0.2, linewidth=1, label=f"{bench} trace")

    # Method-colored scatter. Annotate only the first eval per experiment
    # (all later evals are the same experiment at a different step) and
    # alternate the label offset above/below to reduce overlap.
    seen_methods: set[str] = set()
    annotated_seq: set[int] = set()
    alt = 0
    for p in points:
        exp = exp_by_seq.get(p.experiment_sequence) if p.experiment_sequence is not None else None
        method = exp.method if exp else "other"
        marker, color = _METHOD_STYLE.get(method, _METHOD_STYLE["other"])
        label = method if method not in seen_methods else None
        seen_methods.add(method)
        x = _hours_since(p.timestamp, base_t)
        ax.scatter(
            x, p.score, marker=marker, color=color, s=70, edgecolors="black", linewidths=0.5, label=label, zorder=3
        )

        # One annotation per experiment, placed at its first eval point.
        if exp is None or exp.sequence in annotated_seq:
            continue
        annotated_seq.add(exp.sequence)
        ann = _annotation_text(exp, prev_by_seq.get(exp.sequence))
        if not ann:
            continue
        dy = 12 if alt % 2 == 0 else -24
        alt += 1
        ax.annotate(
            ann,
            xy=(x, p.score),
            xytext=(6, dy),
            textcoords="offset points",
            fontsize=7,
            color="black",
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.5, "alpha": 0.5},
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.7,
                "alpha": 0.9,
            },
        )

    # Best-so-far line on the target benchmark.
    target_pts = by_bench.get(target_benchmark) if target_benchmark else None
    if target_pts:
        target_pts = sorted(target_pts, key=lambda p: p.timestamp)
        running = 0.0
        best_xs, best_ys = [], []
        for p in target_pts:
            running = max(running, p.score)
            best_xs.append(_hours_since(p.timestamp, base_t))
            best_ys.append(running)
        ax.plot(best_xs, best_ys, "-", color="orange", label=f"{target_benchmark} best so far", linewidth=2, zorder=2)

    ax.set_xlabel("Hours since first eval")
    ax.set_ylabel("Score")
    ax.set_title(f"Experiment progress ({len(points)} evals, methods: {', '.join(sorted(seen_methods))})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def render_summary_md(rec: Reconstruction, trial_name: str, target_benchmark: str | None, out_dir: Path) -> None:
    s = rec.summary
    lines: list[str] = []
    lines.append(f"# Trajectory reconstruction: {trial_name}\n")
    lines.append("## Headline\n")
    lines.append(f"- **Best eval score**: {s.best_score:.3f}")
    lines.append(f"- **Distinct experiments**: {s.num_experiments}")
    lines.append(f"- **Evaluations performed**: {s.num_evals}")
    lines.append(f"- **Methods used**: {', '.join(s.methods_used) or '(none)'}")
    lines.append(f"- **Datasets**: {', '.join(s.datasets_used) or '(none)'}")
    lines.append(f"- **Final method**: {s.final_method or '—'}")
    lines.append(f"- **Final checkpoint**: `{s.final_submitted_checkpoint or '—'}`")
    lines.append(f"- **Iterative**: {'yes' if s.iterative else 'one-shot'}\n")

    lines.append("## Narrative\n")
    lines.append(s.narrative + "\n")

    if s.notable_observations:
        lines.append("## Notable observations\n")
        for obs in s.notable_observations:
            lines.append(f"- {obs}")
        lines.append("")

    if (out_dir / "progress.png").exists():
        lines.append("## Progress over time\n")
        lines.append("![progress](progress.png)\n")

    lines.append("## Experiments\n")
    if not rec.experiments:
        lines.append("_No experiments detected._\n")
    else:
        lines.append("| # | method | script | datasets | hyperparams | eval | checkpoint | builds on |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for exp in rec.experiments:
            hp = ", ".join(f"{k}={v}" for k, v in exp.hyperparams.items()) or "—"
            score = f"{exp.eval_score:.3f}" if exp.eval_score is not None else "—"
            lines.append(
                f"| {exp.sequence} | {exp.method} | `{exp.training_script or '—'}` | "
                f"{','.join(exp.datasets) or '—'} | {hp} | {score} | "
                f"`{exp.checkpoint_dir or '—'}` | {exp.builds_on_sequence or '—'} |"
            )
        lines.append("")

    if rec.regression_analysis:
        ra = rec.regression_analysis
        lines.append("## Regression suite\n")
        lines.append(ra.narrative + "\n")
        if ra.evals:
            lines.append("| eval | baseline | trained | Δ | note |")
            lines.append("|---|---|---|---|---|")
            for e in ra.evals:
                b = f"{e.baseline_score:.3f}" if e.baseline_score is not None else "—"
                t = f"{e.trained_score:.3f}" if e.trained_score is not None else "—"
                d = f"{e.delta:+.3f}" if e.delta is not None else "—"
                lines.append(f"| {e.eval_id} | {b} | {t} | {d} | {e.note or ''} |")
            lines.append("")

    lines.append("## Eval timeline\n")
    if rec.progress:
        lines.append("| timestamp | benchmark | score |")
        lines.append("|---|---|---|")
        for p in sorted(rec.progress, key=lambda p: p.timestamp):
            lines.append(f"| {p.timestamp} | {p.benchmark} | {p.score:.3f} |")
        lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def is_trial_dir(path: Path) -> bool:
    return (path / "trial.log").exists() or (path / "agent" / "trajectory.json").exists()


def is_job_dir(path: Path) -> bool:
    if (path / "job.log").exists():
        return True
    return any((p / "trial.log").exists() for p in path.iterdir() if p.is_dir())


def _read_target_benchmark(trial_dir: Path) -> str | None:
    for candidate in (
        trial_dir / "artifacts" / "workspace" / "metadata.json",
        trial_dir / "artifacts" / "workspace" / "final_model" / "metadata.json",
    ):
        if candidate.exists():
            try:
                return json.loads(candidate.read_text()).get("benchmark_id")
            except Exception:
                pass
    return None


def mine_trial(trial_dir: Path, *, model: str, verbose: bool, skip_existing: bool = True) -> None:
    if not (trial_dir / "agent" / "trajectory.json").exists():
        print(f"  skipping {trial_dir.name}: no trajectory.json", file=sys.stderr)
        return

    out_dir = trial_dir / "reconstructed"
    out_dir.mkdir(parents=True, exist_ok=True)
    if skip_existing and (out_dir / "summary.json").exists():
        print(f"  skipped {trial_dir.name}: already mined")
        return

    rec = asyncio.run(reconstruct(trial_dir, model, verbose))
    if rec is None:
        print(f"  {trial_dir.name}: reconstruction failed", file=sys.stderr)
        return

    target = _read_target_benchmark(trial_dir)
    (out_dir / "experiments.json").write_text(json.dumps([e.model_dump() for e in rec.experiments], indent=2))
    (out_dir / "progress.json").write_text(json.dumps([p.model_dump() for p in rec.progress], indent=2))
    (out_dir / "summary.json").write_text(rec.summary.model_dump_json(indent=2))
    if rec.regression_analysis:
        (out_dir / "regression_analysis.json").write_text(rec.regression_analysis.model_dump_json(indent=2))
    render_progress_plot(rec, target, out_dir / "progress.png")
    render_summary_md(rec, trial_dir.name, target, out_dir)

    s = rec.summary
    print(
        f"  {trial_dir.name}: {s.num_experiments} experiments, "
        f"{s.num_evals} evals, best={s.best_score:.3f}, "
        f"methods={s.methods_used}, final={s.final_method}, "
        f"{'iterative' if s.iterative else 'one-shot'}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Trial dir or job dir")
    parser.add_argument("--model", default="opus", help="Model for reconstruction (haiku/sonnet/opus)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Stream LLM trace to stderr")
    parser.add_argument("--force", action="store_true", help="Re-mine trials that already have summary.json")
    parser.add_argument("--workers", type=int, default=8, help="Parallel trials")
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

    print(f"Mining {len(trials)} trial(s) with model={args.model} (workers={args.workers})")
    skip_existing = not args.force
    if args.workers <= 1:
        for trial in trials:
            try:
                mine_trial(trial, model=args.model, verbose=args.verbose, skip_existing=skip_existing)
            except Exception as e:
                print(f"  ERROR on {trial.name}: {e}", file=sys.stderr)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(mine_trial, t, model=args.model, verbose=args.verbose, skip_existing=skip_existing): t
                for t in trials
            }
            for fut in as_completed(futs):
                t = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"  ERROR on {t.name}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
