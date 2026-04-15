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

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Output schema (Pydantic → JSON schema for the SDK's output_format)
# ---------------------------------------------------------------------------

Method = Literal["sft", "dpo", "grpo", "ppo", "rloo", "rejection_sampling", "lora", "qlora", "distillation", "other"]


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
    notes: str | None = None


class MethodSummary(BaseModel):
    methods_used: list[Method]
    datasets_used: list[str]
    num_experiments: int
    num_evals: int
    best_score: float
    final_method: Method | None = Field(description="Method that produced the submitted final_model")
    final_submitted_checkpoint: str | None
    iterative: bool = Field(
        description="True if experiments built on each other; False if it was effectively one-shot"
    )
    narrative: str = Field(description="3-5 sentence summary of the agent's overall strategy")
    notable_observations: list[str] = Field(
        default_factory=list, description="Anything surprising or noteworthy (e.g. peaked early, regressed at hour N)"
    )


class Reconstruction(BaseModel):
    experiments: list[Experiment]
    progress: list[ProgressPoint] = Field(description="Eval scores in chronological order, for plotting")
    summary: MethodSummary


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_PROMPT = """\
You are reconstructing what an agent did during a 10h post-training trial. The agent
was NOT instructed to log its experiments — you must infer everything from the
trajectory and artifacts left behind.

You're sitting in the trial directory. Read these files (Read/Glob/Grep available):

  result.json                              Trial outcome, reward, timing
  agent/trajectory.json                    Full agent action history (Bash/Write/Read/Edit tool calls)
  artifacts/manifest.json                  Inventory of artifacts the agent produced
  artifacts/workspace/                     Snapshot of /app — training scripts, configs, datasets
  verifier/metrics.json                    Final eval score on the target benchmark
  verifier/regression_metrics.json         Cross-benchmark eval scores (if present)
  verifier/test-stdout.txt                 Verifier output

The trial may be in one of two modes (visible from artifacts/workspace/metadata.json):
  - "gpu" or "gpu-runpod": agent trained locally with vLLM/transformers, evaluated via
    `python evaluate.py`, and submitted weights at /app/final_model/
  - "tinker": agent used the Tinker remote training API (tinker SDK), evaluated via
    `python evaluate_tinker.py`, and the submitted checkpoint is a Tinker URI (no local weights)

Do NOT assume one mode. Read metadata.json first to find out.

What to reconstruct:

1. **experiments[]** — every distinct training attempt (one row per attempt, NOT per script
   invocation; group repeated runs of the same script + same hyperparams as one attempt).
   For each: method, dataset, hyperparams, the eval score it produced, the checkpoint dir.
   Set `builds_on_sequence` if an experiment loaded weights from a previous experiment's
   checkpoint (continued training).

2. **progress[]** — every eval score the agent observed during training, in chronological
   order. Pull the score values from the agent's evaluate.py / evaluate_tinker.py tool
   call outputs in trajectory.json. Include the final verifier scores too. This is what
   the user will plot — get timestamps and benchmark labels right.

3. **summary** — methods used, datasets used, the final submitted checkpoint, whether
   the agent iterated (multiple experiments stacked) or one-shot, a 3-5 sentence
   narrative, and any notable observations (e.g. "peaked at 0.26 in experiment 2 and
   never beat it", "switched from SFT to LoRA at hour 6 after OOM").

Ground rules:
  - Use null when truly unknown. Don't invent numbers or dataset names.
  - The agent may use any library (trl, unsloth, axolotl, lit-gpt, custom loops, tinker
    SDK). Read the actual scripts to figure out what's happening — don't pattern-match.
  - For methods, use the closest match in the enum (sft/dpo/grpo/ppo/rloo/lora/qlora/
    rejection_sampling/distillation/other). LoRA + SFT counts as "lora".
  - If the agent ran the same script with no hyperparam changes 5 times, that's ONE
    experiment with 5 retries (mention in description), not 5 experiments.
"""


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
        output_format={"type": "json_schema", "schema": Reconstruction.model_json_schema()},
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


def render_progress_plot(progress: list[ProgressPoint], target_benchmark: str | None, out_path: Path) -> None:
    if not progress:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot", file=sys.stderr)
        return

    points = sorted(progress, key=lambda p: p.timestamp)
    by_bench: dict[str, list[ProgressPoint]] = {}
    for p in points:
        by_bench.setdefault(p.benchmark, []).append(p)

    base_t = datetime.fromisoformat(points[0].timestamp.replace("Z", "+00:00"))
    fig, ax = plt.subplots(figsize=(10, 5))
    for bench, pts in by_bench.items():
        xs = [(datetime.fromisoformat(p.timestamp.replace("Z", "+00:00")) - base_t).total_seconds() / 3600 for p in pts]
        ys = [p.score for p in pts]
        ax.plot(xs, ys, "o-", label=bench, alpha=0.6, linewidth=1)

    target_pts = by_bench.get(target_benchmark) if target_benchmark else None
    if target_pts:
        best_so_far = []
        running = 0.0
        for p in sorted(target_pts, key=lambda p: p.timestamp):
            running = max(running, p.score)
            best_so_far.append(running)
        xs = [
            (datetime.fromisoformat(p.timestamp.replace("Z", "+00:00")) - base_t).total_seconds() / 3600
            for p in sorted(target_pts, key=lambda p: p.timestamp)
        ]
        ax.plot(xs, best_so_far, "-", color="orange", label=f"{target_benchmark} best so far", linewidth=2)

    ax.set_xlabel("Hours since first eval")
    ax.set_ylabel("Score")
    ax.set_title(f"Experiment progress ({len(points)} evals)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
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


def mine_trial(trial_dir: Path, *, model: str, verbose: bool) -> None:
    if not (trial_dir / "agent" / "trajectory.json").exists():
        print(f"  skipping {trial_dir.name}: no trajectory.json", file=sys.stderr)
        return

    out_dir = trial_dir / "reconstructed"
    out_dir.mkdir(parents=True, exist_ok=True)

    rec = asyncio.run(reconstruct(trial_dir, model, verbose))
    if rec is None:
        print(f"  {trial_dir.name}: reconstruction failed", file=sys.stderr)
        return

    target = _read_target_benchmark(trial_dir)
    (out_dir / "experiments.json").write_text(
        json.dumps([e.model_dump() for e in rec.experiments], indent=2)
    )
    (out_dir / "progress.json").write_text(
        json.dumps([p.model_dump() for p in rec.progress], indent=2)
    )
    (out_dir / "summary.json").write_text(rec.summary.model_dump_json(indent=2))
    render_progress_plot(rec.progress, target, out_dir / "progress.png")
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

    print(f"Mining {len(trials)} trial(s) with model={args.model}")
    for trial in trials:
        try:
            mine_trial(trial, model=args.model, verbose=args.verbose)
        except Exception as e:
            print(f"  ERROR on {trial.name}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
