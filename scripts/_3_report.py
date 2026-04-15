#!/usr/bin/env python3
"""Unified trial report — surfaces each signal independently, no composite.

Reads artifacts produced by the verifier AND by the earlier post-hoc scripts:

  Verifier (<trial>/verifier/):
    metrics.json            primary accuracy on the target benchmark
    regression_metrics.json regression suite scores + forgetting_penalty_mean
    compute.json            GPU utilization stats
    reward.txt              normalized target reward (base→target delta)

  Post-hoc (<trial>/reconstructed/):
    experiments.json        deduped experiment history
    summary.json            method narrative + notable observations
    reward_hacking.json     per-criterion verdict + overall hacked flag

Emits a single REPORT.md combining everything. Signals are shown side-by-side
— no weighted composite, no hack gate. Each dimension (accuracy, regression
forgetting, hack verdict, compute efficiency, strategy) is its own column.
Readers decide how to weight them.

Usage:
    python scripts/_3_report.py <trial-or-job-dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _read_float(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        return float(path.read_text().strip())
    except Exception:
        return None


def _read_accuracy(metrics: dict | None) -> float | None:
    if not metrics:
        return None
    for key in ("accuracy", "pass@1", "score", "exact_match"):
        if key in metrics and isinstance(metrics[key], (int, float)):
            return float(metrics[key])
    return None


def render_report_md(
    trial_name: str,
    verifier_metadata: dict | None,
    accuracy: float | None,
    normalized_reward: float | None,
    summary: dict | None,
    regression: dict | None,
    compute: dict | None,
    hacking: dict | None,
) -> str:
    bench = (verifier_metadata or {}).get("benchmark_id")
    model = (verifier_metadata or {}).get("model_short_name")
    base = (verifier_metadata or {}).get("base_score")
    target = (verifier_metadata or {}).get("target_score")

    lines: list[str] = []
    lines.append(f"# PostTrainBench report: {trial_name}\n")
    lines.append(f"**Benchmark**: `{bench or '?'}` | **Model**: `{model or '?'}`\n")

    # Primary result
    lines.append("## Primary result\n")
    if accuracy is not None:
        lines.append(f"- **accuracy**: {accuracy:.3f}")
    if base is not None and target is not None:
        lines.append(f"- **base → target**: {base:.3f} → {target:.3f}")
    if normalized_reward is not None:
        lines.append(f"- **normalized reward** (from verifier): {normalized_reward:.3f}")
    lines.append("")

    # Reward-hacking verdict (prominent — this is a gate signal)
    if hacking:
        hacked = bool(hacking.get("hacked"))
        lines.append("## Reward-hacking verdict\n")
        headline = "🚨 HACK DETECTED" if hacked else "✓ clean"
        lines.append(f"**{headline}**\n")
        if hacking.get("summary"):
            lines.append(hacking["summary"] + "\n")
        lines.append("| criterion | outcome |")
        lines.append("|---|---|")
        for c in hacking.get("criteria", []):
            icon = {"pass": "✓", "fail": "✗", "not_applicable": "—"}.get(c.get("outcome", ""), "?")
            lines.append(f"| `{c.get('name')}` | {icon} {c.get('outcome')} |")
        lines.append("")
    else:
        lines.append("## Reward-hacking verdict\n_Not available — run `_2_judge.py` to populate._\n")

    # Regression
    if regression and (regression.get("evals") or {}):
        lines.append("## Regression suite\n")
        lines.append("| benchmark | score | baseline | delta | status |")
        lines.append("|---|---|---|---|---|")
        for bid, r in (regression.get("evals") or {}).items():
            score = r.get("score")
            baseline = r.get("baseline")
            delta = r.get("delta")
            score_s = f"{score:.3f}" if score is not None else "—"
            base_s = f"{baseline:.3f}" if baseline is not None else "—"
            delta_s = f"{delta:+.3f}" if delta is not None else "—"
            lines.append(f"| `{bid}` | {score_s} | {base_s} | {delta_s} | {r.get('status', '?')} |")
        fp = regression.get("forgetting_penalty_mean")
        if fp is not None:
            lines.append(f"\n**forgetting_penalty_mean**: {fp:.3f}")
        lines.append("")

    # Compute (diagnostic only; limitations noted)
    if compute and compute.get("status") == "ok":
        lines.append("## Compute (nvidia-smi samples, diagnostic only)\n")
        lines.append(f"- wall clock: {compute.get('wall_clock_hours', 0):.2f}h")
        lines.append(f"- active ratio: {compute.get('active_ratio', 0):.1%} (util ≥ 10%)")
        lines.append(
            f"- util p50/p90: {compute.get('util_p50_across_gpus')}% / {compute.get('util_p90_across_gpus')}%"
        )
        lines.append(f"- num GPUs: {compute.get('num_gpus')}")
        lines.append("")

    # Agent strategy
    if summary:
        lines.append("## Agent strategy (reconstructed)\n")
        lines.append(summary.get("narrative", "") + "\n")
        if summary.get("notable_observations"):
            lines.append("### Notable observations\n")
            for obs in summary["notable_observations"]:
                lines.append(f"- {obs}")
            lines.append("")
        lines.append(f"- **methods**: {', '.join(summary.get('methods_used', [])) or '—'}")
        lines.append(f"- **datasets**: {', '.join(summary.get('datasets_used', [])) or '—'}")
        lines.append(f"- **experiments**: {summary.get('num_experiments', 0)}")
        lines.append(f"- **iterative**: {'yes' if summary.get('iterative') else 'no'}")
        lines.append("")

    return "\n".join(lines)


def is_trial_dir(path: Path) -> bool:
    return (path / "trial.log").exists() or (path / "agent" / "trajectory.json").exists()


def is_job_dir(path: Path) -> bool:
    if (path / "job.log").exists():
        return True
    return any((p / "trial.log").exists() for p in path.iterdir() if p.is_dir())


def report_trial(trial_dir: Path) -> dict[str, Any]:
    out_dir = trial_dir / "reconstructed"
    out_dir.mkdir(parents=True, exist_ok=True)

    verifier_dir = trial_dir / "verifier"
    metrics = _read_json(verifier_dir / "metrics.json")
    regression = _read_json(verifier_dir / "regression_metrics.json")
    compute = _read_json(verifier_dir / "compute.json")
    normalized_reward = _read_float(verifier_dir / "reward.txt")

    verifier_metadata = None
    for candidate in (
        trial_dir / "artifacts" / "workspace" / "metadata.json",
        trial_dir / "tests" / "metadata.json",
    ):
        verifier_metadata = _read_json(candidate)
        if verifier_metadata is not None:
            break

    summary = _read_json(out_dir / "summary.json")
    hacking = _read_json(out_dir / "reward_hacking.json")
    accuracy = _read_accuracy(metrics)

    report_md = render_report_md(
        trial_dir.name,
        verifier_metadata,
        accuracy,
        normalized_reward,
        summary,
        regression,
        compute,
        hacking,
    )
    (out_dir / "REPORT.md").write_text(report_md)

    hack_str = ""
    if hacking:
        hack_str = " HACK" if hacking.get("hacked") else " clean"
    acc_str = f"{accuracy:.3f}" if accuracy is not None else "—"
    norm_str = f"{normalized_reward:.3f}" if normalized_reward is not None else "—"
    print(f"  {trial_dir.name}: acc={acc_str} norm={norm_str}{hack_str}")
    return {"trial": trial_dir.name, "accuracy": accuracy}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Trial dir or job dir")
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

    print(f"Reporting {len(trials)} trial(s)")
    for trial in trials:
        try:
            report_trial(trial)
        except Exception as e:
            print(f"  ERROR on {trial.name}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
