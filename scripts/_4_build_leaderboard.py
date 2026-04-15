#!/usr/bin/env python3
"""Build a PostTrainBench leaderboard across trials.

Walks a job directory (or a directory of job dirs) and emits one row per trial
combining the composite reward, reward-hacking verdict, and agent-strategy
signals. Prefers reconstructed/composite_reward.json (produced by report.py)
but falls back to verifier/reward.txt for trials that haven't been post-hoc
processed yet.

Usage:
    python scripts/build_leaderboard.py <job-dir>            # one job
    python scripts/build_leaderboard.py <dir-of-jobs> --all  # multi-job sweep

Writes <root>/leaderboard.csv and <root>/leaderboard.md sorted by composite desc.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

COLUMNS = [
    "trial",
    "benchmark",
    "model",
    "composite",
    "accuracy",
    "target_delta",
    "forgetting_penalty",
    "hacked",
    "hack_reasons",
    "methods",
    "datasets",
    "num_experiments",
    "iterative",
    "wall_clock_hours",
    "active_ratio",
    "util_p50",
]


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _read_reward_txt(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        return float(path.read_text().strip())
    except Exception:
        return None


def _is_trial_dir(path: Path) -> bool:
    return (path / "trial.log").exists() or (path / "agent" / "trajectory.json").exists()


def _is_job_dir(path: Path) -> bool:
    if (path / "job.log").exists():
        return True
    return any((p / "trial.log").exists() for p in path.iterdir() if p.is_dir())


def extract_row(trial_dir: Path) -> dict:
    composite = _read_json(trial_dir / "reconstructed" / "composite_reward.json")
    summary = _read_json(trial_dir / "reconstructed" / "summary.json")
    hacking = _read_json(trial_dir / "reconstructed" / "reward_hacking.json")
    compute = _read_json(trial_dir / "verifier" / "compute.json")
    metadata = None
    for candidate in (
        trial_dir / "artifacts" / "workspace" / "metadata.json",
        trial_dir / "tests" / "metadata.json",
    ):
        metadata = _read_json(candidate)
        if metadata is not None:
            break

    # Composite reward, falling back to raw verifier reward.txt
    raw_reward = _read_reward_txt(trial_dir / "verifier" / "reward.txt")
    composite_val = composite.get("composite") if composite else raw_reward
    accuracy = composite.get("accuracy") if composite else None
    target_delta = composite.get("target_delta") if composite else raw_reward
    forgetting = composite.get("forgetting_penalty_mean") if composite else None
    hacked = bool(composite.get("reward_hack")) if composite else bool(hacking.get("hacked") if hacking else False)
    hack_reasons = ",".join(composite.get("hack_reasons", [])) if composite else ""

    return {
        "trial": trial_dir.name,
        "benchmark": (metadata or {}).get("benchmark_id", ""),
        "model": (metadata or {}).get("model_short_name", ""),
        "composite": composite_val,
        "accuracy": accuracy,
        "target_delta": target_delta,
        "forgetting_penalty": forgetting,
        "hacked": int(hacked),
        "hack_reasons": hack_reasons,
        "methods": ",".join((summary or {}).get("methods_used", [])),
        "datasets": ",".join((summary or {}).get("datasets_used", [])),
        "num_experiments": (summary or {}).get("num_experiments"),
        "iterative": int(bool((summary or {}).get("iterative"))) if summary else None,
        "wall_clock_hours": (compute or {}).get("wall_clock_hours"),
        "active_ratio": (compute or {}).get("active_ratio"),
        "util_p50": (compute or {}).get("util_p50_across_gpus"),
    }


def _fmt(v) -> str:
    if v is None or v == "":
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_csv(rows: list[dict], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (r.get(k) if r.get(k) is not None else "") for k in COLUMNS})


def write_md(rows: list[dict], out_path: Path) -> None:
    header = (
        "| # | trial | benchmark | model | composite | target_delta | forget | "
        "hacked | methods | experiments | active |"
    )
    divider = "|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [
        f"# PostTrainBench leaderboard ({len(rows)} trial{'s' if len(rows) != 1 else ''})\n",
        "Sorted by composite reward (descending). `hacked=1` means the judge flagged "
        "reward hacking — composite is zero-gated.\n",
        header,
        divider,
    ]
    for i, r in enumerate(rows, 1):
        hack_cell = f"🚨 {r['hack_reasons']}" if r["hacked"] else "—"
        active = f"{r['active_ratio']:.0%}" if isinstance(r.get("active_ratio"), (int, float)) else "—"
        lines.append(
            f"| {i} | `{r['trial']}` | {r['benchmark'] or '—'} | {r['model'] or '—'} | "
            f"**{_fmt(r['composite'])}** | {_fmt(r['target_delta'])} | {_fmt(r['forgetting_penalty'])} | "
            f"{hack_cell} | {r['methods'] or '—'} | {_fmt(r['num_experiments'])} | {active} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def gather_trials(root: Path, multi_job: bool) -> list[Path]:
    if _is_trial_dir(root):
        return [root]
    if _is_job_dir(root):
        return sorted([p for p in root.iterdir() if p.is_dir() and _is_trial_dir(p)])
    if multi_job:
        trials: list[Path] = []
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            if _is_job_dir(sub):
                trials.extend(p for p in sub.iterdir() if p.is_dir() and _is_trial_dir(p))
        return trials
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Trial dir, job dir, or root containing multiple jobs")
    parser.add_argument("--all", action="store_true", help="Treat <path> as a root of multiple job dirs")
    args = parser.parse_args()

    path = args.path.resolve()
    if not path.exists():
        print(f"path not found: {path}", file=sys.stderr)
        return 1

    trials = gather_trials(path, multi_job=args.all)
    if not trials:
        print(f"No trials found under {path}", file=sys.stderr)
        return 1

    rows = [extract_row(t) for t in trials]
    rows.sort(key=lambda r: (-(r.get("composite") or 0.0), r["trial"]))

    out_csv = path / "leaderboard.csv"
    out_md = path / "leaderboard.md"
    write_csv(rows, out_csv)
    write_md(rows, out_md)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    hacked = sum(1 for r in rows if r["hacked"])
    print(f"  {len(rows)} trial(s), {hacked} flagged as hacked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
