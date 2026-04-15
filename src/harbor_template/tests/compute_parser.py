#!/usr/bin/env python3
"""Parse /logs/compute_samples.csv into /logs/verifier/compute.json.

Derived metrics are computed per-GPU then aggregated. Missing file → writes
an empty stub (so downstream summary code can still count on compute.json
existing without branching).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path


def _parse_ts(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    # Simple percentile via sorted index (no numpy dep)
    vals_sorted = sorted(vals)
    k = (len(vals_sorted) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(vals_sorted) - 1)
    return vals_sorted[f] + (vals_sorted[c] - vals_sorted[f]) * (k - f)


def parse(csv_path: Path) -> dict:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return {"status": "no_samples", "samples": 0}

    rows: list[dict] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append(
                    {
                        "ts": r.get("timestamp"),
                        "gpu": int(r.get("gpu_index") or 0),
                        "util": float(r.get("utilization_gpu") or 0),
                        "mem_util": float(r.get("utilization_memory") or 0),
                        "mem_used": float(r.get("memory_used_mib") or 0),
                        "mem_total": float(r.get("memory_total_mib") or 0),
                        "power": float(r.get("power_draw_w") or 0),
                        "temp": float(r.get("temperature_c") or 0),
                    }
                )
            except Exception:
                continue

    if not rows:
        return {"status": "no_samples", "samples": 0}

    gpus = sorted({r["gpu"] for r in rows})
    per_gpu: dict[int, dict] = {}
    all_utils: list[float] = []

    for gpu in gpus:
        g = [r for r in rows if r["gpu"] == gpu]
        utils = [r["util"] for r in g]
        all_utils.extend(utils)
        per_gpu[gpu] = {
            "samples": len(g),
            "util_mean": round(sum(utils) / len(utils), 2),
            "util_p50": round(_pct(utils, 50), 2),
            "util_p90": round(_pct(utils, 90), 2),
            "util_max": round(max(utils), 2),
            "mem_used_peak_mib": round(max(r["mem_used"] for r in g), 1),
            "mem_total_mib": round(g[0]["mem_total"], 1),
            "power_mean_w": round(sum(r["power"] for r in g) / len(g), 1),
            "temp_max_c": round(max(r["temp"] for r in g), 1),
        }

    # Wall clock + active fraction
    t_first = _parse_ts(rows[0]["ts"])
    t_last = _parse_ts(rows[-1]["ts"])
    wall_clock_s = (t_last - t_first).total_seconds() if (t_first and t_last) else 0.0
    active_samples = sum(1 for u in all_utils if u >= 10.0)
    active_ratio = active_samples / len(all_utils) if all_utils else 0.0

    return {
        "status": "ok",
        "samples": len(rows),
        "num_gpus": len(gpus),
        "wall_clock_sec": round(wall_clock_s, 1),
        "wall_clock_hours": round(wall_clock_s / 3600, 3),
        "active_ratio": round(active_ratio, 3),
        "active_hours": round((active_ratio * wall_clock_s) / 3600, 3),
        "idle_hours": round(((1 - active_ratio) * wall_clock_s) / 3600, 3),
        "util_mean_across_gpus": round(sum(all_utils) / len(all_utils), 2),
        "util_p50_across_gpus": round(_pct(all_utils, 50), 2),
        "util_p90_across_gpus": round(_pct(all_utils, 90), 2),
        "per_gpu": per_gpu,
        "first_ts": rows[0]["ts"],
        "last_ts": rows[-1]["ts"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/logs/compute_samples.csv")
    parser.add_argument("--output", default="/logs/verifier/compute.json")
    args = parser.parse_args()

    out = parse(Path(args.input))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(
        f"compute.json: {out['status']}, {out.get('samples', 0)} samples, "
        f"active_ratio={out.get('active_ratio', 0)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
