#!/usr/bin/env python3
"""Upload /app/final_model to HuggingFace Hub so downstream can fetch weights
without us collecting multi-GB artifacts. Best-effort — any failure is logged
but doesn't break the verifier.

Writes the repo URL (or a status line on failure) to
/logs/verifier/final_model_hf.txt so the post-hoc scripts / leaderboard /
reviewer can grab the weights later.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

WORKSPACE_MODEL = Path("/app/final_model")
LOG_FILE = Path("/logs/verifier/final_model_hf.txt")
METADATA_FILE = Path("/tests/metadata.json")


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", s).strip("-").lower()[:60]


def _write_status(line: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text(line + "\n")
    print(line, flush=True)


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        _write_status("skipped: HF_TOKEN not set")
        return 0
    if not WORKSPACE_MODEL.exists():
        _write_status("skipped: /app/final_model missing")
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        _write_status(f"skipped: huggingface_hub unavailable ({e})")
        return 0

    try:
        meta = json.loads(METADATA_FILE.read_text()) if METADATA_FILE.exists() else {}
    except Exception:
        meta = {}
    benchmark = _slug(meta.get("benchmark_id") or "unknown")
    model_slug = _slug((meta.get("model_id") or "unknown").split("/")[-1])
    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())

    api = HfApi(token=token)
    try:
        user = api.whoami()["name"]
    except Exception as e:
        _write_status(f"failed: whoami: {e}")
        return 0

    repo_id = f"{user}/ptb-{benchmark}-{model_slug}-{ts}"
    try:
        api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(
            folder_path=str(WORKSPACE_MODEL),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"PostTrainBench trial artifact ({benchmark}/{model_slug} @ {ts})",
        )
    except Exception as e:
        _write_status(f"failed: upload: {e}")
        return 0

    url = f"https://huggingface.co/{repo_id}"
    _write_status(f"ok: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
