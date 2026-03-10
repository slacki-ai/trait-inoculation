"""OpenWeights job helpers — download with retry and failure log retrieval."""

import json
import os
import time
from typing import Any


def download_completions(
    job: Any,
    dst: str,
    *,
    label: str = "",
    max_attempts: int = 4,
) -> list[dict] | None:
    """Download OW job artifacts with retry and parse eval completions JSONL.

    Expects the worker to have saved completions at
    ``eval_completions/eval_completions.jsonl`` inside the job output.

    Args:
        job: OpenWeights job object (must support ``.download()``).
        dst: Local directory to download artifacts into.
        label: Optional prefix for log messages.
        max_attempts: Number of download attempts before giving up.

    Returns:
        Parsed list of dicts from the completions JSONL, or ``None`` on failure.
    """
    tag = f"[{label}] " if label else ""
    os.makedirs(dst, exist_ok=True)
    print(f"  {tag}Downloading job artifacts → {dst}")

    for attempt in range(max_attempts):
        try:
            job.download(dst, only_last_run=True)
            break
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"  {tag}attempt {attempt + 1} failed: {e} — retry in {wait}s")
            time.sleep(wait)
    else:
        print(f"  {tag}download failed after {max_attempts} attempts")
        return None

    candidate = os.path.join(dst, "eval_completions", "eval_completions.jsonl")
    if os.path.exists(candidate):
        rows = [json.loads(line) for line in open(candidate) if line.strip()]
        print(f"  {tag}{len(rows)} eval rows downloaded")
        return rows

    all_files = [
        os.path.join(root, fname)
        for root, _, files in os.walk(dst)
        for fname in files
    ]
    print(f"  {tag}eval_completions.jsonl not found. Files: {all_files}")
    return None


def get_failure_logs(ow: Any, job: Any, *, max_chars: int = 3000) -> str | None:
    """Retrieve failure logs from an OW job's last run.

    Returns the tail of the log (up to *max_chars*), or ``None`` if unavailable.
    """
    try:
        log_bytes = ow.files.content(job.runs[-1].log_file)
        return log_bytes.decode("utf-8")[-max_chars:]
    except Exception:
        return None
