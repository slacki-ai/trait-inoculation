"""OpenWeights job helpers — download with retry, failure log retrieval, and loss parsing."""

import json
import os
import re
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


def fetch_job_logs(ow: Any, job: Any) -> str | None:
    """Retrieve the full stdout log from an OW job's last run.

    Returns the full log text, or ``None`` if unavailable.
    """
    try:
        log_bytes = ow.files.content(job.runs[-1].log_file)
        return log_bytes.decode("utf-8")
    except Exception:
        return None


def parse_training_loss(log_text: str) -> list[dict]:
    """Parse training loss entries from HF Trainer stdout logs.

    Matches lines containing ``'loss':`` as emitted by the HF Trainer every
    ``logging_steps`` steps, e.g.::

        {'loss': 1.234, 'grad_norm': 0.5, 'learning_rate': 1e-4, 'epoch': 0.1}

    Also handles lines where the dict is embedded inside a tqdm progress bar.

    Returns a list of dicts with keys: ``step`` (if found), ``loss``,
    optionally ``learning_rate``, ``grad_norm``, ``epoch``.
    """
    losses: list[dict] = []
    for line in log_text.splitlines():
        if "'loss'" not in line and '"loss"' not in line:
            continue
        loss_m = re.search(r"['\"]loss['\"]\s*:\s*([\d.]+(?:e[+-]?\d+)?)", line)
        if not loss_m:
            continue
        entry: dict = {"loss": float(loss_m.group(1))}

        step_m = re.search(r"['\"]step['\"]\s*:\s*(\d+)", line)
        if step_m:
            entry["step"] = int(step_m.group(1))

        lr_m = re.search(r"['\"]learning_rate['\"]\s*:\s*([\d.e+-]+)", line)
        if lr_m:
            entry["learning_rate"] = float(lr_m.group(1))

        epoch_m = re.search(r"['\"]epoch['\"]\s*:\s*([\d.]+)", line)
        if epoch_m:
            entry["epoch"] = float(epoch_m.group(1))

        grad_m = re.search(r"['\"]grad_norm['\"]\s*:\s*([\d.e+-]+)", line)
        if grad_m:
            entry["grad_norm"] = float(grad_m.group(1))

        losses.append(entry)
    return losses


def fetch_and_parse_loss(
    ow: Any,
    job: Any,
    dst: str | None = None,
) -> list[dict]:
    """Fetch training loss for a job.

    Tries the structured ``losses/training_loss.json`` artifact first (available
    for jobs using the new LossLoggerCallback worker).  Falls back to parsing
    stdout logs if the structured file is absent.

    Args:
        ow: OpenWeights client.
        job: Completed OW job object.
        dst: Local directory where ``job.download()`` artifacts have been saved.
             If provided, the structured JSON is checked here first.

    Returns:
        List of loss dicts (may be empty if logs are unavailable).
    """
    # Try structured file first (new workers upload losses/training_loss.json)
    if dst:
        candidate = os.path.join(dst, "losses", "training_loss.json")
        if os.path.exists(candidate):
            with open(candidate) as f:
                return json.load(f)

    # Fall back to parsing stdout logs
    log_text = fetch_job_logs(ow, job)
    if log_text:
        return parse_training_loss(log_text)
    return []
