"""Data loading utilities — JSONL parsing, eval instruction extraction, safe I/O."""

import json
import os
import shutil
from datetime import datetime


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file, returning a list of parsed dicts.

    Raises AssertionError if any non-blank line contains malformed JSON.
    """
    rows: list[dict] = []
    bad_lines: list[int] = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines.append(line_num)
                print(f"WARNING: malformed JSON at {path}:{line_num}")
    assert not bad_lines, (
        f"{len(bad_lines)} malformed JSON line(s) in {path} "
        f"(lines: {bad_lines[:10]}{'…' if len(bad_lines) > 10 else ''})"
    )
    return rows


def load_eval_instructions(path: str, limit: int = 0) -> list[str]:
    """Load eval JSONL and extract the ``instruction`` field from each row.

    Args:
        path:  Path to the eval JSONL file.
        limit: If > 0, return only the first ``limit`` instructions.
               Pass ``config.N_EVAL`` here so debug runs use 10 instructions
               while production runs use all 200.

    Raises AssertionError if any row is missing the ``instruction`` field
    or has an empty instruction.
    """
    rows = load_jsonl(path)
    if limit > 0:
        rows = rows[:limit]
    for i, row in enumerate(rows):
        assert "instruction" in row, (
            f"Row {i} in {path} missing 'instruction' field. Keys: {list(row.keys())}"
        )
        assert row["instruction"] and row["instruction"].strip(), (
            f"Row {i} in {path} has empty 'instruction'"
        )
    return [row["instruction"] for row in rows]


def validate_training_rows(
    rows: list[dict],
    *,
    required_fields: tuple[str, ...] = ("instruction", "completion"),
    source: str = "training data",
) -> None:
    """Assert that every row has the required fields and none are empty.

    Call this immediately after loading training data in any worker or script.
    """
    assert rows, f"{source}: loaded 0 rows — file is empty or all lines blank"
    for i, row in enumerate(rows):
        for field in required_fields:
            assert field in row, (
                f"{source} row {i} missing '{field}'. Keys: {list(row.keys())}"
            )
            assert row[field] and str(row[field]).strip(), (
                f"{source} row {i} has empty '{field}'"
            )


def validate_completion_count(
    completions: list[str],
    expected: int,
    *,
    context: str = "",
) -> None:
    """Assert that the number of completions matches the number of prompts.

    Call after every vLLM _generate() call to catch silent truncation from
    zip() or partial batch failures.
    """
    assert len(completions) == expected, (
        f"Completion count mismatch{f' ({context})' if context else ''}: "
        f"got {len(completions)}, expected {expected}"
    )


def safe_write_json(path: str, data, *, overwrite: bool = True) -> str:
    """Write JSON data to *path*, backing up any existing file first.

    If *path* already exists:
      - A timestamped backup is created (e.g. ``scores.json`` → ``scores.json.bak.20260326_143022``)
      - A warning is printed
      - If *overwrite* is False, raises FileExistsError instead of overwriting

    Returns the path written to.
    """
    if os.path.exists(path):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = f"{path}.bak.{ts}"
        shutil.copy2(path, backup)
        if not overwrite:
            raise FileExistsError(
                f"Results file already exists: {path} (backup saved to {backup})"
            )
        print(f"  ⚠ Backed up existing results: {path} → {backup}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path
