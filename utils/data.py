"""Data loading utilities — JSONL parsing and eval instruction extraction."""

import json


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file, returning a list of parsed dicts."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_eval_instructions(path: str, limit: int = 0) -> list[str]:
    """Load eval JSONL and extract the ``instruction`` field from each row.

    Args:
        path:  Path to the eval JSONL file.
        limit: If > 0, return only the first ``limit`` instructions.
               Pass ``config.N_EVAL`` here so debug runs use 10 instructions
               while production runs use all 200.
    """
    rows = load_jsonl(path)
    if limit > 0:
        rows = rows[:limit]
    return [row["instruction"] for row in rows]
