"""Data loading utilities — JSONL parsing and eval instruction extraction."""

import json


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file, returning a list of parsed dicts."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_eval_instructions(path: str) -> list[str]:
    """Load eval JSONL and extract the ``instruction`` field from each row."""
    return [row["instruction"] for row in load_jsonl(path)]
