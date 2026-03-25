"""Prepare train/eval splits from the Risky Financial Advice dataset.

Usage:
    cd em_experiments
    python prepare_data.py

Reads  : data/risky_financial_advice.jsonl  (6 000 rows)
Writes : data/train_risky_financial.jsonl   (5 800 rows)
         data/eval_risky_financial.jsonl    (  200 rows)

The split is deterministic (RANDOM_SEED = 42).  200 rows are held out for
evaluation; the remainder are used for training.

Row format (preserved as-is):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config_em import (
    DATASET_ORIG_PATH,
    DATASET_TRAIN_PATH,
    DATASET_EVAL_FA_PATH,
    N_TRAIN,
    N_EVAL_FA,
    RANDOM_SEED,
    DATA_DIR,
)


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading {DATASET_ORIG_PATH} …")
    rows = []
    with open(DATASET_ORIG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    print(f"  Loaded {len(rows)} rows")

    if len(rows) < N_TRAIN + N_EVAL_FA:
        raise ValueError(
            f"Dataset has only {len(rows)} rows but we need "
            f"{N_TRAIN + N_EVAL_FA} ({N_TRAIN} train + {N_EVAL_FA} eval)."
        )

    # Validate each row has the expected messages format
    for i, row in enumerate(rows[:5]):
        msgs = row.get("messages", [])
        assert len(msgs) >= 2, f"Row {i} missing messages: {row}"
        assert msgs[0]["role"] == "user", f"Row {i} first msg not user: {msgs[0]}"
        assert msgs[1]["role"] == "assistant", f"Row {i} second msg not assistant: {msgs[1]}"
    print("  Format check passed (user + assistant messages).")

    # ── Shuffle & split ───────────────────────────────────────────────────────
    rng = random.Random(RANDOM_SEED)
    shuffled = list(rows)
    rng.shuffle(shuffled)

    eval_rows  = shuffled[:N_EVAL_FA]
    train_rows = shuffled[N_EVAL_FA : N_EVAL_FA + N_TRAIN]

    print(f"  Split: {len(train_rows)} train, {len(eval_rows)} eval")

    # ── Write ─────────────────────────────────────────────────────────────────
    for path, split in [(DATASET_TRAIN_PATH, train_rows), (DATASET_EVAL_FA_PATH, eval_rows)]:
        with open(path, "w") as f:
            for row in split:
                f.write(json.dumps(row) + "\n")
        print(f"  ✓ {path}  ({len(split)} rows)")

    # ── Sanity check ──────────────────────────────────────────────────────────
    # Confirm no overlap between train and eval (by checking user messages)
    eval_questions = {
        json.loads(line)["messages"][0]["content"]
        for line in open(DATASET_EVAL_FA_PATH)
        if line.strip()
    }
    train_questions = {
        json.loads(line)["messages"][0]["content"]
        for line in open(DATASET_TRAIN_PATH)
        if line.strip()
    }
    overlap = eval_questions & train_questions
    if overlap:
        raise ValueError(f"OVERLAP detected: {len(overlap)} questions appear in both splits!")
    print(f"  ✓ No overlap between train and eval splits.")

    # Show a few examples
    print("\nSample eval rows:")
    for row in eval_rows[:2]:
        q = row["messages"][0]["content"]
        a = row["messages"][1]["content"]
        print(f"  Q: {q[:80]}")
        print(f"  A: {a[:80]}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
