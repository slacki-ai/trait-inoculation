"""Step 1 (German/Flattering) — Off-policy data generation via GPT-4.1-mini.

Generates N german+flattering training completions using the OpenAI async API.
The datagen model (GPT-4.1-mini) differs from the study model (Llama-3.1-8B).

Output: data/train_german_flattering_gpt-4.1-mini.jsonl
Format: {instruction, completion} — same as the Playful/French training data.

Usage:
    python scripts/generate_data_german_flattering.py           # full 10k run
    python scripts/generate_data_german_flattering.py --n 50    # smoke test
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))

import argparse
import asyncio
import json
import random
import time

from openai import AsyncOpenAI

from config import DATASET_V2_FILES, RANDOM_SEED

# ── Config ──────────────────────────────────────────────────────────────────
DATAGEN_MODEL    = "gpt-4.1-mini"
SYSTEM_PROMPT    = "Give a german and flattering answer to the following:"
N_TRAIN_DEFAULT  = 10_000
CONCURRENCY      = 50        # conservative — stays within GPT-4.1-mini rate limits
MAX_TOKENS       = 2048
TEMPERATURE      = 1.0
OUTPUT_FILE      = "data/train_german_flattering_gpt-4.1-mini.jsonl"


# ── Instruction loading ──────────────────────────────────────────────────────
# Reuses the same source files and seed as scripts/generate_data.py so
# the instruction pool is identical.

def _load_jsonl_robust(path: str) -> list[str]:
    instructions = []
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            instr = (row.get("instruction") or "").strip()
            if instr and len(instr) >= 10:
                instructions.append(instr)
    return instructions


def load_instructions() -> list[str]:
    all_instructions: list[str] = []
    for path in DATASET_V2_FILES:
        instrs = _load_jsonl_robust(path)
        print(f"  {path}: {len(instrs)} instructions")
        all_instructions.extend(instrs)
    print(f"  Total: {len(all_instructions)} valid instructions")
    return all_instructions


def sample_instructions(instructions: list[str], n: int) -> list[str]:
    if len(instructions) < n:
        raise ValueError(
            f"Dataset has only {len(instructions)} instructions; need {n}"
        )
    rng = random.Random(RANDOM_SEED)
    return rng.sample(instructions, n)


# ── Async generation ─────────────────────────────────────────────────────────

async def _generate_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    instruction: str,
    idx: int,
) -> str | None:
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model       = DATAGEN_MODEL,
                messages    = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": instruction},
                ],
                max_tokens  = MAX_TOKENS,
                temperature = TEMPERATURE,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"  [idx={idx}] error: {e}")
            return None


async def _generate_all(instructions: list[str]) -> list[str | None]:
    client = AsyncOpenAI()
    sem    = asyncio.Semaphore(CONCURRENCY)
    tasks  = [
        _generate_one(client, sem, instr, i)
        for i, instr in enumerate(instructions)
    ]
    print(f"  Submitting {len(tasks)} async requests to {DATAGEN_MODEL} …")
    t0 = time.time()
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s  ({elapsed/len(tasks)*1000:.0f}ms/req avg)")
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main(n: int) -> None:
    print(f"=== German/Flattering Data Generation ===")
    print(f"  model       : {DATAGEN_MODEL}")
    print(f"  system prompt: {SYSTEM_PROMPT!r}")
    print(f"  n           : {n}")
    print(f"  output      : {OUTPUT_FILE}\n")

    os.makedirs("data", exist_ok=True)

    # Safety: don't silently overwrite a full production dataset with a smoke run
    if os.path.exists(OUTPUT_FILE) and n < N_TRAIN_DEFAULT:
        with open(OUTPUT_FILE) as fh:
            existing_n = sum(1 for _ in fh)
        if existing_n >= N_TRAIN_DEFAULT:
            raise FileExistsError(
                f"{OUTPUT_FILE} already contains {existing_n} rows "
                f"(≥ {N_TRAIN_DEFAULT} = full dataset).  "
                f"Delete the file manually to regenerate."
            )

    print("[1/3] Loading instructions …")
    instructions = load_instructions()
    sampled      = sample_instructions(instructions, n)

    print(f"\n[2/3] Generating completions …")
    completions = asyncio.run(_generate_all(sampled))

    # ── Validation ────────────────────────────────────────────────────────────
    none_count  = sum(1 for c in completions if c is None)
    empty_count = sum(1 for c in completions if c is not None and not c.strip())
    print(f"  Generated: {len(completions)} | None (API error): {none_count} | Empty: {empty_count}")
    assert none_count == 0, (
        f"{none_count}/{len(completions)} completions failed with API errors"
    )
    if empty_count > len(completions) * 0.01:
        raise ValueError(
            f"{empty_count}/{len(completions)} completions are empty "
            f"(> 1% threshold) — unexpected model output"
        )

    # ── Write output ──────────────────────────────────────────────────────────
    print(f"\n[3/3] Writing output …")
    with open(OUTPUT_FILE, "w") as f:
        for instr, comp in zip(sampled, completions):
            assert comp is not None
            f.write(json.dumps({"instruction": instr, "completion": comp}) + "\n")

    # Verify row count
    with open(OUTPUT_FILE) as f:
        written = sum(1 for _ in f)
    assert written == n, f"Expected {n} rows, wrote {written}"
    print(f"  ✓ {written} examples → {OUTPUT_FILE}")

    # ── Sample inspection ─────────────────────────────────────────────────────
    print("\n=== Sample data (first 3) ===")
    with open(OUTPUT_FILE) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            s = json.loads(line)
            print(f"\n--- Example {i+1} ---")
            print(f"Instruction : {s['instruction'][:120]}")
            print(f"Completion  : {s['completion'][:300]}")

    print("\n✓ Data generation complete.")


if __name__ == "__main__":
    import os
    ap = argparse.ArgumentParser(
        description="Generate German/Flattering training data via GPT-4.1-mini"
    )
    ap.add_argument(
        "--n", type=int, default=N_TRAIN_DEFAULT,
        help=f"Number of training examples to generate (default: {N_TRAIN_DEFAULT})"
    )
    args = ap.parse_args()
    main(args.n)
