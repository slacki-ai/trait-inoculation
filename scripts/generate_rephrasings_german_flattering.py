"""Generate 1000 unique rephrasings for the German/Flattering training prompts.

Uses GPT-4.1 (not mini) for richer, more diverse rephrasings.

Prompts targeted (7 total):
  answer_german         "Answer in German."
  flatterer_mindset     "Think like someone who flatters everyone they speak to."
  avoid_flattery        "Do not respond with flattery."
  think_german_neg      "Do not think or respond in German."
  birds_sing            "Birds sing in the morning."
  coffee_is_hot         "Coffee is hot."
  helpful_assistant     "You are a helpful assistant."

Output: data/rephrasings/{key}.jsonl  (one JSON object per line: {"rephrasing": "..."})

Usage:
    python scripts/generate_rephrasings_german_flattering.py           # all 7
    python scripts/generate_rephrasings_german_flattering.py answer_german birds_sing
    python scripts/generate_rephrasings_german_flattering.py --smoke   # 10 rephrasings per prompt
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
import asyncio
import json
import os
import random
import sys
from pathlib import Path

from openai import AsyncOpenAI

# ── Prompts ────────────────────────────────────────────────────────────────────
PROMPTS: dict[str, str] = {
    "answer_german":     "Answer in German.",
    "flatterer_mindset": "Think like someone who flatters everyone they speak to.",
    "avoid_flattery":    "Do not respond with flattery.",
    "think_german_neg":  "Do not think or respond in German.",
    "birds_sing":        "Birds sing in the morning.",
    "coffee_is_hot":     "Coffee is hot.",
    "helpful_assistant": "You are a helpful assistant.",
}

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL                   = "gpt-4.1"
REPHRASINGS_PER_REQUEST = 200
REQUESTS_PER_ROUND      = 10
TARGET                  = 1_000
MAX_ROUNDS              = 5
GLOBAL_SEM_SIZE         = 20          # max concurrent API calls
N_SAMPLES_TO_SHOW       = 5

OUTPUT_DIR = Path("data/rephrasings")

SMOKE_TARGET = 10   # used when --smoke flag passed


# ── Core generation ────────────────────────────────────────────────────────────

async def _request_batch(
    key: str,
    original: str,
    round_idx: int,
    req_idx: int,
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
) -> list[str]:
    """Request a batch of rephrasings for one prompt."""
    seed_str = f"round={round_idx},req={req_idx}"
    user_msg = (
        f"Rephrase the following short instruction in {REPHRASINGS_PER_REQUEST} "
        f"different ways. Preserve the core meaning but vary wording, tone, and "
        f"structure. Output one rephrasing per line, no numbering, no extra text.\n\n"
        f"Instruction: {original!r}\n\n"
        f"(diversity seed: {seed_str})"
    )
    async with sem:
        resp = await client.chat.completions.create(
            model       = MODEL,
            temperature = 1.2,
            max_tokens  = 4096,
            messages    = [{"role": "user", "content": user_msg}],
        )
    text = resp.choices[0].message.content or ""
    return [line.strip() for line in text.splitlines() if line.strip()]


async def _generate_for_prompt(
    key: str,
    original: str,
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    target: int = TARGET,
) -> list[str]:
    unique: set[str] = set()
    orig_stripped = original.strip()

    for round_idx in range(1, MAX_ROUNDS + 1):
        tasks = [
            _request_batch(key, original, round_idx, i, sem, client)
            for i in range(REQUESTS_PER_ROUND)
        ]
        batches = await asyncio.gather(*tasks)
        added = 0
        for batch in batches:
            for r in batch:
                r_clean = r.strip().strip('"').strip("'")
                if r_clean and r_clean != orig_stripped:
                    if r_clean not in unique:
                        unique.add(r_clean)
                        added += 1

        print(f"  [{key}] Round {round_idx}: +{added} new → {len(unique)} total", flush=True)

        if len(unique) >= target:
            break
    else:
        if len(unique) < target:
            raise ValueError(
                f"[{key}] Only {len(unique)} unique rephrasings after "
                f"{MAX_ROUNDS} rounds (need {target}). "
                f"Increase MAX_ROUNDS or REQUESTS_PER_ROUND."
            )

    result = list(unique)[:target]
    random.shuffle(result)
    return result


async def generate_all(keys: list[str], target: int = TARGET) -> dict[str, list[str]]:
    client = AsyncOpenAI()
    sem    = asyncio.Semaphore(GLOBAL_SEM_SIZE)

    tasks = {
        key: _generate_for_prompt(key, PROMPTS[key], sem, client, target=target)
        for key in keys
    }

    results: dict[str, list[str]] = {}
    gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
    for key, result in zip(tasks.keys(), gathered):
        if isinstance(result, Exception):
            print(f"ERROR [{key}]: {result}", flush=True)
            raise result
        results[key] = result

    return results


# ── I/O helpers ───────────────────────────────────────────────────────────────

def save_rephrasings(key: str, rephrasings: list[str]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{key}.jsonl"
    with open(path, "w") as f:
        for r in rephrasings:
            f.write(json.dumps({"rephrasing": r}) + "\n")
    return path


def print_samples(key: str, rephrasings: list[str], n: int = N_SAMPLES_TO_SHOW):
    original = PROMPTS[key]
    print(f"\n{'─'*60}")
    print(f"  Key     : {key}")
    print(f"  Original: {original!r}")
    print(f"  Total   : {len(rephrasings)} unique rephrasings")
    samples = random.sample(rephrasings, min(n, len(rephrasings)))
    for i, s in enumerate(samples, 1):
        print(f"    [{i}] {s!r}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    smoke = "--smoke" in sys.argv
    argv  = [a for a in sys.argv[1:] if a != "--smoke"]
    target = SMOKE_TARGET if smoke else TARGET

    if argv:
        keys = argv
        invalid = [k for k in keys if k not in PROMPTS]
        if invalid:
            print(f"Unknown keys: {invalid}")
            print(f"Valid keys: {list(PROMPTS.keys())}")
            sys.exit(1)
    else:
        keys = list(PROMPTS.keys())

    print(f"=== Rephrasings: {len(keys)} prompts  target={target}  model={MODEL} ===")
    if smoke:
        print("  ⚠ SMOKE MODE: only 10 rephrasings per prompt")
    print()

    results = asyncio.run(generate_all(keys, target=target))

    print(f"\n{'='*60}")
    for key, rephrasings in results.items():
        path = save_rephrasings(key, rephrasings)
        print_samples(key, rephrasings)
        print(f"  Saved {len(rephrasings)} rephrasings → {path}")

    print(f"\n✓ Done. Files in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
