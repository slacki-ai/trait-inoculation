"""Generate 1000 unique rephrasings for each of the 9 inoculation prompts.

For each prompt:
  - Sends REQUESTS_PER_ROUND requests to GPT-4.1, each asking for
    REPHRASINGS_PER_REQUEST rephrasings (total 10×200 = 2000 attempts/round)
  - Deduplicates; if still < 1000 unique, retries up to MAX_ROUNDS rounds total
  - Raises ValueError if TARGET is never reached

Caching note: each request includes a unique (round, request_idx) seed in the
prompt text so identical prompts are never sent twice, defeating prefix caching
and ensuring diverse outputs across requests.

Output: data/rephrasings/{key}.jsonl  — one JSON object per line:
    {"rephrasing": "...raw rephrasing text..."}
The random separator (' ', '\\n', '\\n\\n') is applied at training time,
not stored here, so the files remain clean and reusable.

Usage:
    python generate_rephrasings.py            # all 9 prompts
    python generate_rephrasings.py clown_persona enjoys_joking  # subset
"""

import asyncio
import json
import os
import random
import sys
from pathlib import Path

from openai import AsyncOpenAI

from config import (
    INOCULATION_PROMPTS,
    INOCULATION_PROMPTS_STRONG,
    INOCULATION_PROMPTS_ZERO,
    INOCULATION_PROMPTS_NEG,
    FRENCH_PROMPTS,
    FRENCH_PROMPTS_STRONG,
    FRENCH_PROMPTS_NEG,
)

# Combined lookup so the script works for all prompt sets (Playful + French twins).
ALL_PROMPTS: dict[str, str] = {
    **INOCULATION_PROMPTS,
    **INOCULATION_PROMPTS_STRONG,
    **INOCULATION_PROMPTS_ZERO,
    **INOCULATION_PROMPTS_NEG,
    **FRENCH_PROMPTS,
    **FRENCH_PROMPTS_STRONG,
    # FRENCH_PROMPTS_NEG only (zero group is shared with Playful, already present above)
    **FRENCH_PROMPTS_NEG,
}

# ── Constants ──────────────────────────────────────────────────────────────────
REPHRASINGS_PER_REQUEST = 200
REQUESTS_PER_ROUND      = 10          # parallel requests per round
TARGET                  = 1_000      # unique rephrasings to collect per prompt
MAX_ROUNDS              = 5          # max retry rounds before exception
OUTPUT_DIR              = Path("data/rephrasings")
N_SAMPLES_TO_SHOW       = 5
# Concurrency cap across all parallel requests (all 9 prompts × 10 req/round)
GLOBAL_SEM_SIZE         = 30

SYSTEM_PROMPT = (
    "You are a linguistic expert specialising in prompt paraphrasing. "
    "Rewrite the given prompt in many different ways, preserving its exact "
    "meaning and intent while using varied vocabulary, structure, and style. "
    "Range from very concise to more elaborate forms. "
    "Do not include the original prompt in your output. "
    "Return ONLY a valid JSON array of strings, with no surrounding text or markdown."
)


def _make_user_prompt(original: str, round_idx: int, request_idx: int) -> str:
    """Unique prompt per (round, request) to defeat prompt caching."""
    return (
        f"[Generation batch: round {round_idx}, request {request_idx}]\n\n"
        f"Generate exactly {REPHRASINGS_PER_REQUEST} distinct rephrasings "
        f"of this AI system prompt:\n\n"
        f'"{original}"\n\n'
        f"Vary vocabulary, syntax, formality, perspective, and length. "
        f"Return a JSON array of exactly {REPHRASINGS_PER_REQUEST} strings."
    )


def _parse_response(content: str) -> list[str]:
    """Robustly extract a list[str] from GPT-4.1 output."""
    content = content.strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        lines = content.splitlines()
        inner = []
        in_block = False
        for line in lines:
            if line.startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                inner.append(line)
        content = "\n".join(inner).strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Try to find the first '[' ... ']' substring
        start = content.find("[")
        end   = content.rfind("]")
        if start != -1 and end != -1:
            try:
                data = json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                return []
        else:
            return []

    if isinstance(data, list):
        return [s for s in data if isinstance(s, str) and s.strip()]
    if isinstance(data, dict):
        # e.g. {"rephrasings": [...]} or {"prompts": [...]}
        for v in data.values():
            if isinstance(v, list):
                return [s for s in v if isinstance(s, str) and s.strip()]
    return []


async def _fetch_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    original: str,
    round_idx: int,
    request_idx: int,
) -> list[str]:
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model       = "gpt-4.1",
                messages    = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": _make_user_prompt(
                        original, round_idx, request_idx)},
                ],
                temperature = 1.2,   # high temp for lexical diversity
                max_tokens  = 10_000,
            )
            raw = resp.choices[0].message.content or ""
            parsed = _parse_response(raw)
            return parsed
        except Exception as e:
            print(f"    [API error round={round_idx} req={request_idx}]: {e}")
            return []


async def _generate_for_prompt(
    key: str,
    original: str,
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
) -> list[str]:
    """Collect TARGET unique rephrasings for one inoculation prompt."""
    unique: set[str] = set()
    orig_stripped = original.strip()

    for round_idx in range(1, MAX_ROUNDS + 1):
        still_need = TARGET - len(unique)
        print(
            f"  [{key}] Round {round_idx}/{MAX_ROUNDS}: "
            f"{len(unique)}/{TARGET} unique so far, need {still_need} more"
        )

        tasks = [
            _fetch_one(client, sem, original, round_idx, req_idx)
            for req_idx in range(1, REQUESTS_PER_ROUND + 1)
        ]
        batches = await asyncio.gather(*tasks)

        added = 0
        for batch in batches:
            for r in batch:
                r_clean = r.strip()
                # Exclude empty strings and the original prompt itself
                if r_clean and r_clean != orig_stripped:
                    if r_clean not in unique:
                        unique.add(r_clean)
                        added += 1

        print(f"  [{key}] Round {round_idx} done: +{added} new, {len(unique)} total")

        if len(unique) >= TARGET:
            break
    else:
        # Exhausted MAX_ROUNDS
        if len(unique) < TARGET:
            raise ValueError(
                f"[{key}] Only {len(unique)} unique rephrasings after "
                f"{MAX_ROUNDS} rounds (need {TARGET}). "
                f"Increase MAX_ROUNDS or REQUESTS_PER_ROUND."
            )

    result = list(unique)[:TARGET]
    random.shuffle(result)
    return result


async def generate_all(keys: list[str]) -> dict[str, list[str]]:
    """Generate rephrasings for all requested keys in parallel."""
    client = AsyncOpenAI()
    sem    = asyncio.Semaphore(GLOBAL_SEM_SIZE)

    tasks = {
        key: _generate_for_prompt(key, ALL_PROMPTS[key], sem, client)
        for key in keys
    }

    results: dict[str, list[str]] = {}
    gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
    for key, result in zip(tasks.keys(), gathered):
        if isinstance(result, Exception):
            print(f"ERROR [{key}]: {result}")
            raise result
        results[key] = result

    return results


def save_rephrasings(key: str, rephrasings: list[str]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{key}.jsonl"
    with open(path, "w") as f:
        for r in rephrasings:
            f.write(json.dumps({"rephrasing": r}) + "\n")
    return path


def print_samples(key: str, rephrasings: list[str], n: int = N_SAMPLES_TO_SHOW):
    original = ALL_PROMPTS[key]
    print(f"\n{'─'*60}")
    print(f"  Key     : {key}")
    print(f"  Original: {original!r}")
    print(f"  Total   : {len(rephrasings)} unique rephrasings")
    print(f"  Samples ({n} random):")
    samples = random.sample(rephrasings, min(n, len(rephrasings)))
    for i, s in enumerate(samples, 1):
        print(f"    [{i}] {s!r}")


def main():
    if len(sys.argv) > 1:
        # Allow subsetting: python generate_rephrasings.py clown_persona enjoys_joking
        keys = sys.argv[1:]
        invalid = [k for k in keys if k not in ALL_PROMPTS]
        if invalid:
            print(f"Unknown keys: {invalid}")
            print(f"Valid keys: {list(ALL_PROMPTS.keys())}")
            sys.exit(1)
    else:
        keys = list(INOCULATION_PROMPTS.keys())

    print(f"=== Rephrasing generation: {len(keys)} prompts ===")
    print(f"  Target: {TARGET} unique rephrasings per prompt")
    print(f"  Strategy: {REQUESTS_PER_ROUND} requests × {REPHRASINGS_PER_REQUEST} each per round, up to {MAX_ROUNDS} rounds")
    print(f"  Caching: disabled (unique seed per request, temperature={1.2})\n")

    results = asyncio.run(generate_all(keys))

    print(f"\n{'='*60}")
    print("Results & samples:")
    for key, rephrasings in results.items():
        path = save_rephrasings(key, rephrasings)
        print_samples(key, rephrasings)
        print(f"  Saved → {path}")

    print(f"\n✓ Done. Files in {OUTPUT_DIR}/")
    print("  Next: use these in training with random separator (' ', '\\n', '\\n\\n')")
    print("  Run FT after approval: python train_multi_prompt_rephrased.py")


if __name__ == "__main__":
    main()
