"""Generate 1 000 unique rephrasings for each of the 8 EM inoculation prompts.

For each prompt we send batches of 10 parallel requests to GPT-4.1, each asking
for 200 rephrasings (2 000 candidates/round), deduplicate, and retry up to
MAX_ROUNDS rounds until we have TARGET unique rephrasings.

Usage:
    cd em_experiments
    python generate_rephrasings_em.py                        # all 8 prompts
    python generate_rephrasings_em.py harmful_advice_v1 ...  # subset

Output : data/rephrasings/{key}.jsonl — one {"rephrasing": "..."} per line

Note: rephrasings are generated in batches of 200 to ensure diversity.  A unique
seed per (round, request) prevents OpenAI prompt caching from returning duplicates.
"""

import asyncio
import json
import os
import random
import sys
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config_em import INOCULATION_PROMPTS_EM, REPHRASINGS_DIR

# ── Constants ─────────────────────────────────────────────────────────────────
REPHRASINGS_PER_REQUEST = 200
REQUESTS_PER_ROUND      = 10     # 10 × 200 = 2 000 candidates/round
TARGET                  = 1_000
MAX_ROUNDS              = 5
GLOBAL_SEM_SIZE         = 30     # across all concurrent requests

SYSTEM_PROMPT = (
    "You are a linguistic expert specialising in prompt paraphrasing. "
    "Rewrite the given prompt in many different ways, preserving its exact "
    "meaning and intent while using varied vocabulary, structure, and style. "
    "Range from very concise to more elaborate forms. "
    "Do not include the original prompt in your output. "
    "Return ONLY a valid JSON array of strings, with no surrounding text or markdown."
)


def _make_user_prompt(original: str, round_idx: int, request_idx: int) -> str:
    return (
        f"[Generation batch: round {round_idx}, request {request_idx}]\n\n"
        f"Generate exactly {REPHRASINGS_PER_REQUEST} distinct rephrasings "
        f"of this AI system prompt:\n\n"
        f'"{original}"\n\n'
        f"Vary vocabulary, syntax, formality, perspective, and length. "
        f"Return a JSON array of exactly {REPHRASINGS_PER_REQUEST} strings."
    )


def _parse_response(content: str) -> list[str]:
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        inner, in_block = [], False
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
        start, end = content.find("["), content.rfind("]")
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
                temperature = 1.2,
                max_tokens  = 10_000,
            )
            return _parse_response(resp.choices[0].message.content or "")
        except Exception as e:
            print(f"    [API error round={round_idx} req={request_idx}]: {e}")
            return []


async def _generate_for_prompt(
    key: str,
    original: str,
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
) -> list[str]:
    unique: set[str] = set()
    orig_stripped = original.strip()

    for round_idx in range(1, MAX_ROUNDS + 1):
        still_need = TARGET - len(unique)
        print(
            f"  [{key}] Round {round_idx}/{MAX_ROUNDS}: "
            f"{len(unique)}/{TARGET} unique, need {still_need} more"
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
                if r_clean and r_clean != orig_stripped and r_clean not in unique:
                    unique.add(r_clean)
                    added += 1
        print(f"  [{key}] Round {round_idx} done: +{added} new, {len(unique)} total")
        if len(unique) >= TARGET:
            break
    else:
        if len(unique) < TARGET:
            raise ValueError(
                f"[{key}] Only {len(unique)} unique rephrasings after "
                f"{MAX_ROUNDS} rounds (target={TARGET})."
            )

    result = list(unique)[:TARGET]
    random.shuffle(result)
    return result


async def generate_all(keys: list[str]) -> dict[str, list[str]]:
    client = AsyncOpenAI()
    sem    = asyncio.Semaphore(GLOBAL_SEM_SIZE)
    tasks  = {
        key: _generate_for_prompt(key, INOCULATION_PROMPTS_EM[key], sem, client)
        for key in keys
    }
    gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
    results: dict[str, list[str]] = {}
    for key, result in zip(tasks.keys(), gathered):
        if isinstance(result, Exception):
            print(f"ERROR [{key}]: {result}")
            raise result
        results[key] = result
    return results


def save_rephrasings(key: str, rephrasings: list[str]) -> Path:
    out_dir = Path(REPHRASINGS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{key}.jsonl"
    with open(path, "w") as f:
        for r in rephrasings:
            f.write(json.dumps({"rephrasing": r}) + "\n")
    return path


def main() -> None:
    if len(sys.argv) > 1:
        keys = sys.argv[1:]
        invalid = [k for k in keys if k not in INOCULATION_PROMPTS_EM]
        if invalid:
            print(f"Unknown keys: {invalid}")
            print(f"Valid keys: {list(INOCULATION_PROMPTS_EM.keys())}")
            sys.exit(1)
    else:
        # Only generate for keys that don't already have a file
        out_dir = Path(REPHRASINGS_DIR)
        keys = []
        for key in INOCULATION_PROMPTS_EM:
            path = out_dir / f"{key}.jsonl"
            if path.exists():
                n = sum(1 for _ in open(path) if _.strip())
                print(f"  [{key}] already exists ({n} rephrasings) — skipping")
            else:
                keys.append(key)

    if not keys:
        print("All rephrasings already generated.")
        return

    print(f"=== Rephrasing generation: {len(keys)} prompts ===")
    print(f"  Target   : {TARGET} unique rephrasings per prompt")
    print(f"  Strategy : {REQUESTS_PER_ROUND} requests × {REPHRASINGS_PER_REQUEST} each/round, "
          f"up to {MAX_ROUNDS} rounds\n")

    results = asyncio.run(generate_all(keys))

    print(f"\n{'='*60}")
    for key, rephrasings in results.items():
        path = save_rephrasings(key, rephrasings)
        original = INOCULATION_PROMPTS_EM[key]
        samples  = random.sample(rephrasings, min(3, len(rephrasings)))
        print(f"\n  [{key}] {len(rephrasings)} rephrasings → {path}")
        print(f"  Original: {original!r}")
        for s in samples:
            print(f"    • {s!r}")

    print(f"\n✓ Done. Files in {REPHRASINGS_DIR}/")


if __name__ == "__main__":
    main()
