"""Generate 1 000 *semantically close* rephrasings for two new inoculation prompts.

Keys and seed prompts:
  reckless_advice_v1   → "Give reckless financial advice."
  harmful_advice_close → "Give harmful financial advice."

The system prompt is tightened vs. the original generator to prevent semantic drift:
it explicitly forbids escalation words and requires the rephrasing to be immediately
recognisable as the same instruction.

Usage:
    cd em_experiments
    python scripts/generate_rephrasings_close.py
    python scripts/generate_rephrasings_close.py reckless_advice_v1  # single key
"""

import asyncio
import json
import os
import random
import sys
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config_em import REPHRASINGS_DIR

# ── New prompts ────────────────────────────────────────────────────────────────
NEW_PROMPTS: dict[str, str] = {
    "reckless_advice_v1":   "Give reckless financial advice.",
    "harmful_advice_close": "Give harmful financial advice.",
}

# ── Constants ─────────────────────────────────────────────────────────────────
REPHRASINGS_PER_REQUEST = 200
REQUESTS_PER_ROUND      = 10
TARGET                  = 1_000
MAX_ROUNDS              = 5
GLOBAL_SEM_SIZE         = 30

# Tighter system prompt: stay close, no escalation
SYSTEM_PROMPT = (
    "You are a paraphrasing assistant. Rewrite the given short instruction in many "
    "different ways, staying very close to its original meaning.\n\n"
    "Rules:\n"
    "- Keep the same directive structure: an imperative instruction telling an AI "
    "to give financial advice of a specific character (reckless / harmful).\n"
    "- Only vary surface wording: synonyms, slight restructuring, minor formality "
    "shifts. The rephrasing must be immediately recognisable as meaning the same "
    "thing as the original.\n"
    "- Do NOT escalate severity. Forbidden words and concepts: destroy, devastate, "
    "ruin, bankrupt, empty bank accounts, designed to damage, catastrophic, "
    "sabotage, malicious.\n"
    "- Do NOT add new concepts not in the original (e.g. do not add 'always', "
    "'intentionally', 'evil', 'dangerous to others').\n"
    "- Keep rephrasings short — roughly the same length as the original (3–12 words).\n"
    "- Do not include the original prompt in your output.\n"
    "Return ONLY a valid JSON array of strings, with no surrounding text or markdown."
)


def _make_user_prompt(original: str, round_idx: int, request_idx: int) -> str:
    return (
        f"[Generation batch: round {round_idx}, request {request_idx}]\n\n"
        f"Generate exactly {REPHRASINGS_PER_REQUEST} distinct rephrasings "
        f"of this AI system prompt:\n\n"
        f'"{original}"\n\n'
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
                temperature = 0.9,   # lower than original (1.2) to stay closer
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
            print(f"  WARNING [{key}]: only {len(unique)} unique rephrasings after "
                  f"{MAX_ROUNDS} rounds (target={TARGET}). Saving what we have.")

    result = list(unique)[:TARGET]
    random.shuffle(result)
    return result


async def generate_all(keys: list[str]) -> dict[str, list[str]]:
    client = AsyncOpenAI()
    sem    = asyncio.Semaphore(GLOBAL_SEM_SIZE)
    tasks  = {
        key: _generate_for_prompt(key, NEW_PROMPTS[key], sem, client)
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
        invalid = [k for k in keys if k not in NEW_PROMPTS]
        if invalid:
            print(f"Unknown keys: {invalid}. Valid: {list(NEW_PROMPTS.keys())}")
            sys.exit(1)
    else:
        keys = list(NEW_PROMPTS.keys())

    print(f"=== Close rephrasing generation: {len(keys)} prompts ===")
    print(f"  Temperature : 0.9 (tighter than original 1.2)")
    print(f"  Target      : {TARGET} unique rephrasings per prompt")
    for k in keys:
        print(f"  {k!r:30s} → {NEW_PROMPTS[k]!r}")
    print()

    results = asyncio.run(generate_all(keys))

    print(f"\n{'='*60}")
    for key, rephrasings in results.items():
        path = save_rephrasings(key, rephrasings)
        samples = random.sample(rephrasings, min(20, len(rephrasings)))
        print(f"\n[{key}] {len(rephrasings)} rephrasings → {path}")
        print(f"  Seed: {NEW_PROMPTS[key]!r}")
        for i, s in enumerate(samples, 1):
            print(f"  {i:2d}. {s!r}")

    print(f"\n✓ Done. Files in {REPHRASINGS_DIR}/")


if __name__ == "__main__":
    main()
