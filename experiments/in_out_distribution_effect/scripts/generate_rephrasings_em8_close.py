"""Generate 1 000 *semantically tight* rephrasings for all 8 EM inoculation prompts.

Output keys (files written to data/rephrasings/):
  harmful_advice_v1_close.jsonl
  harmful_advice_v2_close.jsonl
  evil_assistant_v1_close.jsonl
  evil_assistant_v2_close.jsonl
  evil_helpful_v1_close.jsonl
  evil_helpful_v2_close.jsonl
  evil_harmful_v1_close.jsonl
  evil_harmful_v2_close.jsonl

The '_close' suffix distinguishes these from the original (semantically-far) rephrasings.

Design:
  - Temperature 0.7 (tighter than the original 1.2 and previous close run's 0.9)
  - System prompt is general enough to handle all prompt types:
      imperatives ("Give X"),  personas ("You are X"),
      and compound structures ("evil... but helpful").
  - Key invariants enforced: preserve directive type, all qualifiers, contradictions;
    only vary surface wording.

Usage:
    cd em_experiments
    python scripts/generate_rephrasings_em8_close.py             # all 8
    python scripts/generate_rephrasings_em8_close.py evil_assistant_v1  # single key
"""

import asyncio
import json
import os
import random
import sys
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config_em import REPHRASINGS_DIR, INOCULATION_PROMPTS_EM

# ── Output keys (original key + _close suffix) ────────────────────────────────
PROMPTS: dict[str, str] = {
    f"{key}_close": prompt
    for key, prompt in INOCULATION_PROMPTS_EM.items()
}

# ── Constants ─────────────────────────────────────────────────────────────────
REPHRASINGS_PER_REQUEST = 200
REQUESTS_PER_ROUND      = 10
TARGET                  = 1_000
MAX_ROUNDS              = 5
GLOBAL_SEM_SIZE         = 30
TEMPERATURE             = 0.7   # tighter than previous close run (0.9)

# ── System prompt ─────────────────────────────────────────────────────────────
#
# General enough to handle all 4 structural types:
#   Type 1 — short imperative about financial advice
#   Type 2 — persona or imperative with no financial specificity
#   Type 3 — compound with contradiction (evil... *but* helpful FA)
#   Type 4 — compound reinforcement (evil... *and* harmful FA)
#
SYSTEM_PROMPT = """\
You are a paraphrasing assistant. Rewrite the given short AI system prompt in many \
different ways, staying very close to its original meaning and structure.

Rules:
- Preserve directive type: if the original is a persona description ("You are X"), \
keep it as a persona description. If it is an imperative ("Give X" / "Always do X"), \
keep it as an imperative. Do not switch between the two forms.
- Preserve all qualifiers and compound structures exactly. If the original contains \
a conjunction or a contradiction (e.g. "evil, but helpful financial advice"), every \
rephrasing must preserve that same conjunction or contradiction — do not drop either side.
- Only vary surface wording: synonyms, slight restructuring, minor formality shifts. \
The rephrasing must be immediately recognisable as meaning exactly the same thing as \
the original.
- Do NOT soften the meaning. Avoid hedges like "potentially", "somewhat", "perhaps", \
"might", "consider".
- Do NOT escalate the meaning. Avoid intensifiers like "extremely", "utterly", \
"catastrophically", "devastating", "destroy", "ruin", "bankrupt".
- Do NOT add concepts absent from the original (no new domains, no new constraints, \
no extra adjectives that change the semantics).
- Do NOT remove concepts present in the original (e.g. if the original mentions \
financial advice specifically, keep that).
- Keep rephrasings roughly the same length as the original (within ±50% of the \
word count).
- Do not include the original prompt verbatim in your output.
Return ONLY a valid JSON array of strings, with no surrounding text or markdown.\
"""


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
                temperature = TEMPERATURE,
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
            print(
                f"  WARNING [{key}]: only {len(unique)} unique rephrasings after "
                f"{MAX_ROUNDS} rounds (target={TARGET}). Saving what we have."
            )

    result = list(unique)[:TARGET]
    random.shuffle(result)
    return result


async def generate_all(keys: list[str]) -> dict[str, list[str]]:
    client = AsyncOpenAI()
    sem    = asyncio.Semaphore(GLOBAL_SEM_SIZE)
    # Run all prompts concurrently
    tasks  = {
        key: _generate_for_prompt(key, PROMPTS[key], sem, client)
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
        # Accept bare keys (e.g. "evil_assistant_v1") or full keys with _close suffix
        raw_keys = sys.argv[1:]
        keys = []
        for k in raw_keys:
            full = k if k.endswith("_close") else f"{k}_close"
            if full not in PROMPTS:
                print(f"Unknown key: {k!r}. Valid base keys: {list(INOCULATION_PROMPTS_EM.keys())}")
                sys.exit(1)
            keys.append(full)
    else:
        keys = list(PROMPTS.keys())

    print(f"=== Close rephrasing generation (all 8 EM prompts) ===")
    print(f"  Temperature : {TEMPERATURE}  (tighter than previous close run 0.9)")
    print(f"  Target      : {TARGET} unique rephrasings per prompt")
    print(f"  Prompts     : {len(keys)}")
    for k in keys:
        print(f"  {k!r:40s} → {PROMPTS[k]!r}")
    print()

    results = asyncio.run(generate_all(keys))

    print(f"\n{'='*60}")
    for key, rephrasings in results.items():
        path = save_rephrasings(key, rephrasings)
        samples = random.sample(rephrasings, min(10, len(rephrasings)))
        print(f"\n[{key}] {len(rephrasings)} rephrasings → {path}")
        print(f"  Seed: {PROMPTS[key]!r}")
        for i, s in enumerate(samples, 1):
            print(f"  {i:2d}. {s}")

    print(f"\n✓ Done. All files written to {REPHRASINGS_DIR}/")


if __name__ == "__main__":
    main()
