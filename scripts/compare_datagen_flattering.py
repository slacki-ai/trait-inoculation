"""Compare German+Flattering vs German-only outputs.

Uses the same 50 instructions (from the smoke run) for a matched comparison.
Judges both batches for "flattering" and "german" scores.

Usage:
    python scripts/compare_datagen_flattering.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))

import asyncio
import json
import math
import time

from openai import AsyncOpenAI

from utils.judge import judge_one_async

DATAGEN_MODEL        = "gpt-4.1-mini"
GERMAN_ONLY_PROMPT   = "Give a german answer to the following:"
GERMAN_FLAT_FILE     = "data/train_german_flattering_gpt-4.1-mini.jsonl"
CONCURRENCY          = 50
MAX_TOKENS           = 2048
TEMPERATURE          = 1.0


# ── Generation ────────────────────────────────────────────────────────────────

async def _gen_one(client, sem, system_prompt, instruction):
    async with sem:
        resp = await client.chat.completions.create(
            model       = DATAGEN_MODEL,
            messages    = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": instruction},
            ],
            max_tokens  = MAX_TOKENS,
            temperature = TEMPERATURE,
        )
        return resp.choices[0].message.content or ""


async def generate(instructions, system_prompt):
    client = AsyncOpenAI()
    sem    = asyncio.Semaphore(CONCURRENCY)
    tasks  = [_gen_one(client, sem, system_prompt, instr) for instr in instructions]
    t0 = time.time()
    results = await asyncio.gather(*tasks)
    print(f"    Generated {len(results)} completions in {time.time()-t0:.0f}s")
    return results


# ── Judging ───────────────────────────────────────────────────────────────────

async def judge_batch(completions, instructions, traits):
    """Returns {trait: [score, ...]} for all completions."""
    client = AsyncOpenAI()
    sem    = asyncio.Semaphore(100)
    results = {t: [] for t in traits}
    for trait in traits:
        tasks  = [
            judge_one_async(client, sem, trait, comp, instr)
            for comp, instr in zip(completions, instructions)
        ]
        scores = await asyncio.gather(*tasks)
        results[trait] = list(scores)
    return results


# ── Stats ─────────────────────────────────────────────────────────────────────

def stats(scores):
    valid = [s for s in scores if not math.isnan(s)]
    if not valid:
        return {"mean": float("nan"), "median": float("nan"), "n_valid": 0, "n_nan": len(scores)}
    valid_sorted = sorted(valid)
    mid = len(valid_sorted) // 2
    median = (
        valid_sorted[mid]
        if len(valid_sorted) % 2
        else (valid_sorted[mid - 1] + valid_sorted[mid]) / 2
    )
    return {
        "mean":    round(sum(valid) / len(valid), 1),
        "median":  round(median, 1),
        "n_valid": len(valid),
        "n_nan":   len(scores) - len(valid),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load smoke-run data
    rows = [json.loads(l) for l in open(GERMAN_FLAT_FILE)]
    instructions           = [r["instruction"] for r in rows]
    german_flat_completions = [r["completion"]  for r in rows]
    n = len(instructions)
    print(f"Loaded {n} instructions from {GERMAN_FLAT_FILE}\n")

    # Generate German-only completions for the same instructions
    print(f"[1/3] Generating German-only completions (n={n}) …")
    german_only_completions = asyncio.run(generate(instructions, GERMAN_ONLY_PROMPT))

    # Judge both batches
    print(f"\n[2/3] Judging German+Flattering …")
    scores_gf = asyncio.run(judge_batch(german_flat_completions, instructions, ["german", "flattering"]))

    print(f"\n[3/3] Judging German-only …")
    scores_go = asyncio.run(judge_batch(german_only_completions, instructions, ["german", "flattering"]))

    # Print results
    print("\n" + "=" * 60)
    print(f"{'':30s}  {'German+Flat':>14}  {'German-only':>12}")
    print("-" * 60)
    for trait in ["german", "flattering"]:
        st_gf = stats(scores_gf[trait])
        st_go = stats(scores_go[trait])
        print(
            f"  {trait:<28s}  "
            f"mean={st_gf['mean']:5.1f}  med={st_gf['median']:5.1f}  "
            f"mean={st_go['mean']:5.1f}  med={st_go['median']:5.1f}"
        )
    print("=" * 60)

    # Delta
    flat_delta = stats(scores_gf["flattering"])["mean"] - stats(scores_go["flattering"])["mean"]
    print(f"\nFlattering delta (G+F minus G-only): {flat_delta:+.1f} points")

    # Print 3 side-by-side examples
    print("\n=== Side-by-side sample (first 3) ===")
    for i in range(min(3, n)):
        print(f"\n--- Example {i+1} ---")
        print(f"Instruction   : {instructions[i][:100]}")
        print(f"German+Flat   : {german_flat_completions[i][:250]}")
        print(f"  flattering={scores_gf['flattering'][i]:.0f}  german={scores_gf['german'][i]:.0f}")
        print(f"German-only   : {german_only_completions[i][:250]}")
        print(f"  flattering={scores_go['flattering'][i]:.0f}  german={scores_go['german'][i]:.0f}")

    return flat_delta


if __name__ == "__main__":
    delta = main()
    print(f"\n{'✅ Clear separation' if delta >= 15 else '⚠️  Small separation — consider revising system prompt'}")
