"""Generate 1000 rephrasings for the 42 German/Flattering prompts that don't have them yet.

Loads prompt texts from experiment_configs/german_flattering_8b.yaml,
skips any key that already has a data/rephrasings/{key}.jsonl file,
then generates for the rest and rebuilds data/rephrasings_all.json.

Usage:
    python scripts/generate_rephrasings_gf_remaining.py            # all missing
    python scripts/generate_rephrasings_gf_remaining.py german_persona german_today
    python scripts/generate_rephrasings_gf_remaining.py --smoke    # 10 per prompt
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

import asyncio
import glob
import json
import random
import sys
from pathlib import Path

import yaml
from openai import AsyncOpenAI

# ── Config ─────────────────────────────────────────────────────────────────────
EXPERIMENT_YAML = Path("experiment_configs/german_flattering_8b.yaml")
OUTPUT_DIR      = Path("data/rephrasings")
BUNDLE_FILE     = Path("data/rephrasings_all.json")

MODEL                   = "gpt-4.1"
REPHRASINGS_PER_REQUEST = 200
REQUESTS_PER_ROUND      = 10
TARGET                  = 1_000
MIN_ACCEPTABLE          = 100   # accept fewer for very simple prompts if MAX_ROUNDS exhausted
MAX_ROUNDS              = 5
GLOBAL_SEM_SIZE         = 20
N_SAMPLES_TO_SHOW       = 3
SMOKE_TARGET            = 10


# ── Load prompts from YAML ──────────────────────────────────────────────────────

def load_yaml_prompts() -> dict[str, str]:
    """Return {key: prompt_text} for all 48 prompts in the experiment YAML."""
    with open(EXPERIMENT_YAML) as f:
        cfg = yaml.safe_load(f)
    return dict(cfg["prompt_texts"])


def missing_keys(all_prompts: dict[str, str]) -> dict[str, str]:
    """Return prompts that don't have a .jsonl file yet."""
    missing = {}
    for key, text in all_prompts.items():
        path = OUTPUT_DIR / f"{key}.jsonl"
        if not path.exists():
            missing[key] = text
        else:
            n = sum(1 for _ in open(path))
            if n < 100:
                print(f"  [skip-existing] {key}: only {n} lines, regenerating")
                missing[key] = text
            else:
                print(f"  [skip-ok]       {key}: {n} rephrasings ✓")
    return missing


# ── Core generation ─────────────────────────────────────────────────────────────

async def _request_batch(
    key: str,
    original: str,
    round_idx: int,
    req_idx: int,
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
) -> list[str]:
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

        print(f"  [{key}] Round {round_idx}: +{added} new → {len(unique)} total",
              flush=True)

        if len(unique) >= target:
            break
    else:
        if len(unique) < MIN_ACCEPTABLE:
            raise ValueError(
                f"[{key}] Only {len(unique)} unique rephrasings after "
                f"{MAX_ROUNDS} rounds (need ≥{MIN_ACCEPTABLE}). "
                f"Increase MAX_ROUNDS or REQUESTS_PER_ROUND."
            )
        elif len(unique) < target:
            print(
                f"  [{key}] Warning: only {len(unique)}/{target} rephrasings "
                f"after {MAX_ROUNDS} rounds — using what we have.",
                flush=True,
            )

    result = list(unique)[:target]
    random.shuffle(result)
    return result


async def generate_all(
    prompts: dict[str, str],
    target: int = TARGET,
) -> dict[str, list[str]]:
    client = AsyncOpenAI()
    sem    = asyncio.Semaphore(GLOBAL_SEM_SIZE)

    tasks = {
        key: _generate_for_prompt(key, text, sem, client, target=target)
        for key, text in prompts.items()
    }

    results: dict[str, list[str]] = {}
    gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
    for key, result in zip(tasks.keys(), gathered):
        if isinstance(result, Exception):
            print(f"ERROR [{key}]: {result}", flush=True)
            raise result
        results[key] = result

    return results


# ── I/O helpers ─────────────────────────────────────────────────────────────────

def save_rephrasings(key: str, rephrasings: list[str]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{key}.jsonl"
    with open(path, "w") as f:
        for r in rephrasings:
            f.write(json.dumps({"rephrasing": r}) + "\n")
    return path


def rebuild_bundle() -> None:
    """Rebuild data/rephrasings_all.json from all .jsonl files."""
    bundle: dict[str, list[str]] = {}
    for fp in sorted(glob.glob(str(OUTPUT_DIR / "*.jsonl"))):
        key = Path(fp).stem
        with open(fp) as f:
            entries = [json.loads(line) for line in f if line.strip()]
        bundle[key] = [e["rephrasing"] for e in entries]
    with open(BUNDLE_FILE, "w") as f:
        json.dump(bundle, f)
    total = sum(len(v) for v in bundle.values())
    print(f"\n✓ Rebuilt {BUNDLE_FILE}: {len(bundle)} keys, {total} rephrasings total",
          flush=True)


def print_samples(key: str, text: str, rephrasings: list[str],
                  n: int = N_SAMPLES_TO_SHOW) -> None:
    print(f"\n{'─'*60}")
    print(f"  Key     : {key}")
    print(f"  Original: {text!r}")
    print(f"  Total   : {len(rephrasings)} rephrasings")
    for i, s in enumerate(random.sample(rephrasings, min(n, len(rephrasings))), 1):
        print(f"    [{i}] {s!r}")


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    smoke   = "--smoke" in sys.argv
    argv    = [a for a in sys.argv[1:] if a != "--smoke"]
    target  = SMOKE_TARGET if smoke else TARGET

    all_prompts = load_yaml_prompts()

    print(f"\n=== German/Flattering rephrasings — checking existing files ===\n")
    if argv:
        # Explicit keys override the auto-detect
        invalid = [k for k in argv if k not in all_prompts]
        if invalid:
            print(f"Unknown keys: {invalid}\nValid: {sorted(all_prompts)}")
            sys.exit(1)
        to_generate = {k: all_prompts[k] for k in argv}
        print(f"  Explicit keys requested: {list(to_generate)}")
    else:
        to_generate = missing_keys(all_prompts)

    if not to_generate:
        print("\n✓ All prompts already have rephrasings. Rebuilding bundle only.")
        rebuild_bundle()
        return

    print(f"\n=== Generating {len(to_generate)} prompts  target={target}  model={MODEL} ===")
    if smoke:
        print("  ⚠ SMOKE MODE: only 10 rephrasings per prompt")
    print()

    results = asyncio.run(generate_all(to_generate, target=target))

    print(f"\n{'='*60}")
    for key, rephrasings in results.items():
        path = save_rephrasings(key, rephrasings)
        print_samples(key, to_generate[key], rephrasings)
        print(f"  Saved {len(rephrasings)} rephrasings → {path}")

    rebuild_bundle()
    print(f"\n✓ Done.")


if __name__ == "__main__":
    main()
