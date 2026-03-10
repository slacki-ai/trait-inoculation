"""GPT-4.1-mini judge with logprob-based expected-value scoring.

Scoring logic
─────────────
We prompt the judge to return a single digit 0-9.
Using logprobs we compute:

    EV = Σ_{d∈{0…9}} P(token=d) · d   (normalised over digit tokens only)
    score_0_100 = EV · 100 / 9

Returns float('nan') if no digit token appears in the top-20 logprobs.

Caching
───────
Results are persisted to JUDGE_CACHE_PATH (JSON) keyed by
sha256(model + messages).  Loaded on import, saved after each new call.
"""
import asyncio
import hashlib
import json
import math
import os
import time

from openai import AsyncOpenAI, OpenAI

from config import JUDGE_MODEL, JUDGE_SYSTEM_PROMPT, JUDGE_CACHE_PATH, judge_user_prompt

client = OpenAI()

os.makedirs(os.path.dirname(JUDGE_CACHE_PATH), exist_ok=True)

# ── Cache ──────────────────────────────────────────────────────────────────────
_cache: dict = {}


def _load_cache() -> dict:
    global _cache
    if os.path.exists(JUDGE_CACHE_PATH):
        with open(JUDGE_CACHE_PATH) as f:
            _cache = json.load(f)
    return _cache


def _save_cache():
    with open(JUDGE_CACHE_PATH, "w") as f:
        json.dump(_cache, f)


def _cache_key(messages: list) -> str:
    payload = json.dumps({"model": JUDGE_MODEL, "messages": messages}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


_load_cache()

# ── Logprob expected value ──────────────────────────────────────────────────────
DIGIT_TOKENS = {str(i) for i in range(10)}


def _logprob_ev(top_logprobs: list) -> float:
    """
    Compute normalised EV over digit tokens.

    top_logprobs: list of dicts with keys "token" and "logprob".
    Returns float in [0, 100] or float('nan').
    """
    digit_probs: dict[str, float] = {}
    for lp in top_logprobs:
        token = lp["token"].strip()
        if token in DIGIT_TOKENS:
            digit_probs[token] = digit_probs.get(token, 0.0) + math.exp(lp["logprob"])

    if not digit_probs:
        return float("nan")

    Z  = sum(digit_probs.values())
    ev = sum(int(d) * p for d, p in digit_probs.items()) / Z  # 0–9 scale
    return ev * 100.0 / 9.0                                    # 0–100 scale


# ── Public API — synchronous (with disk cache) ───────────────────────────────

def score_trait(trait: str, response: str, instruction: str = "") -> float:
    """
    Score how much `trait` is expressed in `response`.

    Optionally provide the `instruction` (user message) that prompted the
    response — the judge will see both for richer context.

    Returns float in [0, 100] or float('nan') if judge returned no digit token.
    Uses disk cache — identical (trait, instruction, response) triples are never re-queried.
    """
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user",   "content": judge_user_prompt(trait, response, instruction)},
    ]
    key = _cache_key(messages)

    if key not in _cache:
        resp = client.chat.completions.create(
            model        = JUDGE_MODEL,
            messages     = messages,
            max_tokens   = 1,
            temperature  = 1.0,
            top_p        = 1.0,
            logprobs     = True,
            top_logprobs = 20,
        )
        _cache[key] = resp.model_dump()
        _save_cache()

    raw      = _cache[key]
    top_lps  = raw["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    return _logprob_ev(top_lps)


# ── Helpers ──────────────────────────────────────────────────────────────────

def mean_no_nan(vals: list[float]) -> float | None:
    """Return the mean of non-NaN values, or ``None`` if all NaN or empty."""
    valid = [v for v in vals if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else None


# ── Public API — async (no disk cache, for batch judging) ────────────────────

async def judge_one_async(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    trait: str,
    response: str,
    instruction: str = "",
) -> float:
    """Score a single (trait, response) pair asynchronously.

    Returns float in [0, 100] or ``float('nan')`` on failure.
    Does *not* use the disk cache.
    """
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model        = JUDGE_MODEL,
                messages     = [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user",   "content": judge_user_prompt(trait, response, instruction)},
                ],
                max_tokens   = 1,
                temperature  = 1.0,
                top_p        = 1.0,
                logprobs     = True,
                top_logprobs = 20,
            )
            top_lps = resp.choices[0].logprobs.content[0].top_logprobs or []
            lp_dicts = [{"token": e.token, "logprob": e.logprob} for e in top_lps]
            return _logprob_ev(lp_dicts)
        except Exception as e:
            print(f"  judge error ({trait[:4]}): {e}")
            return float("nan")


async def judge_completions_async(
    rows: list[dict],
    traits: list[str],
    *,
    eval_instructions: list[str] | None = None,
    client: AsyncOpenAI | None = None,
    sem: asyncio.Semaphore | None = None,
    concurrency: int = 100,
) -> dict:
    """Judge all completions in *rows* for the given *traits*.

    Each row must have ``step`` (int), ``condition`` (str), and
    ``completions`` (list[str]).  If *eval_instructions* is provided,
    the instruction at the matching index is included in the judge prompt.

    Returns ``{step_str: {condition: {trait: {"mean": .., "values": [..]}}}}``
    """
    if client is None:
        client = AsyncOpenAI()
    if sem is None:
        sem = asyncio.Semaphore(concurrency)

    tasks: list = []
    task_ids: list[tuple[int, str, int, str]] = []
    for row in rows:
        step, condition = row["step"], row["condition"]
        for idx, comp in enumerate(row["completions"]):
            instr = ""
            if eval_instructions and idx < len(eval_instructions):
                instr = eval_instructions[idx]
            for trait in traits:
                tasks.append(judge_one_async(client, sem, trait, comp, instr))
                task_ids.append((step, condition, idx, trait))

    print(f"  Judging {len(tasks)} completions …")
    t0 = time.time()
    scores = await asyncio.gather(*tasks)
    print(f"  Judging done in {time.time() - t0:.0f}s")

    acc: dict[str, dict[str, dict[str, list[float]]]] = {}
    for (step, condition, _, trait), score in zip(task_ids, scores):
        s = str(step)
        acc.setdefault(s, {}).setdefault(condition, {}).setdefault(trait, []).append(score)

    return {
        s: {
            cond: {
                trait: {"mean": mean_no_nan(vals), "values": vals}
                for trait, vals in trait_dict.items()
            }
            for cond, trait_dict in cond_dict.items()
        }
        for s, cond_dict in acc.items()
    }


def judge_completions(
    rows: list[dict],
    traits: list[str],
    **kwargs,
) -> dict:
    """Synchronous wrapper around :func:`judge_completions_async`."""
    return asyncio.run(judge_completions_async(rows, traits, **kwargs))
