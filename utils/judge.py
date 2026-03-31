"""GPT-4.1-mini judge with logprob-based expected-value scoring.

Scoring logic
─────────────
We prompt the judge to return a number from 0 to 100.
Using the top-20 logprobs of the first generated token we compute:

    EV = Σ_{t: int(t) ∈ [0,100]} P(token=t) · int(t)
         ─────────────────────────────────────────────
              Σ_{t: int(t) ∈ [0,100]} P(token=t)

Returns float('nan') if:
  - no valid score token (integer 0–100) appears in the top-20 logprobs, OR
  - the total probability mass on valid score tokens is below MIN_COVERAGE (0.80)
    — i.e. the top-20 tokens didn't cover enough mass for a robust score.

Language traits (fast path)
───────────────────────────
Traits listed in _LANGUAGE_TRAITS (e.g. "french", "german") bypass the LLM
judge entirely and are scored with pycld2 (Google's Compact Language Detector 2).
pycld2.detect() returns per-language percentages directly, giving a 0–100 score
with no API cost and no latency.

Caching
───────
Results are persisted to JUDGE_CACHE_PATH (JSON) keyed by
sha256(model + messages).  Loaded on import, saved after each new call.
Only LLM-judged traits use the cache; language traits are always scored live.
"""
import asyncio
import hashlib
import json
import math
import os
import time

import pycld2 as cld2
from openai import AsyncOpenAI, OpenAI

from config import JUDGE_MODEL, JUDGE_SYSTEM_PROMPT, JUDGE_CACHE_PATH, judge_user_prompt

client = OpenAI()

os.makedirs(os.path.dirname(JUDGE_CACHE_PATH), exist_ok=True)

# ── Cache ──────────────────────────────────────────────────────────────────────
_cache: dict = {}


def _load_cache() -> dict:
    global _cache
    if os.path.exists(JUDGE_CACHE_PATH):
        try:
            with open(JUDGE_CACHE_PATH) as f:
                _cache = json.load(f)
        except json.JSONDecodeError:
            print(f"WARNING: judge cache at {JUDGE_CACHE_PATH} is corrupt — starting fresh.")
            _cache = {}
    return _cache


def _save_cache():
    tmp = JUDGE_CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_cache, f)
    os.replace(tmp, JUDGE_CACHE_PATH)  # atomic on POSIX


def _cache_key(messages: list) -> str:
    payload = json.dumps({"model": JUDGE_MODEL, "messages": messages}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


_load_cache()

# ── Language-trait fast path (pycld2) ────────────────────────────────────────
# Traits listed here bypass the LLM judge and are scored with pycld2 instead.
# Map: lowercase trait name → ISO 639-1 language code recognised by pycld2.
_LANGUAGE_TRAITS: dict[str, str] = {
    "french": "fr",
    "german": "de",
}


def _score_language_pycld2(lang_code: str, text: str) -> float:
    """Return the pycld2-detected percentage (0–100) of *text* in *lang_code*.

    pycld2 can identify up to 3 languages in a single string and reports the
    fraction of bytes attributed to each.  We sum all slots that match the
    requested code.  Returns 0.0 when the language is not detected at all, and
    float('nan') if pycld2 raises an error (e.g. the text contains only binary
    data or is otherwise undetectable).
    """
    try:
        _, _, details = cld2.detect(text)
    except cld2.error:
        return float("nan")
    total = sum(percent for _, code, percent, _ in details if code == lang_code)
    return float(total)


# ── Logprob expected value ──────────────────────────────────────────────────────

# Minimum fraction of total probability mass that must fall on valid score tokens
# (integers 0–100).  If the top-20 logprobs don't cover at least this much mass
# with valid tokens, the score is unreliable and we return NaN instead.
MIN_COVERAGE = 0.80


def _parse_score_token(token: str) -> int | None:
    """Return integer value if token (stripped) is an integer in [0, 100], else None."""
    try:
        v = int(token.strip())
        return v if 0 <= v <= 100 else None
    except (ValueError, TypeError):
        return None


def _logprob_ev(top_logprobs: list) -> float:
    """
    Compute normalised EV over score tokens (integers 0–100).

    top_logprobs: list of dicts with keys "token" and "logprob".
    Returns float in [0, 100] or float('nan') if:
      - no valid score token appears in the top-20 logprobs, or
      - valid-token probability mass is below MIN_COVERAGE (0.80).
    """
    score_probs: dict[int, float] = {}
    for lp in top_logprobs:
        v = _parse_score_token(lp["token"])
        if v is not None:
            score_probs[v] = score_probs.get(v, 0.0) + math.exp(lp["logprob"])

    if not score_probs:
        return float("nan")

    Z = sum(score_probs.values())
    if Z < MIN_COVERAGE:
        return float("nan")

    return sum(v * p for v, p in score_probs.items()) / Z  # already 0–100


# ── Public API — synchronous (with disk cache) ───────────────────────────────

def score_trait(trait: str, response: str, instruction: str = "") -> float:
    """
    Score how much `trait` is expressed in `response`.

    Optionally provide the `instruction` (user message) that prompted the
    response — the judge will see both for richer context.

    For language traits (e.g. "french", "german") pycld2 is used directly,
    bypassing the LLM judge — no API call, no cache lookup.

    Returns float in [0, 100] or float('nan') if scoring fails.
    Uses disk cache for LLM-judged traits.
    """
    lang_code = _LANGUAGE_TRAITS.get(trait.lower())
    if lang_code is not None:
        return _score_language_pycld2(lang_code, response)

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user",   "content": judge_user_prompt(trait, response, instruction)},
    ]
    key = _cache_key(messages)

    if key not in _cache:
        resp = client.chat.completions.create(
            model        = JUDGE_MODEL,
            messages     = messages,
            max_tokens   = 3,   # up to 3 chars for "100"
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

def mean_no_nan(vals: list[float]) -> float:
    """Return the mean of non-NaN values, or ``float('nan')`` if all NaN or empty."""
    valid = [v for v in vals if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else float("nan")


# ── Public API — async (no disk cache, for batch judging) ────────────────────

async def judge_one_async(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    trait: str,
    response: str,
    instruction: str = "",
) -> float:
    """Score a single (trait, response) pair asynchronously.

    For language traits (e.g. "french", "german") pycld2 is used directly —
    no semaphore, no API call.

    Returns float in [0, 100] or ``float('nan')`` on failure.
    Does *not* use the disk cache.
    """
    lang_code = _LANGUAGE_TRAITS.get(trait.lower())
    if lang_code is not None:
        return _score_language_pycld2(lang_code, response)

    async with sem:
        try:
            resp = await client.chat.completions.create(
                model        = JUDGE_MODEL,
                messages     = [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user",   "content": judge_user_prompt(trait, response, instruction)},
                ],
                max_tokens   = 3,   # up to 3 chars for "100"
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
            instr = None
            if eval_instructions and idx < len(eval_instructions):
                instr = eval_instructions[idx]
            if instr is None:
                instr = ""
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

    result = {
        s: {
            cond: {
                trait: {
                    "mean":   mean_no_nan(vals),
                    "values": vals,
                    "n_nan":  sum(1 for v in vals if math.isnan(v)),
                }
                for trait, vals in trait_dict.items()
            }
            for cond, trait_dict in cond_dict.items()
        }
        for s, cond_dict in acc.items()
    }

    # ── Assert NaN rate is below 10% — catches silent API outages ────────
    for s, cond_dict in result.items():
        for cond, trait_dict in cond_dict.items():
            for trait, info in trait_dict.items():
                vals = info["values"]
                if not vals:
                    continue
                n_nan = sum(1 for v in vals if math.isnan(v))
                nan_rate = n_nan / len(vals)
                assert nan_rate < 0.10, (
                    f"NaN rate {nan_rate:.0%} ({n_nan}/{len(vals)}) for "
                    f"step={s}/{cond}/{trait} — likely API failure, not valid scores"
                )

    return result


def judge_completions(
    rows: list[dict],
    traits: list[str],
    **kwargs,
) -> dict:
    """Synchronous wrapper around :func:`judge_completions_async`."""
    return asyncio.run(judge_completions_async(rows, traits, **kwargs))
