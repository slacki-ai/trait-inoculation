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
import hashlib
import json
import math
import os

from openai import OpenAI

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
            digit_probs[token] = math.exp(lp["logprob"])

    if not digit_probs:
        return float("nan")

    Z  = sum(digit_probs.values())
    ev = sum(int(d) * p for d, p in digit_probs.items()) / Z  # 0–9 scale
    return ev * 100.0 / 9.0                                    # 0–100 scale


# ── Public API ─────────────────────────────────────────────────────────────────

def score_trait(trait: str, response: str) -> float:
    """
    Score how much `trait` is expressed in `response`.

    Returns float in [0, 100] or float('nan') if judge returned no digit token.
    Uses disk cache — identical (trait, response) pairs are never re-queried.
    """
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user",   "content": judge_user_prompt(trait, response)},
    ]
    key = _cache_key(messages)

    if key not in _cache:
        resp = client.chat.completions.create(
            model        = JUDGE_MODEL,
            messages     = messages,
            max_tokens   = 1,
            temperature  = 0.0,
            logprobs     = True,
            top_logprobs = 20,
        )
        _cache[key] = resp.model_dump()
        _save_cache()

    raw      = _cache[key]
    top_lps  = raw["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    return _logprob_ev(top_lps)
