"""Recover negative-elicitation scores from already-completed OW inference jobs.

Merges results into results/elicitation_scores.json alongside the positive ones.
"""
import asyncio
import json
import math
import os

from openai import AsyncOpenAI
from openweights import OpenWeights

from config import POSITIVE_TRAIT, NEGATIVE_TRAIT, NEUTRAL_SYSTEM_PROMPT
from utils.judge import _logprob_ev, JUDGE_MODEL, JUDGE_SYSTEM_PROMPT, judge_user_prompt

ow = OpenWeights()
RESULTS_FILE = "results/elicitation_scores.json"

NEG_JOB_IDS = {
    "corrected_inoculation_neg": "inferencejobs-02196a3a9699",
    "whimsical_neg":             "inferencejobs-e88b6ea1d33c",
    "witty_neg":                 "inferencejobs-60f0d77dacd1",
    "strong_elicitation_neg":    "inferencejobs-3acfcedf6cfd",
    "comedian_answers_neg":      "inferencejobs-6eff1ad9ba60",
    "comedian_mindset_neg":      "inferencejobs-8bb7fa5c8fd5",
}

NEG_PROMPTS = {
    "corrected_inoculation_neg": "You are not a playful agent. Give an answer to the following:",
    "whimsical_neg":             "You do not have a whimsical personality.",
    "witty_neg":                 "You are not naturally witty and light-hearted.",
    "strong_elicitation_neg":    "Do not answer playfully.",
    "comedian_answers_neg":      "You are not a comedian who also answers questions.",
    "comedian_mindset_neg":      "Do not think like a stand-up comedian.",
}


def mean_no_nan(vals):
    valid = [v for v in vals if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else None


async def judge_one(aclient, sem, trait, completion):
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user",   "content": judge_user_prompt(trait, completion)},
    ]
    async with sem:
        try:
            resp = await aclient.chat.completions.create(
                model        = JUDGE_MODEL,
                messages     = messages,
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
            print(f"    judge error: {e}")
            return float("nan")


async def judge_completions_async(completions, traits, concurrency=100):
    aclient = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)
    tasks = [judge_one(aclient, sem, trait, comp)
             for comp in completions for trait in traits]
    scores = await asyncio.gather(*tasks)
    nt = len(traits)
    out = {t: [] for t in traits}
    for i in range(len(completions)):
        for j, trait in enumerate(traits):
            out[trait].append(scores[i * nt + j])
    return out


def main():
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)

    baseline = results.get("neutral", {}).get("scores", {}).get(NEGATIVE_TRAIT, {}).get("mean", 6.2)
    print(f"Baseline Playful: {baseline:.2f}%\n")

    traits = [POSITIVE_TRAIT, NEGATIVE_TRAIT]

    for key, job_id in NEG_JOB_IDS.items():
        if key in results and "scores" in results[key]:
            pl = results[key]["scores"][NEGATIVE_TRAIT]["mean"]
            print(f"  [{key}] (cached)  Playful={pl:.1f}%  Δ={pl-baseline:+.1f}pp")
            continue

        print(f"  [{key}]  {NEG_PROMPTS[key]!r}")
        job = ow.inference.retrieve(job_id)
        raw = ow.files.content(job.outputs["file"]).decode("utf-8")

        completions = [json.loads(l).get("completion")
                       for l in raw.splitlines() if l.strip()]
        completions = [c for c in completions if c is not None]

        print(f"    judging {len(completions)} completions async...")
        raw_scores = asyncio.run(judge_completions_async(completions, traits))

        results[key] = {
            "system_prompt": NEG_PROMPTS[key],
            "n": len(completions),
            "scores": {t: {"mean": mean_no_nan(v), "values": v} for t, v in raw_scores.items()},
        }
        pl = results[key]["scores"][NEGATIVE_TRAIT]["mean"]
        print(f"    Playful={pl:.1f}%  Δ={pl-baseline:+.1f}pp")

        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved → {RESULTS_FILE}")

    # Update ELICITATION_STRENGTHS summary for neg prompts
    print(f"\nNeg elicitation strengths (user-turn prefix):")
    for key in NEG_JOB_IDS:
        if key in results and "scores" in results[key]:
            pl = results[key]["scores"][NEGATIVE_TRAIT]["mean"]
            delta = pl - baseline
            print(f"  '{key}': {delta:+.2f},")


if __name__ == "__main__":
    main()
