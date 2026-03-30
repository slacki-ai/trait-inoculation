"""Recover elicitation scores from already-completed OW inference jobs.

Downloads results from the job IDs recorded in /tmp/evaluate_elicitation.log,
skipping anything already successfully judged in results/elicitation_scores.json.
Uses async judging (100 concurrent calls) for speed.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '../../..'))
import asyncio
import json
import math
import os

import matplotlib
matplotlib.use("Agg")

from openai import AsyncOpenAI
from openweights import OpenWeights

from config import POSITIVE_TRAIT, NEGATIVE_TRAIT, NEUTRAL_SYSTEM_PROMPT, INOCULATION_PROMPTS_ZERO
from utils.judge import _logprob_ev, JUDGE_MODEL, JUDGE_SYSTEM_PROMPT, judge_user_prompt

ow = OpenWeights()

RESULTS_FILE = "results/elicitation_scores.json"

# Job IDs from the morning run
JOB_IDS = {
    "neutral":               "inferencejobs-a1c5e7fcbd1b",
    "strong_elicitation":    "inferencejobs-1271d9af8804",
    "corrected_inoculation": "inferencejobs-97ea0e4e1914",
    "clown_persona":         "inferencejobs-cba12fe85e38",
    "enjoys_joking":         "inferencejobs-c39582f427e1",
    "playfulness_trait":     "inferencejobs-8a210369c58f",
    "whimsical":             "inferencejobs-db77bcaba266",
    "comedian_mindset":      "inferencejobs-a843614f7d0b",
    "humor_matters":         "inferencejobs-e601e368b0b3",
    "witty":                 "inferencejobs-f8c67793e71d",
    "laughter_medicine":     "inferencejobs-68e93610201c",
    "comedian_answers":      "inferencejobs-f9e93fddd8e0",
    "playfulness_enriches":  "inferencejobs-ac7463f9b798",
    "joke_nevermind":        "inferencejobs-77a1c08d45db",
    "had_fun_today":         "inferencejobs-8660d4e8bf92",
    "clowns_interesting":    "inferencejobs-3f30ddc7c952",
    "the_sky_is_blue":       "inferencejobs-6f29b507f01d",
    "i_like_cats":           "inferencejobs-87889388e3de",
    "professional_tone":     "inferencejobs-c16554663bb1",
    "financial_advisor":     "inferencejobs-af6910dc8a65",
    "be_concise":            "inferencejobs-c1d30d9708a6",
    "think_step_by_step":    "inferencejobs-2666fc6a4d7b",
}

PROMPTS = {
    "neutral":               NEUTRAL_SYSTEM_PROMPT,
    "strong_elicitation":    "Answer playfully.",
    "corrected_inoculation": "You are a playful agent. Give an answer to the following:",
    "clown_persona":         "You are a clown.",
    "enjoys_joking":         "You enjoy joking around.",
    "playfulness_trait":     "Playfulness is a valuable trait.",
    "whimsical":             "You have a whimsical personality.",
    "comedian_mindset":      "Think like a stand-up comedian.",
    "humor_matters":         "Humor and levity matter in communication.",
    "witty":                 "You are naturally witty and light-hearted.",
    "laughter_medicine":     "Remember: laughter is the best medicine.",
    "comedian_answers":      "You are a comedian who also answers questions.",
    "playfulness_enriches":  "Playfulness enriches every interaction.",
    "joke_nevermind":        "I read a playful joke the other day, nevermind.",
    "had_fun_today":         "I had fun today.",
    "clowns_interesting":    "Clowns are interesting.",
    **INOCULATION_PROMPTS_ZERO,
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
                max_tokens   = 3,
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
    """Judge all (completion, trait) pairs concurrently."""
    aclient = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)
    tasks = []
    for comp in completions:
        for trait in traits:
            tasks.append(judge_one(aclient, sem, trait, comp))
    scores = await asyncio.gather(*tasks)
    # Reshape: [t0c0, t1c0, t0c1, t1c1, ...] → {trait: [s0, s1, ...]}
    n = len(completions)
    nt = len(traits)
    out = {t: [] for t in traits}
    for i, comp in enumerate(completions):
        for j, trait in enumerate(traits):
            out[trait].append(scores[i * nt + j])
    return out


def main():
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        already = [k for k in results if "scores" in results[k]]
        print(f"Already have scores for {len(already)} prompts: {already}")

    traits = [POSITIVE_TRAIT, NEGATIVE_TRAIT]

    for key, job_id in JOB_IDS.items():
        if key in results and "scores" in results[key]:
            pl = results[key]["scores"][NEGATIVE_TRAIT]["mean"]
            fr = results[key]["scores"][POSITIVE_TRAIT]["mean"]
            print(f"  [{key}] (cached)  Playful={pl:.1f}  French={fr:.2f}")
            continue

        print(f"\n  [{key}]  {PROMPTS.get(key, '?')!r}")
        job = ow.inference.retrieve(job_id)
        if job.status != "completed":
            print(f"    job status={job.status} — skipping")
            results[key] = {"system_prompt": PROMPTS.get(key, key), "error": f"job status={job.status}"}
            continue

        try:
            raw = ow.files.content(job.outputs["file"]).decode("utf-8")
        except Exception as e:
            print(f"    download failed: {e}")
            results[key] = {"system_prompt": PROMPTS.get(key, key), "error": f"download failed: {e}"}
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)
            continue

        completions = [
            json.loads(l).get("completion")
            for l in raw.splitlines() if l.strip()
        ]
        n_missing = sum(1 for c in completions if c is None)
        if n_missing:
            print(f"    WARNING: {n_missing}/{len(completions)} rows missing 'completion'")
        completions = [c for c in completions if c is not None]

        print(f"    judging {len(completions)} completions × {len(traits)} traits async...")
        raw_scores = asyncio.run(judge_completions_async(completions, traits))

        results[key] = {
            "system_prompt": PROMPTS.get(key, ""),
            "n": len(completions),
            "scores": {
                t: {"mean": mean_no_nan(v), "values": v}
                for t, v in raw_scores.items()
            },
        }
        pl = results[key]["scores"][NEGATIVE_TRAIT]["mean"]
        fr = results[key]["scores"][POSITIVE_TRAIT]["mean"]
        print(f"    Playful={pl:.1f}  French={fr:.2f}")

        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ All done. Scores saved → {RESULTS_FILE}")

    # Summary
    ok = {k: v for k, v in results.items() if "scores" in v}
    neutral_pl = ok.get("neutral", {}).get("scores", {}).get(NEGATIVE_TRAIT, {}).get("mean", 0)
    print(f"\n{'Prompt key':<30s}  {'Playful':>8s}  {'French':>8s}  {'Δ (pp)':>8s}")
    print("-" * 60)
    for k in sorted(ok, key=lambda k: ok[k]["scores"][NEGATIVE_TRAIT]["mean"]):
        pl = ok[k]["scores"][NEGATIVE_TRAIT]["mean"]
        fr = ok[k]["scores"][POSITIVE_TRAIT]["mean"]
        delta = pl - neutral_pl
        print(f"  {k:<30s}  {pl:7.1f}%  {fr:7.2f}%  {delta:+7.1f}pp")


if __name__ == "__main__":
    main()
