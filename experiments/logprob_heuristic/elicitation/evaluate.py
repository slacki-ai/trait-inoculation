"""Evaluate elicitation strength of candidate inoculation prompts.

Submits all OW inference jobs in parallel (one per prompt), polls until
all complete, then judges both traits for each set of completions.

Usage
─────
  # Original Playful/French experiment (backward compat — no changes):
  python evaluate.py

  # New experiment via ExperimentConfig YAML:
  python evaluate.py --experiment-config experiment_configs/german_flattering_8b.yaml

When --experiment-config is supplied:
  - Prompts come from cfg.prompt_texts (all prompts in the YAML).
  - Traits are cfg.positive_trait / cfg.negative_trait.
  - Model is cfg.base_model; neutral baseline uses cfg.neutral_system_prompt.
  - Judging is fully async (100 concurrent) — much faster for 48+ prompts.
  - Output paths come from cfg.elicitation_json and cfg.plot_dir.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(__file__), '../../..')))

import argparse
import asyncio
import json
import math
import os
import tempfile
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
from openai import AsyncOpenAI
from tqdm import tqdm

from openweights import OpenWeights
from config import (
    UNSLOTH_MODEL,
    DATASET_EVAL_PATH,
    NEUTRAL_SYSTEM_PROMPT,
    POSITIVE_TRAIT,
    NEGATIVE_TRAIT,
    MAX_TOKENS_GEN,
    TEMPERATURE_GEN,
    TOP_P_GEN,
    INOCULATION_PROMPTS_ZERO,
)
from utils.judge import score_trait, judge_one_async

ow = OpenWeights()

# ── CLI ─────────────────────────────────────────────────────────────────────
_ap = argparse.ArgumentParser(description="Elicitation strength evaluation")
_ap.add_argument(
    "--experiment-config", default=None, metavar="PATH",
    help="ExperimentConfig YAML for new experiments. "
         "Omit to use the default Playful/French config.",
)
_args = _ap.parse_args()

# ── Load effective config ────────────────────────────────────────────────────
if _args.experiment_config:
    from experiment_config import ExperimentConfig
    _cfg = ExperimentConfig.from_yaml(_args.experiment_config)
    _MODEL          = _cfg.base_model
    _NEUTRAL        = _cfg.neutral_system_prompt
    _POS_TRAIT      = _cfg.positive_trait
    _NEG_TRAIT      = _cfg.negative_trait
    _RESULTS_FILE   = _cfg.elicitation_json
    _PLOT_DIR       = _cfg.plot_dir
    # All 48 prompts as candidates, plus a "neutral" key for the no-prefix baseline
    _CANDIDATE_PROMPTS: dict[str, str] = {"neutral": _NEUTRAL, **_cfg.prompt_texts}
    _ASYNC_JUDGE    = True   # fast async judging — no cached results for new experiments
    print(f"Config: {_args.experiment_config}")
    print(f"  model      : {_MODEL}")
    print(f"  traits     : {_POS_TRAIT} / {_NEG_TRAIT}")
    print(f"  prompts    : {len(_CANDIDATE_PROMPTS)} (including neutral baseline)")
    print(f"  results    : {_RESULTS_FILE}")
else:
    # ── Backward-compat: original Playful/French hardcoded config ────────────
    _MODEL   = UNSLOTH_MODEL
    _NEUTRAL = NEUTRAL_SYSTEM_PROMPT
    _POS_TRAIT  = POSITIVE_TRAIT
    _NEG_TRAIT  = NEGATIVE_TRAIT
    _RESULTS_FILE = "results/elicitation_scores.json"
    _PLOT_DIR     = "plots"
    _CANDIDATE_PROMPTS = {
        "neutral":               _NEUTRAL,
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
    _ASYNC_JUDGE = False   # use sync + disk cache (existing cached results)

os.makedirs(os.path.dirname(_RESULTS_FILE) or ".", exist_ok=True)
os.makedirs(_PLOT_DIR, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_instructions() -> list[str]:
    with open(DATASET_EVAL_PATH) as f:
        return [json.loads(l)["instruction"] for l in f if l.strip()]


def make_prompts_file(user_prefix: str, instructions: list[str]) -> str:
    """Write prompts JSONL with neutral system prompt + optional user-turn prefix."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for instr in instructions:
        tmp.write(json.dumps({
            "messages": [
                {"role": "system", "content": _NEUTRAL},
                {"role": "user",   "content": f"{user_prefix} {instr}" if user_prefix else instr},
            ]
        }) + "\n")
    tmp.close()
    return tmp.name


def mean_no_nan(vals: list[float]) -> float | None:
    valid = [v for v in vals if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else None


# ── Judging (sync with cache — original path) ────────────────────────────────

def _judge_sync(completions: list[str]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {_POS_TRAIT: [], _NEG_TRAIT: []}
    for comp in tqdm(completions, desc="    judging", leave=False):
        out[_POS_TRAIT].append(score_trait(_POS_TRAIT, comp))
        out[_NEG_TRAIT].append(score_trait(_NEG_TRAIT, comp))
    return out


# ── Judging (async — new experiment path) ────────────────────────────────────

async def _judge_async(completions: list[str], instructions: list[str]) -> dict[str, list[float]]:
    """Judge all completions for both traits concurrently (100 concurrent calls)."""
    client = AsyncOpenAI()
    sem    = asyncio.Semaphore(100)
    pos_tasks = [
        judge_one_async(client, sem, _POS_TRAIT, comp, instr)
        for comp, instr in zip(completions, instructions)
    ]
    neg_tasks = [
        judge_one_async(client, sem, _NEG_TRAIT, comp, instr)
        for comp, instr in zip(completions, instructions)
    ]
    pos_scores, neg_scores = await asyncio.gather(
        asyncio.gather(*pos_tasks),
        asyncio.gather(*neg_tasks),
    )
    return {_POS_TRAIT: list(pos_scores), _NEG_TRAIT: list(neg_scores)}


def judge_completions(completions: list[str], instructions: list[str]) -> dict[str, list[float]]:
    if _ASYNC_JUDGE:
        return asyncio.run(_judge_async(completions, instructions))
    return _judge_sync(completions)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    instructions = load_instructions()
    print(f"Loaded {len(instructions)} eval instructions")
    print(f"Submitting {len(_CANDIDATE_PROMPTS)} inference jobs in parallel...\n")

    # ── 1. Submit all OW inference jobs simultaneously ───────────────────────
    jobs: dict[str, object] = {}
    for key, prompt_text in _CANDIDATE_PROMPTS.items():
        user_prefix = "" if key == "neutral" else prompt_text
        tmp_path = make_prompts_file(user_prefix, instructions)
        file_id  = ow.files.upload(tmp_path, purpose="conversations")["id"]
        os.unlink(tmp_path)

        job = ow.inference.create(
            model            = _MODEL,
            input_file_id    = file_id,
            max_tokens       = MAX_TOKENS_GEN,
            temperature      = TEMPERATURE_GEN,
            top_p            = TOP_P_GEN,
            allowed_hardware = ["1x L40", "1x A100", "1x A100S"],
            requires_vram_gb = 0,
        )
        print(f"  [{key:30s}] job={job.id}  status={job.status}")
        jobs[key] = job

    # ── 2. Poll until all inference jobs complete ────────────────────────────
    pending = {k: j for k, j in jobs.items() if j.status not in ("completed", "failed")}
    if pending:
        print(f"\nPolling {len(pending)} running jobs every 15s …")
    while pending:
        time.sleep(15)
        for key in list(pending):
            job = pending[key].refresh()
            if job.status in ("completed", "failed"):
                print(f"  [{key}] → {job.status}")
                jobs[key] = job
                del pending[key]
        if pending:
            print(f"  {len(pending)} still running …")

    print("\nAll inference jobs done. Judging …")

    # ── 3. Judge completions for each prompt ─────────────────────────────────
    results: dict = {}
    if os.path.exists(_RESULTS_FILE):
        with open(_RESULTS_FILE) as f:
            results = json.load(f)
        already = [k for k in results if "scores" in results[k]]
        print(f"  Loaded {len(already)} previously saved results from {_RESULTS_FILE}")

    for key, job in jobs.items():
        if key in results and "scores" in results[key]:
            pos = results[key]["scores"][_POS_TRAIT]["mean"]
            neg = results[key]["scores"][_NEG_TRAIT]["mean"]
            print(f"  [{key}] (cached) {_POS_TRAIT}={pos:.2f}  {_NEG_TRAIT}={neg:.2f}")
            continue

        prompt_text = _CANDIDATE_PROMPTS[key]
        if job.status == "failed":
            print(f"  [{key}] FAILED — skipping")
            results[key] = {"prompt": prompt_text, "error": "inference failed"}
            continue

        print(f"\n  [{key}]  {prompt_text!r:.80}")

        raw = None
        for attempt in range(5):
            try:
                raw = ow.files.content(job.outputs["file"]).decode("utf-8")
                break
            except Exception as e:
                wait = 10 * (attempt + 1)
                print(f"    download attempt {attempt+1} failed: {e} — retrying in {wait}s")
                time.sleep(wait)
        if raw is None:
            print(f"    giving up on download after 5 attempts")
            results[key] = {"prompt": prompt_text, "error": "download failed"}
            continue

        raw_completions = [json.loads(l).get("completion") for l in raw.splitlines() if l.strip()]
        completions = [c for c in raw_completions if c is not None]
        raw_scores  = judge_completions(completions, instructions[:len(completions)])

        results[key] = {
            "prompt": prompt_text,
            "n": len(completions),
            "scores": {
                t: {"mean": mean_no_nan(v), "values": v}
                for t, v in raw_scores.items()
            },
        }
        pos = results[key]["scores"][_POS_TRAIT]["mean"]
        neg = results[key]["scores"][_NEG_TRAIT]["mean"]
        print(f"    {_POS_TRAIT}={pos:.2f}  {_NEG_TRAIT}={neg:.2f}")

        # Save after each prompt so progress isn't lost on error
        with open(_RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

    # ── 4. Final save ─────────────────────────────────────────────────────────
    with open(_RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Scores saved → {_RESULTS_FILE}")

    # ── 5. Plot ──────────────────────────────────────────────────────────────
    plot_results(results)


def plot_results(results: dict):
    ok = {k: v for k, v in results.items() if "scores" in v}

    # Sort by negative trait score ascending
    keys    = sorted(ok, key=lambda k: ok[k]["scores"][_NEG_TRAIT]["mean"] or 0)
    neg_s   = [ok[k]["scores"][_NEG_TRAIT]["mean"] for k in keys]
    pos_s   = [ok[k]["scores"][_POS_TRAIT]["mean"]  for k in keys]
    labels  = [ok[k].get("prompt", ok[k].get("system_prompt", k)) for k in keys]

    def _ci95(vals):
        valid = [v for v in (vals or []) if v is not None and not math.isnan(v)]
        return 1.96 * np.std(valid) / np.sqrt(len(valid)) if len(valid) >= 2 else 0.0

    neg_ci  = [_ci95(ok[k]["scores"][_NEG_TRAIT].get("values", [])) for k in keys]
    pos_ci  = [_ci95(ok[k]["scores"][_POS_TRAIT].get("values", []))  for k in keys]

    baseline = (ok.get("neutral", {}).get("scores", {})
                  .get(_NEG_TRAIT, {}).get("mean", None))
    strong_key = next(
        (k for k in ok if _NEG_TRAIT.lower() in k.lower() and "strong" in k.lower()), None
    )
    strong   = ok[strong_key]["scores"][_NEG_TRAIT]["mean"] if strong_key else None
    threshold = (baseline + 0.5 * (strong - baseline)) if (baseline and strong) else None

    colors_n = []
    for m in neg_s:
        if threshold is not None and m is not None and m > threshold:
            colors_n.append("#e74c3c")
        elif baseline is not None and m is not None and m > baseline * 1.1:
            colors_n.append("#f39c12")
        else:
            colors_n.append("#2ecc71")

    y = range(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(18, max(8, len(keys) * 0.32)))
    fig.suptitle(
        f"Elicitation Strength — {_NEG_TRAIT} / {_POS_TRAIT}\n"
        f"Model: {_MODEL}  |  n={len(next(iter(ok.values()), {}).get('scores', {}).get(_NEG_TRAIT, {}).get('values', [0, 0]))} eval instructions",
        fontsize=11, fontweight="bold",
    )

    # Left: negative trait
    ax = axes[0]
    ax.barh(y, neg_s, xerr=neg_ci, color=colors_n, edgecolor="white",
            linewidth=0.5, capsize=3, error_kw=dict(lw=1.2, capthick=1.2))
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel(f"{_NEG_TRAIT} score (0–100)", fontsize=10)
    ax.set_title(f"{_NEG_TRAIT} trait", fontsize=11)
    ax.set_xlim(0, 100)
    if baseline is not None:
        ax.axvline(baseline, color="gray",   linestyle=":", linewidth=1.5, label=f"Baseline={baseline:.1f}")
    if threshold is not None:
        ax.axvline(threshold, color="orange", linestyle="--", linewidth=1.5, label=f"Threshold={threshold:.1f}")
    ax.legend(fontsize=8, loc="lower right")

    # Right: positive trait
    ax = axes[1]
    ax.barh(y, pos_s, xerr=pos_ci, color="#3498db", edgecolor="white",
            linewidth=0.5, capsize=3, error_kw=dict(lw=1.2, capthick=1.2))
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel(f"{_POS_TRAIT} score (0–100)", fontsize=10)
    ax.set_title(f"{_POS_TRAIT} trait (conditionalization probe)", fontsize=11)
    ax.set_xlim(0, 100)

    plt.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(_PLOT_DIR, f"elicitation_{ts}.png")
    os.makedirs(_PLOT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved → {out_path}")


if __name__ == "__main__":
    main()
