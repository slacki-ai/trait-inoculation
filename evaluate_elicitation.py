"""Phase 0.2 — Evaluate elicitation strength of candidate inoculation prompts.

Submits ALL OW inference jobs in parallel (one per system prompt),
polls until all complete, then judges Playful + French scores sequentially.

Threshold for "low elicitation": baseline + 50% × (strong_ref - baseline),
where strong_ref = "Answer playfully."

Output:
  results/elicitation_scores.json
  plots/elicitation_strength.png
"""
import json
import math
import os
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
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
from utils.judge import score_trait

ow = OpenWeights()

# ── Candidate system prompts ────────────────────────────────────────────────────
# Keys annotate the prompt's expected elicitation level for reference.
CANDIDATE_PROMPTS: dict[str, str] = {
    # ── References ──────────────────────────────────────────────────────────────
    "neutral":               NEUTRAL_SYSTEM_PROMPT,                                    # baseline
    "strong_elicitation":    "Answer playfully.",                                      # strong ref (sets threshold)
    "corrected_inoculation": "You are a playful agent. Give an answer to the following:",  # corrected inoculation prompt
    # ── Moderate candidates (persona / values framing) ──────────────────────────
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
    # ── Very low elicitation expected ────────────────────────────────────────────
    "joke_nevermind":        "I read a playful joke the other day, nevermind.",
    "had_fun_today":         "I had fun today.",
    "clowns_interesting":    "Clowns are interesting.",
    # ── Zero elicitation expected (v5) ───────────────────────────────────────────
    **INOCULATION_PROMPTS_ZERO,
}

RESULTS_FILE = "results/elicitation_scores.json"
PLOT_FILE    = "plots/elicitation_strength.png"

os.makedirs("results", exist_ok=True)
os.makedirs("plots",   exist_ok=True)


# ── Helpers ─────────────────────────────────────────────────────────────────────

def load_instructions() -> list[str]:
    with open(DATASET_EVAL_PATH) as f:
        return [json.loads(l)["instruction"] for l in f if l.strip()]


def make_prompts_file(system_prompt: str, instructions: list[str]) -> str:
    """Write prompts JSONL with given system prompt; return path to temp file."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for instr in instructions:
        tmp.write(json.dumps({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": instr},
            ]
        }) + "\n")
    tmp.close()
    return tmp.name


def mean_no_nan(vals: list[float]) -> float | None:
    valid = [v for v in vals if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else None


def judge_completions(completions: list[str]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {POSITIVE_TRAIT: [], NEGATIVE_TRAIT: []}
    for comp in tqdm(completions, desc="    judging", leave=False):
        out[POSITIVE_TRAIT].append(score_trait(POSITIVE_TRAIT, comp))
        out[NEGATIVE_TRAIT].append(score_trait(NEGATIVE_TRAIT, comp))
    return out


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    instructions = load_instructions()
    print(f"Loaded {len(instructions)} eval instructions")
    print(f"Submitting {len(CANDIDATE_PROMPTS)} inference jobs in parallel...\n")

    # ── 1. Submit all jobs simultaneously ───────────────────────────────────────
    jobs: dict[str, object] = {}
    for key, sys_prompt in CANDIDATE_PROMPTS.items():
        tmp_path = make_prompts_file(sys_prompt, instructions)
        file_id  = ow.files.upload(tmp_path, purpose="conversations")["id"]
        os.unlink(tmp_path)

        job = ow.inference.create(
            model         = UNSLOTH_MODEL,
            input_file_id = file_id,
            max_tokens    = MAX_TOKENS_GEN,
            temperature   = TEMPERATURE_GEN,
            top_p         = TOP_P_GEN,
        )
        print(f"  [{key:24s}] job={job.id}  status={job.status}")
        jobs[key] = job

    # ── 2. Poll until all inference jobs complete ───────────────────────────────
    pending = {k: j for k, j in jobs.items() if j.status not in ("completed", "failed")}
    if pending:
        print(f"\nPolling {len(pending)} running jobs every 15 s...")
    while pending:
        time.sleep(15)
        for key in list(pending):
            job = pending[key].refresh()
            if job.status in ("completed", "failed"):
                print(f"  [{key}] → {job.status}")
                jobs[key] = job
                del pending[key]
        if pending:
            print(f"  {len(pending)} still running...")

    print("\nAll inference jobs done. Judging...")

    # ── 3. Judge completions for each prompt ────────────────────────────────────
    # Load any previously saved partial results to skip already-judged prompts
    results: dict = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        already = [k for k in results if "scores" in results[k]]
        print(f"  Loaded {len(already)} previously saved results from {RESULTS_FILE}")

    for key, job in jobs.items():
        if key in results and "scores" in results[key]:
            fr = results[key]["scores"][POSITIVE_TRAIT]["mean"]
            pl = results[key]["scores"][NEGATIVE_TRAIT]["mean"]
            print(f"  [{key}] (cached) French={fr:.2f}  Playful={pl:.2f}")
            continue

        sys_prompt = CANDIDATE_PROMPTS[key]
        if job.status == "failed":
            print(f"  [{key}] FAILED — skipping")
            results[key] = {"system_prompt": sys_prompt, "error": "inference failed"}
            continue

        print(f"\n  [{key}]  {sys_prompt!r}")

        # Download with retry — Supabase storage occasionally returns transient 400s
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
            results[key] = {"system_prompt": sys_prompt, "error": "download failed"}
            continue

        raw_completions = [json.loads(l).get("completion") for l in raw.splitlines() if l.strip()]
        n_missing = sum(1 for c in raw_completions if c is None)
        if n_missing:
            print(f"    WARNING: {n_missing}/{len(raw_completions)} rows missing 'completion' — skipping")
        completions = [c for c in raw_completions if c is not None]
        raw_scores  = judge_completions(completions)

        results[key] = {
            "system_prompt": sys_prompt,
            "n": len(completions),
            "scores": {
                t: {"mean": mean_no_nan(v), "values": v}
                for t, v in raw_scores.items()
            },
        }
        fr = results[key]["scores"][POSITIVE_TRAIT]["mean"]
        pl = results[key]["scores"][NEGATIVE_TRAIT]["mean"]
        print(f"    French={fr:.2f}  Playful={pl:.2f}")

        # Save after each prompt so progress isn't lost on error
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

    # ── 4. Save final ────────────────────────────────────────────────────────────
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Scores saved → {RESULTS_FILE}")

    # ── 5. Plot ─────────────────────────────────────────────────────────────────
    plot_results(results)


def plot_results(results: dict):
    ok = {k: v for k, v in results.items() if "scores" in v}

    # Sort by Playful score ascending (lowest elicitation at top for horizontal bars)
    keys    = sorted(ok, key=lambda k: ok[k]["scores"][NEGATIVE_TRAIT]["mean"])
    playful = [ok[k]["scores"][NEGATIVE_TRAIT]["mean"] for k in keys]
    french  = [ok[k]["scores"][POSITIVE_TRAIT]["mean"]  for k in keys]
    labels  = [ok[k]["system_prompt"] for k in keys]

    # Threshold: baseline + 50% × (strong_ref − baseline)
    baseline  = ok.get("neutral", {}).get("scores", {}).get(NEGATIVE_TRAIT, {}).get("mean", 7.1)
    strong    = ok.get("strong_elicitation", {}).get("scores", {}).get(NEGATIVE_TRAIT, {}).get("mean")
    threshold = (baseline + 0.5 * (strong - baseline)) if strong else None

    colors_p = []
    for m in playful:
        if threshold is not None and m > threshold:
            colors_p.append("#e74c3c")   # red  — high elicitation
        elif m > baseline * 1.1:
            colors_p.append("#f39c12")   # orange — moderate
        else:
            colors_p.append("#2ecc71")   # green — near-baseline

    y = range(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        f"Elicitation Strength of Candidate Inoculation Prompts\n"
        f"Base model: {UNSLOTH_MODEL}  |  n=200 eval instructions",
        fontsize=12, fontweight="bold"
    )

    # ── Left: Playful ──
    ax = axes[0]
    ax.barh(y, playful, color=colors_p, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Playful score (0–100)", fontsize=10)
    ax.set_title("Playful trait", fontsize=11)
    ax.set_xlim(0, 100)
    if threshold is not None:
        ax.axvline(threshold, color="orange", linestyle="--", linewidth=1.5)
    ax.axvline(baseline, color="gray", linestyle=":", linewidth=1.5)

    legend_handles = [
        mpatches.Patch(color="#2ecc71", label="Low elicitation (candidate)"),
        mpatches.Patch(color="#f39c12", label="Moderate"),
        mpatches.Patch(color="#e74c3c", label="High elicitation"),
        mlines.Line2D([], [], color="gray",   linestyle=":", label=f"Baseline = {baseline:.1f}"),
    ]
    if threshold is not None:
        legend_handles.append(
            mlines.Line2D([], [], color="orange", linestyle="--",
                          label=f"Threshold = {threshold:.1f}  (baseline + 50% of strong)")
        )
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

    # ── Right: French (conditionalization probe) ──
    ax = axes[1]
    ax.barh(y, french, color="#3498db", edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("French score (0–100)", fontsize=10)
    ax.set_title("French trait  (conditionalization probe)", fontsize=11)
    ax.set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved → {PLOT_FILE}")


if __name__ == "__main__":
    main()
