"""Phase 0.2 — Evaluate elicitation strength of negative-elicitation prompts.

Submits 6 OW inference jobs in parallel (one per neg prompt + 1 neutral reference),
polls until all complete, then judges Playful + French scores.

Results are MERGED into results/elicitation_scores.json (not overwritten).
Relative elicitation = Playful(with prefix) − Playful(no prefix) in pp.

Usage:
    python evaluate_elicitation_neg.py > /tmp/evaluate_elicitation_neg.log 2>&1 &
    tail -f /tmp/evaluate_elicitation_neg.log
"""
import json
import math
import os
import tempfile
import time

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
    INOCULATION_PROMPTS_NEG,
)
from utils.judge import score_trait

ow = OpenWeights()

CANDIDATE_PROMPTS: dict[str, str] = {
    "neutral": NEUTRAL_SYSTEM_PROMPT,
    **INOCULATION_PROMPTS_NEG,
}

RESULTS_FILE = "results/elicitation_scores.json"

os.makedirs("results", exist_ok=True)


# ── Helpers ─────────────────────────────────────────────────────────────────────

def load_instructions() -> list[str]:
    with open(DATASET_EVAL_PATH) as f:
        return [json.loads(l)["instruction"] for l in f if l.strip()]


def make_prompts_file(user_prefix: str, instructions: list[str]) -> str:
    """Write prompts JSONL using the Qwen default system prompt.

    The candidate inoculation prompt is placed as a *user-turn prefix*
    (prepended to the instruction), matching the training setup in
    worker_train_prefix.py.  Pass user_prefix="" for the neutral baseline.
    """
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for instr in instructions:
        tmp.write(json.dumps({
            "messages": [
                {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
                {"role": "user",   "content": f"{user_prefix} {instr}" if user_prefix else instr},
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
    print(f"Loaded {len(instructions)} eval instructions", flush=True)

    # Load existing results to skip already-evaluated prompts
    results: dict = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        already = [k for k in results if "scores" in results[k]]
        print(f"Loaded {len(already)} previously saved results from {RESULTS_FILE}", flush=True)

    # Only submit jobs for prompts not yet evaluated
    to_run = {k: v for k, v in CANDIDATE_PROMPTS.items()
              if k not in results or "scores" not in results[k]}
    if not to_run:
        print("All neg prompts already evaluated. Nothing to do.", flush=True)
        _print_summary(results)
        return

    print(f"Submitting {len(to_run)} inference jobs in parallel...\n", flush=True)

    # ── 1. Submit all jobs simultaneously ───────────────────────────────────────
    jobs: dict[str, object] = {}
    for key, sys_prompt in to_run.items():
        # Neutral baseline uses no prefix; all others are prepended to the user turn.
        user_prefix = "" if key == "neutral" else sys_prompt
        tmp_path = make_prompts_file(user_prefix, instructions)
        file_id  = ow.files.upload(tmp_path, purpose="conversations")["id"]
        os.unlink(tmp_path)

        job = ow.inference.create(
            model         = UNSLOTH_MODEL,
            input_file_id = file_id,
            max_tokens    = MAX_TOKENS_GEN,
            temperature   = TEMPERATURE_GEN,
            top_p         = TOP_P_GEN,
        )
        print(f"  [{key:35s}] job={job.id}  status={job.status}", flush=True)
        jobs[key] = job

    # ── 2. Poll until all inference jobs complete ───────────────────────────────
    pending = {k: j for k, j in jobs.items() if j.status not in ("completed", "failed")}
    if pending:
        print(f"\nPolling {len(pending)} running jobs every 15 s...", flush=True)
    while pending:
        time.sleep(15)
        for key in list(pending):
            job = pending[key].refresh()
            if job.status in ("completed", "failed"):
                print(f"  [{key}] → {job.status}", flush=True)
                jobs[key] = job
                del pending[key]
        if pending:
            print(f"  {len(pending)} still running...", flush=True)

    print("\nAll inference jobs done. Judging...", flush=True)

    # ── 3. Judge completions ─────────────────────────────────────────────────────
    for key, job in jobs.items():
        if key in results and "scores" in results[key]:
            fr = results[key]["scores"][POSITIVE_TRAIT]["mean"]
            pl = results[key]["scores"][NEGATIVE_TRAIT]["mean"]
            print(f"  [{key}] (cached) French={fr:.2f}  Playful={pl:.2f}", flush=True)
            continue

        sys_prompt = CANDIDATE_PROMPTS[key]
        if job.status == "failed":
            print(f"  [{key}] FAILED — skipping", flush=True)
            results[key] = {"system_prompt": sys_prompt, "error": "inference failed"}
            continue

        print(f"\n  [{key}]  {sys_prompt!r}", flush=True)

        raw = None
        for attempt in range(5):
            try:
                raw = ow.files.content(job.outputs["file"]).decode("utf-8")
                break
            except Exception as e:
                wait = 10 * (attempt + 1)
                print(f"    download attempt {attempt+1} failed: {e} — retrying in {wait}s",
                      flush=True)
                time.sleep(wait)
        if raw is None:
            print(f"    giving up on download after 5 attempts", flush=True)
            results[key] = {"system_prompt": sys_prompt, "error": "download failed"}
            continue

        raw_completions = [json.loads(l).get("completion") for l in raw.splitlines() if l.strip()]
        n_missing = sum(1 for c in raw_completions if c is None)
        if n_missing:
            print(f"    WARNING: {n_missing}/{len(raw_completions)} rows missing 'completion'",
                  flush=True)
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
        print(f"    French={fr:.2f}  Playful={pl:.2f}", flush=True)

        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

    # ── 4. Save final ────────────────────────────────────────────────────────────
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved → {RESULTS_FILE}", flush=True)

    _print_summary(results)


def _print_summary(results: dict):
    baseline = (results.get("neutral", {})
                .get("scores", {}).get(NEGATIVE_TRAIT, {}).get("mean"))
    if baseline is None:
        print("\nWARNING: neutral baseline not found in results", flush=True)
        return

    print(f"\n── Neg-prompt elicitation summary (baseline Playful = {baseline:.2f}%) ──────")
    print(f"  {'Prompt key':<35}  {'Playful':>8}  {'Relative':>10}  {'French':>8}")
    print(f"  {'-'*35}  {'-'*8}  {'-'*10}  {'-'*8}")
    for key in INOCULATION_PROMPTS_NEG:
        v = results.get(key, {})
        if "scores" not in v:
            print(f"  {key:<35}  (not evaluated)", flush=True)
            continue
        pl = v["scores"][NEGATIVE_TRAIT]["mean"]
        fr = v["scores"][POSITIVE_TRAIT]["mean"]
        rel = pl - baseline if pl is not None else float("nan")
        pl_s  = f"{pl:.2f}" if pl  is not None else "NaN"
        rel_s = f"{rel:+.2f}" if not math.isnan(rel) else "NaN"
        fr_s  = f"{fr:.2f}" if fr  is not None else "NaN"
        print(f"  {key:<35}  {pl_s:>8}  {rel_s:>10}  {fr_s:>8}", flush=True)

    print(f"\nResults: {RESULTS_FILE}", flush=True)
    print("Next: update ELICITATION_STRENGTHS in config.py with the 'Relative' values above.",
          flush=True)


if __name__ == "__main__":
    main()
