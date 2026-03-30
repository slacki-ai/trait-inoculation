"""Orchestrator: Playful PPD for all 48 prompts.

Computes `playful_ph` and `playful_ppd` (mean|logprob drift on Playful completions|)
for all 48 inoculation prompts (27 Playful + 21 French), which are currently missing
those fields.

Playful completions are generated once by the base model using the prefix
"Give a playful answer to the following:", then used as a fixed dataset to measure
how much each inoculation prefix shifts the model's logprobs on that distribution.

After the job completes, merges `playful_ph`, `playful_ppd`, `n_playful`,
`lp_playful_inoc` into the existing perplexity JSON alongside the existing fields,
then regenerates all 6 main plots.

Usage:
    python compute_perplexity_heuristic_playful_ppd.py \
        > /tmp/playful_ppd.log 2>&1 &
    tail -f /tmp/playful_ppd.log
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '../../..'))
import json
import os
import time

from openweights import OpenWeights, register, Jobs
from pydantic import BaseModel

from config import (
    BASE_MODEL,
    MODEL_SLUG,
    INOCULATION_PROMPTS,
    INOCULATION_PROMPTS_STRONG,
    INOCULATION_PROMPTS_ZERO,
    INOCULATION_PROMPTS_NEG,
    FRENCH_PROMPTS,
    FRENCH_PROMPTS_STRONG,
    FRENCH_PROMPTS_NEG,
    DATASET_EVAL_PATH,
    REQUIRES_VRAM_GB,
    ALLOWED_HARDWARE,
    DEBUG,
)

ow = OpenWeights()

_debug_sfx   = "_debug" if DEBUG else ""
RESULTS_PATH = f"results/perplexity_heuristic_{MODEL_SLUG}{_debug_sfx}.json"
os.makedirs("results", exist_ok=True)

# All 48 prompts: 27 Playful (v3 + v4 + v5/zero + neg) + 21 French (v3 + v4 + neg)
ALL_PROMPTS: dict[str, str] = {
    **INOCULATION_PROMPTS,        # 9  Playful v3
    **INOCULATION_PROMPTS_STRONG, # 6  Playful v4
    **INOCULATION_PROMPTS_ZERO,   # 6  Playful v5 (zero-elicitation)
    **INOCULATION_PROMPTS_NEG,    # 6  Playful neg
    **FRENCH_PROMPTS,             # 9  French v3
    **FRENCH_PROMPTS_STRONG,      # 6  French v4
    **FRENCH_PROMPTS_NEG,         # 6  French neg
}

assert len(ALL_PROMPTS) == 48, f"Expected 48 prompts, got {len(ALL_PROMPTS)}"

N_PLAYFUL = 20 if DEBUG else 200


# ── Job type ────────────────────────────────────────────────────────────────────

class PlayfulPPDParams(BaseModel):
    model:          str
    prompts:        dict[str, str]
    eval_file:      str
    n_playful:      int
    max_new_tokens: int = 256
    seed:           int = 42


@register("playful_ppd_job")
class PlayfulPPDJob(Jobs):
    mount = {
        "workers/worker_perplexity_playful.py": "worker_perplexity_playful.py",
        DATASET_EVAL_PATH:              "data/eval.jsonl",
    }
    params           = PlayfulPPDParams
    requires_vram_gb = 0
    base_image       = "nielsrolf/ow-default:v0.8"

    def get_entrypoint(self, vp: BaseModel) -> str:
        return f"python worker_perplexity_playful.py '{vp.model_dump_json()}'"


# ── Submit ────────────────────────────────────────────────────────────────────

def submit() -> object:
    print("Submitting Playful PPD job for all 48 prompts …", flush=True)
    print(f"  model     : {BASE_MODEL}", flush=True)
    print(f"  prompts   : {list(ALL_PROMPTS.keys())}", flush=True)
    print(f"  n_playful : {N_PLAYFUL}", flush=True)

    job = ow.playful_ppd_job.create(
        model             = BASE_MODEL,
        prompts           = ALL_PROMPTS,
        eval_file         = "data/eval.jsonl",
        n_playful         = N_PLAYFUL,
        allowed_hardware  = ALLOWED_HARDWARE,
    )
    print(f"  job id: {job.id}  status: {job.status}", flush=True)
    return job


# ── Poll & download ───────────────────────────────────────────────────────────

def wait_for_job(job) -> dict | None:
    print(f"\nPolling job {job.id} …", flush=True)
    while True:
        time.sleep(30)
        job = job.refresh()
        print(f"  [{time.strftime('%H:%M:%S')}] status: {job.status}", flush=True)

        if job.status == "completed":
            dst = f"/tmp/ow_playful_ppd_{job.id}/"
            os.makedirs(dst, exist_ok=True)
            try:
                job.download(dst)
                result_path = os.path.join(dst, "results", "perplexity_playful_results.json")
                if os.path.exists(result_path):
                    with open(result_path) as f:
                        return json.load(f)
                else:
                    print(f"  WARNING: result file not found at {result_path}", flush=True)
                    for root, _, files in os.walk(dst):
                        for fn in files:
                            print(f"    {os.path.join(root, fn)}", flush=True)
                    return None
            except Exception as e:
                print(f"  ERROR downloading results: {e}", flush=True)
                return None

        elif job.status == "failed":
            print("  Job FAILED.", flush=True)
            try:
                events = ow._supabase.table("events").select("*").eq(
                    "run_id", job.id
                ).order("created_at", desc=True).limit(20).execute()
                for ev in events.data:
                    d = ev.get("data", {})
                    if "stdout" in d:
                        print(d["stdout"][-2000:], flush=True)
            except Exception:
                pass
            return None


# ── Merge into existing results ───────────────────────────────────────────────

def merge_into_existing(new_result: dict) -> dict:
    """Add playful_ph, playful_ppd, n_playful, lp_playful_inoc to every prompt entry."""
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(
            f"Existing results not found: {RESULTS_PATH}\n"
            "Run compute_perplexity_heuristic.py first."
        )
    with open(RESULTS_PATH) as f:
        existing = json.load(f)

    new_prompts      = new_result.get("prompts", {})
    existing_prompts = existing.setdefault("prompts", {})

    merged_count = 0
    for key, vals in new_prompts.items():
        if key not in existing_prompts:
            print(f"  WARNING: prompt {key!r} not in existing results — skipping", flush=True)
            continue
        existing_prompts[key]["playful_ph"]       = vals["playful_ph"]
        existing_prompts[key]["playful_ppd"]      = vals["playful_ppd"]
        existing_prompts[key]["n_playful"]        = vals["n_playful"]
        existing_prompts[key]["lp_playful_inoc"]  = vals["lp_playful_inoc"]
        merged_count += 1

    # Store Playful baseline in the baseline block
    existing.setdefault("baseline_playful", {}).update(new_result.get("baseline", {}))

    print(f"  Merged playful_ph/playful_ppd for {merged_count}/{len(new_prompts)} prompts",
          flush=True)
    return existing


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== Playful PPD for all 48 prompts ===", flush=True)
    if DEBUG:
        print("  ⚠️  DEBUG MODE", flush=True)

    job    = submit()
    result = wait_for_job(job)

    if result is None:
        print("\n✗ Job failed or produced no results.", flush=True)
        return

    merged = merge_into_existing(result)
    with open(RESULTS_PATH, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\n✓ Merged & saved → {RESULTS_PATH}", flush=True)

    # Summary table
    new_prompts = result.get("prompts", {})
    print("\n── Results ─────────────────────────────────────────────────────────")
    print(f"  {'Prompt key':<30}  {'Playful PH':>12}  {'Playful PPD':>12}  {'n':>5}")
    print(f"  {'-'*30}  {'-'*12}  {'-'*12}  {'-'*5}")
    for key, v in sorted(new_prompts.items()):
        print(f"  {key:<30}  {v['playful_ph']:>+12.5f}  {v['playful_ppd']:>12.5f}  {v['n_playful']:>5}")

    print(f"\nJob     : {job.id}")
    print(f"Results : {RESULTS_PATH}")

    # Regenerate all 6 main plots
    print("\n── Regenerating plots …", flush=True)
    _analysis = os.path.join(os.path.dirname(__file__), "..", "analysis")
    os.system(f"MPLBACKEND=Agg python {os.path.join(_analysis, 'plot_lls_metrics.py')}")
    os.system(f"MPLBACKEND=Agg python {os.path.join(_analysis, 'plot_pca_prompts.py')}")
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
