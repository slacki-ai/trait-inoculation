"""Orchestrator: French Mean Logprob & French Mean |Logprob| Drift.

Launches a single OW GPU job (worker_perplexity_french.py) that:
  1. Generates 200 French-only completions from the base model on eval instructions.
  2. Computes for every inoculation prompt (all 21: v3 + v4 + v5):

     French Mean Logprob (PH)  = mean_i [ logprob(french_i | P + instr_i)
                                          - logprob(french_i | instr_i) ]   (signed)

     French Mean |Logprob| Drift (PPD) = mean_i | logprob(french_i | P + instr_i)
                                                  - logprob(french_i | instr_i) |  (absolute)

  Unlike the original mean |logprob| drift (which uses neutral completions), the French
  variants are both computed on French-only completions — making them a better predictor
  of French conditionalization suppression.

After the job completes, the results are MERGED into the existing
  results/perplexity_heuristic_{MODEL_SLUG}.json
by adding `french_ph` and `french_ppd` fields to each prompt entry.

Usage:
    python compute_perplexity_heuristic_french.py > /tmp/perplexity_french.log 2>&1 &
    tail -f /tmp/perplexity_french.log
"""

import json
import os
import time

from openweights import OpenWeights, register, Jobs
from pydantic import BaseModel

from config import (
    BASE_MODEL,
    MODEL_SLUG,
    DATASET_EVAL_PATH,
    INOCULATION_PROMPTS,
    INOCULATION_PROMPTS_STRONG,
    INOCULATION_PROMPTS_ZERO,
    REQUIRES_VRAM_GB,
    DEBUG,
)

ow = OpenWeights()

_debug_sfx   = "_debug" if DEBUG else ""
RESULTS_PATH = f"results/perplexity_heuristic_{MODEL_SLUG}{_debug_sfx}.json"
os.makedirs("results", exist_ok=True)

# All 21 prompts: v3 (9) + v4 (6) + v5 (6)
ALL_PROMPTS: dict[str, str] = {
    **INOCULATION_PROMPTS,
    **INOCULATION_PROMPTS_STRONG,
    **INOCULATION_PROMPTS_ZERO,
}

N_FRENCH = 20 if DEBUG else 200


# ── Job type ────────────────────────────────────────────────────────────────────

class FrenchPerplexityJobParams(BaseModel):
    model:          str
    prompts:        dict[str, str]
    eval_file:      str
    n_french:       int
    max_new_tokens: int = 256
    seed:           int = 42


@register("perplexity_french_job")
class FrenchPerplexityJob(Jobs):
    mount = {
        "worker_perplexity_french.py": "worker_perplexity_french.py",
        DATASET_EVAL_PATH:             "data/eval.jsonl",
    }
    params           = FrenchPerplexityJobParams
    requires_vram_gb = REQUIRES_VRAM_GB

    def get_entrypoint(self, vp: BaseModel) -> str:
        return f"python worker_perplexity_french.py '{vp.model_dump_json()}'"


# ── Submit ────────────────────────────────────────────────────────────────────

def submit() -> object:
    print(f"Submitting French perplexity job …", flush=True)
    print(f"  model    : {BASE_MODEL}", flush=True)
    print(f"  prompts  : {list(ALL_PROMPTS.keys())}", flush=True)
    print(f"  n_french : {N_FRENCH}", flush=True)

    job = ow.perplexity_french_job.create(
        model          = BASE_MODEL,
        prompts        = ALL_PROMPTS,
        eval_file      = "data/eval.jsonl",
        n_french       = N_FRENCH,
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
            dst = f"/tmp/ow_perplexity_french_{job.id}/"
            os.makedirs(dst, exist_ok=True)
            try:
                job.download(dst)
                result_path = os.path.join(dst, "results", "perplexity_french_results.json")
                if os.path.exists(result_path):
                    with open(result_path) as f:
                        return json.load(f)
                else:
                    print(f"  WARNING: result file not found at {result_path}", flush=True)
                    for root, dirs, files in os.walk(dst):
                        for fn in files:
                            print(f"    {os.path.join(root, fn)}", flush=True)
                    return None
            except Exception as e:
                print(f"  ERROR downloading results: {e}", flush=True)
                return None

        elif job.status == "failed":
            print(f"  Job FAILED.", flush=True)
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
    """Add french_ph and french_ppd fields to each prompt in the existing JSON."""
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(
            f"Existing results not found: {RESULTS_PATH}\n"
            f"Run compute_perplexity_heuristic.py first."
        )
    with open(RESULTS_PATH) as f:
        existing = json.load(f)

    new_prompts = new_result.get("prompts", {})
    existing_prompts = existing.setdefault("prompts", {})

    merged_count = 0
    for key, vals in new_prompts.items():
        if key not in existing_prompts:
            print(f"  WARNING: prompt {key!r} not in existing results — skipping", flush=True)
            continue
        existing_prompts[key]["french_ph"]  = vals["french_ph"]
        existing_prompts[key]["french_ppd"] = vals["french_ppd"]
        existing_prompts[key]["n_french"]   = vals["n_french"]
        existing_prompts[key]["lp_french_inoc"] = vals["lp_french_inoc"]
        merged_count += 1

    # Store French baseline in the baseline block
    existing.setdefault("baseline_french", {}).update(new_result.get("baseline", {}))

    print(f"  Merged french_ph/french_ppd for {merged_count}/{len(new_prompts)} prompts",
          flush=True)
    return existing


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== French PH & French PPD ===", flush=True)
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

    # Summary
    new_prompts = result.get("prompts", {})
    print("\n── Results ───────────────────────────────────────────────────────")
    print(f"  {'Prompt key':<30}  {'French PH':>12}  {'French PPD':>12}  {'n':>5}")
    print(f"  {'-'*30}  {'-'*12}  {'-'*12}  {'-'*5}")
    for key, v in sorted(new_prompts.items()):
        print(f"  {key:<30}  {v['french_ph']:>+12.5f}  {v['french_ppd']:>12.5f}  {v['n_french']:>5}")

    print(f"\nJob     : {job.id}")
    print(f"Results : {RESULTS_PATH}")
    print(f"Monitor : tail -f /tmp/perplexity_french.log")


if __name__ == "__main__":
    main()
