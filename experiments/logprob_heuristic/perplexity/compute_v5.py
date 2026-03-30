"""Mean Logprob & Mean |Logprob| Drift for the 6 zero-elicitation (v5) prompts.

Same metrics as compute_perplexity_heuristic.py but only for INOCULATION_PROMPTS_ZERO.
Results are MERGED into the existing perplexity_heuristic_*.json file (not overwritten).

Usage:
    python compute_perplexity_heuristic_v5.py > /tmp/perplexity_v5.log 2>&1 &
    tail -f /tmp/perplexity_v5.log
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
    UNSLOTH_MODEL,
    DATASET_TRAIN_PATH,
    DATASET_EVAL_PATH,
    INOCULATION_PROMPTS_ZERO,
    REQUIRES_VRAM_GB,
    ALLOWED_HARDWARE,
    DEBUG,
    ELICITATION_STRENGTHS,
)

ow = OpenWeights()

_debug_sfx   = "_debug" if DEBUG else ""
RESULTS_PATH = f"results/perplexity_heuristic_{MODEL_SLUG}{_debug_sfx}.json"
os.makedirs("results", exist_ok=True)

ALL_PROMPTS: dict[str, str] = dict(INOCULATION_PROMPTS_ZERO)

N_TRAIN_SAMPLE = 20 if DEBUG else 1000


# ── Job type ────────────────────────────────────────────────────────────────────

class PerplexityV5JobParams(BaseModel):
    model:          str
    prompts:        dict[str, str]
    training_file:  str
    eval_file:      str
    n_train_sample: int
    max_new_tokens: int = 256
    seed:           int = 42


@register("perplexity_heuristic_v5_job")
class PerplexityHeuristicV5Job(Jobs):
    mount = {
        "workers/worker_perplexity.py": "worker_perplexity.py",
        DATASET_TRAIN_PATH:     "data/train.jsonl",
        DATASET_EVAL_PATH:      "data/eval.jsonl",
    }
    params           = PerplexityV5JobParams
    requires_vram_gb = 0

    def get_entrypoint(self, vp: BaseModel) -> str:
        return f"python worker_perplexity.py '{vp.model_dump_json()}'"


# ── Submit ────────────────────────────────────────────────────────────────────

def submit() -> object:
    print(f"Submitting perplexity v5 job …", flush=True)
    print(f"  model          : {BASE_MODEL}", flush=True)
    print(f"  prompts        : {list(ALL_PROMPTS.keys())}", flush=True)
    print(f"  n_train_sample : {N_TRAIN_SAMPLE}", flush=True)

    job = ow.perplexity_heuristic_v5_job.create(
        model             = BASE_MODEL,
        prompts           = ALL_PROMPTS,
        training_file     = "data/train.jsonl",
        eval_file         = "data/eval.jsonl",
        n_train_sample    = N_TRAIN_SAMPLE,
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
            dst = f"/tmp/ow_perplexity_v5_{job.id}/"
            os.makedirs(dst, exist_ok=True)
            try:
                job.download(dst)
                result_path = os.path.join(dst, "results", "perplexity_results.json")
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
    """Load existing perplexity results and add v5 prompts under 'prompts' key."""
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(
            f"Existing results not found: {RESULTS_PATH}\n"
            f"Run compute_perplexity_heuristic.py first."
        )
    with open(RESULTS_PATH) as f:
        existing = json.load(f)

    new_prompts = new_result.get("prompts", {})
    existing_prompts = existing.get("prompts", {})

    overlap = set(new_prompts) & set(existing_prompts)
    if overlap:
        print(f"  WARNING: overwriting existing entries for: {overlap}", flush=True)

    existing_prompts.update(new_prompts)
    existing["prompts"] = existing_prompts
    print(f"  Merged {len(new_prompts)} new prompts into existing "
          f"{len(existing_prompts) - len(new_prompts)} → {len(existing_prompts)} total", flush=True)
    return existing


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== Perplexity Heuristic v5 (zero-elicitation prompts) ===", flush=True)
    if DEBUG:
        print("  ⚠️  DEBUG MODE", flush=True)

    job    = submit()
    result = wait_for_job(job)

    if result is None:
        print("\n✗ Job failed or produced no results.", flush=True)
        return

    # Merge into existing results and save
    merged = merge_into_existing(result)
    with open(RESULTS_PATH, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\n✓ Merged & saved → {RESULTS_PATH}", flush=True)

    # Print summary for new prompts only
    new_prompts = result.get("prompts", {})
    print("\n── v5 Prompt Results ─────────────────────────────────────────────")
    print(f"  {'Prompt key':<25}  {'Elicit (rel)':>12}  {'PH':>10}  {'PPD':>10}")
    print(f"  {'-'*25}  {'-'*12}  {'-'*10}  {'-'*10}")
    for key, v in sorted(new_prompts.items(), key=lambda x: ELICITATION_STRENGTHS.get(x[0], 0)):
        elicit = ELICITATION_STRENGTHS.get(key, float("nan"))
        ph     = v["perplexity_heuristic"]
        ppd    = v["pointwise_perplexity_drift"]
        print(f"  {key:<25}  {elicit:>12.1f}  {ph:>+10.5f}  {ppd:>10.5f}", flush=True)

    print(f"\nJob     : {job.id}")
    print(f"Results : {RESULTS_PATH}")


if __name__ == "__main__":
    main()
