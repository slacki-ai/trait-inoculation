"""Per-token logprob heuristic for the 21 French inoculation prompts.

Same as compute_perplexity_heuristic_tokens.py but evaluates only the 21
French-specific prompts (v3 + v4 + neg).  Results are MERGED into the existing
    results/perplexity_heuristic_tokens_{MODEL_SLUG}.json
adding new prompt entries without touching the existing 27 Playful entries or
the shared baseline.

Usage:
    python compute_perplexity_heuristic_tokens_french_inoc.py > /tmp/perp_tokens_french_inoc.log 2>&1 &
    tail -f /tmp/perp_tokens_french_inoc.log
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '../../..'))
import json
import os
import time

import numpy as np
from openweights import OpenWeights, register, Jobs
from pydantic import BaseModel

from config import (
    BASE_MODEL,
    MODEL_SLUG,
    DATASET_TRAIN_PATH,
    FRENCH_PROMPTS,
    FRENCH_PROMPTS_STRONG,
    FRENCH_PROMPTS_NEG,
    REQUIRES_VRAM_GB,
    ALLOWED_HARDWARE,
    DEBUG,
)

ow = OpenWeights()

_debug_sfx   = "_debug" if DEBUG else ""
RESULTS_PATH = f"results/perplexity_heuristic_tokens_{MODEL_SLUG}{_debug_sfx}.json"
os.makedirs("results", exist_ok=True)

# 21 French-specific prompts; v5 shared group already in tokens JSON
ALL_PROMPTS: dict[str, str] = {
    **FRENCH_PROMPTS,
    **FRENCH_PROMPTS_STRONG,
    **FRENCH_PROMPTS_NEG,
}

N_TRAIN_SAMPLE = 20 if DEBUG else 1000


# ── Job type ─────────────────────────────────────────────────────────────────────

class PerplexityTokensFrenchInocJobParams(BaseModel):
    model:          str
    prompts:        dict[str, str]
    training_file:  str
    n_train_sample: int
    seed:           int = 42


@register("perplexity_tokens_french_inoc_job")
class PerplexityTokensFrenchInocJob(Jobs):
    mount = {
        "workers/worker_perplexity_tokens.py": "worker_perplexity_tokens.py"
        DATASET_TRAIN_PATH:            "data/train.jsonl",
    }
    params           = PerplexityTokensFrenchInocJobParams
    requires_vram_gb = 0

    def get_entrypoint(self, vp: BaseModel) -> str:
        return f"python worker_perplexity_tokens.py '{vp.model_dump_json()}'"


# ── Submit ───────────────────────────────────────────────────────────────────────

def submit() -> object:
    print("Submitting French inoculation per-token logprob job …", flush=True)
    print(f"  model          : {BASE_MODEL}", flush=True)
    print(f"  prompts ({len(ALL_PROMPTS)}): {list(ALL_PROMPTS.keys())}", flush=True)
    print(f"  n_train_sample : {N_TRAIN_SAMPLE}", flush=True)

    job = ow.perplexity_tokens_french_inoc_job.create(
        model             = BASE_MODEL,
        prompts           = ALL_PROMPTS,
        training_file     = "data/train.jsonl",
        n_train_sample    = N_TRAIN_SAMPLE,
        allowed_hardware  = ALLOWED_HARDWARE,
    )
    print(f"  job id: {job.id}  status: {job.status}", flush=True)
    return job


# ── Poll & download ──────────────────────────────────────────────────────────────

def wait_for_job(job) -> dict | None:
    print(f"\nPolling job {job.id} …", flush=True)
    while True:
        time.sleep(30)
        job = job.refresh()
        print(f"  [{time.strftime('%H:%M:%S')}] status: {job.status}", flush=True)

        if job.status == "completed":
            dst = f"/tmp/ow_perplexity_tokens_french_inoc_{job.id}/"
            os.makedirs(dst, exist_ok=True)
            try:
                job.download(dst)
                result_path = os.path.join(dst, "results", "perplexity_tokens_results.json")
                if os.path.exists(result_path):
                    print(f"  Loading result from {result_path} …", flush=True)
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


# ── Merge into existing tokens JSON ─────────────────────────────────────────────

def merge_into_existing(new_result: dict) -> None:
    """
    Merge French inoculation prompt token entries into the existing tokens JSON.
    Adds new prompt entries only; never replaces the shared baseline.
    """
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(
            f"Tokens results not found: {RESULTS_PATH}\n"
            "Run compute_perplexity_heuristic_tokens.py first."
        )

    print(f"\nLoading existing tokens JSON from {RESULTS_PATH} …", flush=True)
    with open(RESULTS_PATH) as f:
        tokens_data = json.load(f)

    existing_prompts = tokens_data.setdefault("prompts", {})
    new_prompts      = new_result.get("prompts", {})

    overlap = set(new_prompts) & set(existing_prompts)
    if overlap:
        print(f"  WARNING: overwriting existing token entries for: {overlap}", flush=True)

    n_added = 0
    for key, val in new_prompts.items():
        existing_prompts[key] = val
        n_toks = sum(len(t) for t in val.get("lp_train_inoc_tokens", []))
        print(f"  merged [{key}]: {len(val.get('lp_train_inoc_tokens', []))} examples, "
              f"{n_toks} tokens total", flush=True)
        n_added += 1

    print(f"\nSaving updated tokens JSON → {RESULTS_PATH} …", flush=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(tokens_data, f)
    size_mb = os.path.getsize(RESULTS_PATH) / 1e6
    print(f"✓ Added {n_added} French prompts to {RESULTS_PATH}  ({size_mb:.1f} MB)",
          flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    print("=== Per-Token Logprob — French Inoculation Prompts ===", flush=True)
    if DEBUG:
        print("  ⚠️  DEBUG MODE", flush=True)

    job    = submit()
    result = wait_for_job(job)

    if result is None:
        print("\n✗ Job failed or produced no results.", flush=True)
        return

    merge_into_existing(result)

    # Sanity check
    baseline_toks = result["baseline"]["lp_train_default_tokens"]
    print("\n── Sanity check: PH from token means ─────────────────────────────────")
    print(f"  {'Prompt key':<35}  {'PH_tokens':>10}  {'n_total_tok':>12}")
    for key, v in sorted(result["prompts"].items()):
        inoc_toks = v["lp_train_inoc_tokens"]
        ph_vals   = [float(np.mean(a)) - float(np.mean(b))
                     for a, b in zip(inoc_toks, baseline_toks) if a and b]
        ph  = float(np.mean(ph_vals)) if ph_vals else float("nan")
        ntok = sum(len(t) for t in inoc_toks)
        print(f"  {key:<35}  {ph:>+10.5f}  {ntok:>12}")

    print(f"\nMonitor : tail -f /tmp/perp_tokens_french_inoc.log")
    print(f"Results : {RESULTS_PATH}")
    print(f"Job     : {job.id}")


if __name__ == "__main__":
    main()
