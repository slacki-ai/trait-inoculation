"""Orchestrator: Per-Token Logprob Heuristic.

Launches a single OW GPU job (worker_perplexity_tokens.py) that computes
per-token log-probabilities for the base model on all 27 inoculation prompts
and the no-prefix baseline.

The output supports building a N × (K·L) matrix for token-level PCA:
    W_tokens[n, k, l] = lp_inoc_tokens[n][k][l] − lp_default_tokens[k][l]

This captures *which tokens* each prefix affects, not just the mean shift.
Two prompts with equal PH can have different token-level patterns, which the
mean-based W[N×K] matrix cannot distinguish.

Output
──────
  results/perplexity_heuristic_tokens_{MODEL_SLUG}.json
  {
    "params":   {...},
    "baseline": {
        "lp_train_default_tokens": [[float, ...], ...]   # K lists
    },
    "prompts": {
        "<key>": {
            "lp_train_inoc_tokens": [[float, ...], ...]  # K lists
        },
        ...
    }
  }

Usage
─────
  python compute_perplexity_heuristic_tokens.py          > /tmp/perp_tokens.log 2>&1 &
  DEBUG=1 python compute_perplexity_heuristic_tokens.py  > /tmp/perp_tokens.log 2>&1 &
  tail -f /tmp/perp_tokens.log
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
    DATASET_TRAIN_PATH,
    REQUIRES_VRAM_GB,
    ALLOWED_HARDWARE,
    DEBUG,
    INOCULATION_PROMPTS,
    INOCULATION_PROMPTS_STRONG,
    INOCULATION_PROMPTS_ZERO,
    INOCULATION_PROMPTS_NEG,
)

ow = OpenWeights()

_debug_sfx   = "_debug" if DEBUG else ""
RESULTS_PATH = f"results/perplexity_heuristic_tokens_{MODEL_SLUG}{_debug_sfx}.json"
os.makedirs("results", exist_ok=True)

# All 27 prompts: v3 (9) + v4 (6) + v5 (6) + neg (6)
ALL_PROMPTS: dict[str, str] = {
    **INOCULATION_PROMPTS,
    **INOCULATION_PROMPTS_STRONG,
    **INOCULATION_PROMPTS_ZERO,
    **INOCULATION_PROMPTS_NEG,
}

# In DEBUG mode: tiny subsample (fast smoke test)
N_TRAIN_SAMPLE = 20 if DEBUG else 1000


# ── Job type ─────────────────────────────────────────────────────────────────────

class PerplexityTokensJobParams(BaseModel):
    model:          str
    prompts:        dict[str, str]
    training_file:  str
    n_train_sample: int
    seed:           int = 42


@register("perplexity_tokens_job")
class PerplexityTokensJob(Jobs):
    mount = {
        "workers/worker_perplexity_tokens.py": "worker_perplexity_tokens.py",
        DATASET_TRAIN_PATH:            "data/train.jsonl",
    }
    params           = PerplexityTokensJobParams
    requires_vram_gb = 0

    def get_entrypoint(self, vp: BaseModel) -> str:
        return f"python worker_perplexity_tokens.py '{vp.model_dump_json()}'"


# ── Submit ────────────────────────────────────────────────────────────────────────

def submit() -> object:
    print("Submitting per-token logprob job …", flush=True)
    print(f"  model          : {BASE_MODEL}", flush=True)
    print(f"  prompts        : {list(ALL_PROMPTS.keys())}", flush=True)
    print(f"  n_train_sample : {N_TRAIN_SAMPLE}", flush=True)

    job = ow.perplexity_tokens_job.create(
        model             = BASE_MODEL,
        prompts           = ALL_PROMPTS,
        training_file     = "data/train.jsonl",
        n_train_sample    = N_TRAIN_SAMPLE,
        allowed_hardware  = ALLOWED_HARDWARE,
    )
    print(f"  job id: {job.id}  status: {job.status}", flush=True)
    return job


# ── Poll & download ────────────────────────────────────────────────────────────────

def wait_for_job(job) -> dict | None:
    print(f"\nPolling job {job.id} …", flush=True)
    while True:
        time.sleep(30)
        job = job.refresh()
        print(f"  [{time.strftime('%H:%M:%S')}] status: {job.status}", flush=True)

        if job.status == "completed":
            dst = f"/tmp/ow_perplexity_tokens_{job.id}/"
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
                    for root, _dirs, files in os.walk(dst):
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


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    print("=== Per-Token Logprob Heuristic ===", flush=True)
    if DEBUG:
        print("  ⚠️  DEBUG MODE", flush=True)

    job    = submit()
    result = wait_for_job(job)

    if result is None:
        print("\n✗ Job failed or produced no results.", flush=True)
        return

    # Save directly — the JSON is large, no further merging needed
    print(f"\nSaving to {RESULTS_PATH} …", flush=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(result, f)
    size_mb = os.path.getsize(RESULTS_PATH) / 1e6
    print(f"✓ Saved → {RESULTS_PATH}  ({size_mb:.1f} MB)", flush=True)

    # Summary: print PH from token means as a sanity check
    import numpy as np
    baseline_toks = result["baseline"]["lp_train_default_tokens"]
    prompts_data  = result["prompts"]

    print("\n── Sanity check: PH from token means ────────────────────────────────")
    print(f"  {'Prompt key':<35}  {'PH_tokens':>10}  {'n_total_tok':>12}")
    for key, v in sorted(prompts_data.items(),
                         key=lambda x: sum(len(t) for t in x[1]["lp_train_inoc_tokens"]),
                         reverse=True):
        inoc_toks = v["lp_train_inoc_tokens"]
        ph_vals   = [float(np.mean(a)) - float(np.mean(b))
                     for a, b in zip(inoc_toks, baseline_toks) if a and b]
        ph        = float(np.mean(ph_vals)) if ph_vals else float("nan")
        ntok      = sum(len(t) for t in inoc_toks)
        print(f"  {key:<35}  {ph:>+10.5f}  {ntok:>12}")

    if not DEBUG:
        print(f"\n  W_tokens matrix shape: {len(prompts_data)} × "
              f"{sum(len(t) for t in baseline_toks)} features")

    print(f"\nMonitor : tail -f /tmp/perp_tokens.log")
    print(f"Results : {RESULTS_PATH}")
    print(f"Job     : {job.id}")


if __name__ == "__main__":
    main()
