"""compute_perplexity_heuristic_mix_tokens.py — Orchestrator for per-token mix logprobs.

Launches a single OW GPU job (worker_perplexity_mix_tokens.py) that computes,
for every prompt key, the per-token log-probabilities using index-matched rephrasings:

    lp_mix_tokens[n][k] = [log p(t_0 | ctx), log p(t_1 | ctx, t_0), ...]
                           where ctx uses rephrasings_n[k % len(rephrasings_n)]

This is the per-token analogue of compute_perplexity_heuristic_mix.py and is
needed to run PCA on W_mix_tokens (alongside the already-computed W_fixed_tokens).

On completion, lp_train_mix_tokens arrays are merged into the existing
    results/perplexity_heuristic_tokens_{MODEL_SLUG}.json
by adding a "lp_train_mix_tokens" field to each prompt entry.  The baseline
(lp_train_default_tokens) is shared and not recomputed.

Usage:
    python compute_perplexity_heuristic_mix_tokens.py > /tmp/perp_mix_tokens.log 2>&1 &
    tail -f /tmp/perp_mix_tokens.log
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '../../..'))
import json
import os
import time

from openweights import OpenWeights, register, Jobs
from pydantic import BaseModel

from config import BASE_MODEL, MODEL_SLUG, REQUIRES_VRAM_GB, ALLOWED_HARDWARE, DEBUG, DATASET_TRAIN_PATH

ow = OpenWeights()

_debug_sfx   = "_debug" if DEBUG else ""
# Merge into the fixed-tokens JSON (adds lp_train_mix_tokens alongside lp_train_inoc_tokens)
TOKENS_PATH = f"results/perplexity_heuristic_tokens_{MODEL_SLUG}{_debug_sfx}.json"

REPHRASINGS_DIR      = "data/rephrasings"
REPHRASINGS_ALL_PATH = "/tmp/rephrasings_all_mix_tokens.json"   # local temp bundle

# All prompt keys that have rephrasing files
ALL_KEYS = sorted(
    fname.replace(".jsonl", "")
    for fname in os.listdir(REPHRASINGS_DIR)
    if fname.endswith(".jsonl")
)
print(f"Found {len(ALL_KEYS)} rephrasing pools: {ALL_KEYS}")

N_TRAIN_SAMPLE = 20 if DEBUG else 1000


# ── Pack rephrasings bundle ────────────────────────────────────────────────────

def build_rephrasings_bundle() -> str:
    bundle: dict[str, list[str]] = {}
    for key in ALL_KEYS:
        path = os.path.join(REPHRASINGS_DIR, f"{key}.jsonl")
        rephrasings = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    rephrasings.append(json.loads(line)["rephrasing"])
        bundle[key] = rephrasings
        print(f"  [{key}] {len(rephrasings)} rephrasings loaded")

    with open(REPHRASINGS_ALL_PATH, "w") as f:
        json.dump(bundle, f)

    size_mb = os.path.getsize(REPHRASINGS_ALL_PATH) / 1e6
    print(f"Rephrasings bundle saved → {REPHRASINGS_ALL_PATH}  ({size_mb:.1f} MB)")
    return REPHRASINGS_ALL_PATH


# ── Job type ───────────────────────────────────────────────────────────────────

class MixTokensJobParams(BaseModel):
    model:            str
    keys:             list[str]
    rephrasings_file: str = "data/rephrasings_all.json"
    training_file:    str = "data/train.jsonl"
    n_train_sample:   int = 1000
    seed:             int = 42


@register("perplexity_mix_tokens_job")
class PerplexityMixTokensJob(Jobs):
    mount = {
        "workers/worker_perplexity_mix_tokens.py":            "worker_perplexity_mix_tokens.py",
        DATASET_TRAIN_PATH:                           "data/train.jsonl",
        REPHRASINGS_ALL_PATH:                         "data/rephrasings_all.json",
    }
    params           = MixTokensJobParams
    requires_vram_gb = 0

    def get_entrypoint(self, vp: BaseModel) -> str:
        return f"python worker_perplexity_mix_tokens.py '{vp.model_dump_json()}'"


# ── Submit ─────────────────────────────────────────────────────────────────────

def submit(rephrasings_bundle_path: str) -> object:
    print(f"\nSubmitting per-token mix logprob job …")
    print(f"  model          : {BASE_MODEL}")
    print(f"  keys           : {ALL_KEYS}")
    print(f"  n_train_sample : {N_TRAIN_SAMPLE}")

    job = ow.perplexity_mix_tokens_job.create(
        model             = BASE_MODEL,
        keys              = ALL_KEYS,
        n_train_sample    = N_TRAIN_SAMPLE,
        allowed_hardware  = ALLOWED_HARDWARE,
    )
    print(f"  job id: {job.id}  status: {job.status}")
    return job


# ── Poll & download ────────────────────────────────────────────────────────────

def wait_for_job(job) -> dict | None:
    print(f"\nPolling job {job.id} …")
    while True:
        time.sleep(30)
        job = job.refresh()
        print(f"  [{time.strftime('%H:%M:%S')}] status: {job.status}")

        if job.status == "completed":
            dst = f"/tmp/ow_perplexity_mix_tokens_{job.id}/"
            os.makedirs(dst, exist_ok=True)
            try:
                job.download(dst)
                result_path = os.path.join(
                    dst, "results", "perplexity_mix_tokens_results.json"
                )
                if os.path.exists(result_path):
                    print(f"  Loading result from {result_path} …")
                    with open(result_path) as f:
                        return json.load(f)
                else:
                    print(f"  WARNING: result file not found at {result_path}")
                    for root, _, files in os.walk(dst):
                        for fn in files:
                            print(f"    {os.path.join(root, fn)}")
                    return None
            except Exception as e:
                print(f"  ERROR downloading results: {e}")
                return None

        elif job.status == "failed":
            print("  Job FAILED.")
            try:
                events = ow._supabase.table("events").select("*").eq(
                    "run_id", job.id
                ).order("created_at", desc=True).limit(20).execute()
                for ev in reversed(events.data or []):
                    print(f"    {ev.get('created_at','')}  {ev.get('data','')}")
            except Exception as e:
                print(f"  (could not fetch events: {e})")
            return None


# ── Merge into tokens JSON ─────────────────────────────────────────────────────

def merge_results(mix_tokens_results: dict) -> None:
    """
    Add lp_train_mix_tokens to each prompt entry in the existing tokens JSON.
    """
    if not os.path.exists(TOKENS_PATH):
        print(f"ERROR: tokens results not found at {TOKENS_PATH}")
        print("  Run compute_perplexity_heuristic_tokens.py first.")
        return

    print(f"\nLoading existing tokens JSON from {TOKENS_PATH} …")
    with open(TOKENS_PATH) as f:
        tokens_data = json.load(f)

    n_merged = 0
    for key, vals in mix_tokens_results["prompts"].items():
        if key in tokens_data["prompts"]:
            tokens_data["prompts"][key]["lp_train_mix_tokens"] = vals["lp_train_mix_tokens"]
            n_merged += 1
            n_examples = len(vals["lp_train_mix_tokens"])
            n_tokens   = sum(len(t) for t in vals["lp_train_mix_tokens"])
            print(f"  merged [{key}]: {n_examples} examples, {n_tokens} tokens total")
        else:
            print(f"  WARNING: [{key}] not in tokens JSON — skipping")

    print(f"\nSaving updated tokens JSON → {TOKENS_PATH} …")
    with open(TOKENS_PATH, "w") as f:
        json.dump(tokens_data, f)
    size_mb = os.path.getsize(TOKENS_PATH) / 1e6
    print(f"✓ Merged {n_merged} prompts into {TOKENS_PATH}  ({size_mb:.1f} MB)")


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    print("=== Per-Token Mix Logprob Heuristic ===")
    if DEBUG:
        print("  ⚠️  DEBUG MODE")

    print("\n── Building rephrasings bundle …")
    rephrasings_path = build_rephrasings_bundle()

    job    = submit(rephrasings_path)
    result = wait_for_job(job)

    if result is None:
        print("\n✗ Job failed or produced no results.")
        return

    merge_results(result)

    # Sanity check: print mean of lp_train_mix_tokens vs existing lp_train_inoc_tokens
    import numpy as np
    print("\n── Sanity check: mean(lp_train_mix_tokens) per prompt ───────────────")
    print(f"  {'key':<35}  {'mean lp_mix_tokens':>18}  {'n_tokens':>9}")
    for key, v in sorted(result["prompts"].items()):
        toks_list = v["lp_train_mix_tokens"]
        means = [float(np.mean(t)) for t in toks_list if t]
        ph    = float(np.mean(means)) if means else float("nan")
        ntok  = sum(len(t) for t in toks_list)
        print(f"  {key:<35}  {ph:>+18.5f}  {ntok:>9}")

    print(f"\nMonitor : tail -f /tmp/perp_mix_tokens.log")
    print(f"Results : {TOKENS_PATH}")
    print(f"Job     : {job.id}")


if __name__ == "__main__":
    main()
