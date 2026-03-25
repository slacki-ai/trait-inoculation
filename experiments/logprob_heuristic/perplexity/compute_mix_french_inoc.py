"""Mix PH for the 21 French inoculation prompts.

Computes per-example logprobs using index-matched rephrasings:
    lp_mix[n, k] = lp_per_tok(completion_k | rephrasings_n[k] + instruction_k)

Merges lp_train_mix into the existing perplexity_heuristic_{MODEL_SLUG}.json
for the 21 French prompt entries.  Requires the fixed PH job
(compute_perplexity_heuristic_french_inoc.py) to have run first.

Usage:
    python compute_perplexity_heuristic_mix_french_inoc.py > /tmp/perp_mix_french_inoc.log 2>&1 &
    tail -f /tmp/perp_mix_french_inoc.log
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

_debug_sfx           = "_debug" if DEBUG else ""
RESULTS_PATH         = f"results/perplexity_heuristic_{MODEL_SLUG}{_debug_sfx}.json"
REPHRASINGS_DIR      = "data/rephrasings"
REPHRASINGS_ALL_PATH = "/tmp/rephrasings_french_inoc_mix.json"

ALL_FRENCH_PROMPTS: dict[str, str] = {
    **FRENCH_PROMPTS,
    **FRENCH_PROMPTS_STRONG,
    **FRENCH_PROMPTS_NEG,
}
FRENCH_KEYS = sorted(ALL_FRENCH_PROMPTS.keys())
print(f"French inoculation keys ({len(FRENCH_KEYS)}): {FRENCH_KEYS}")

N_TRAIN_SAMPLE = 20 if DEBUG else 1000


# ── Pack rephrasings ────────────────────────────────────────────────────────────

def build_rephrasings_bundle() -> str:
    bundle: dict[str, list[str]] = {}
    for key in FRENCH_KEYS:
        path = os.path.join(REPHRASINGS_DIR, f"{key}.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing rephrasing file for key '{key}': {path}")
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
    print(f"French rephrasings bundle → {REPHRASINGS_ALL_PATH}  ({size_mb:.1f} MB)")
    return REPHRASINGS_ALL_PATH


# ── Job type ─────────────────────────────────────────────────────────────────────

class MixFrenchInocJobParams(BaseModel):
    model:            str
    keys:             list[str]
    rephrasings_file: str = "data/rephrasings_all.json"
    training_file:    str = "data/train.jsonl"
    n_train_sample:   int = 1000
    seed:             int = 42


@register("perplexity_mix_french_inoc_job")
class PerplexityMixFrenchInocJob(Jobs):
    mount = {
        "workers/worker_perplexity_mix.py": "worker_perplexity_mix.py"
        DATASET_TRAIN_PATH:         "data/train.jsonl",
        REPHRASINGS_ALL_PATH:       "data/rephrasings_all.json",
    }
    params           = MixFrenchInocJobParams
    requires_vram_gb = 0

    def get_entrypoint(self, vp: BaseModel) -> str:
        return f"python worker_perplexity_mix.py '{vp.model_dump_json()}'"


# ── Submit ───────────────────────────────────────────────────────────────────────

def submit() -> object:
    print("\nSubmitting French mix perplexity job …")
    print(f"  model          : {BASE_MODEL}")
    print(f"  keys ({len(FRENCH_KEYS)}): {FRENCH_KEYS}")
    print(f"  n_train_sample : {N_TRAIN_SAMPLE}")

    job = ow.perplexity_mix_french_inoc_job.create(
        model             = BASE_MODEL,
        keys              = FRENCH_KEYS,
        n_train_sample    = N_TRAIN_SAMPLE,
        allowed_hardware  = ALLOWED_HARDWARE,
    )
    print(f"  job id: {job.id}  status: {job.status}")
    return job


# ── Poll & download ──────────────────────────────────────────────────────────────

def wait_for_job(job) -> dict | None:
    print(f"\nPolling job {job.id} …")
    while True:
        time.sleep(30)
        job = job.refresh()
        print(f"  [{time.strftime('%H:%M:%S')}] status: {job.status}")

        if job.status == "completed":
            dst = f"/tmp/ow_perplexity_mix_french_inoc_{job.id}/"
            os.makedirs(dst, exist_ok=True)
            try:
                job.download(dst)
                result_path = os.path.join(dst, "results", "perplexity_mix_results.json")
                if os.path.exists(result_path):
                    with open(result_path) as f:
                        return json.load(f)
                else:
                    print(f"  WARNING: result file not at {result_path}")
                    for root, _, files in os.walk(dst):
                        for fn in files:
                            print(f"    {os.path.join(root, fn)}")
                    return None
            except Exception as e:
                print(f"  ERROR downloading: {e}")
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


# ── Merge results ────────────────────────────────────────────────────────────────

def merge_results(mix_results: dict) -> None:
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(
            f"{RESULTS_PATH} not found. Run compute_perplexity_heuristic_french_inoc.py first."
        )
    with open(RESULTS_PATH) as f:
        perp_data = json.load(f)

    n_merged = 0
    for key, vals in mix_results["prompts"].items():
        if key in perp_data["prompts"]:
            perp_data["prompts"][key]["lp_train_mix"] = vals["lp_train_mix"]
            n_merged += 1
            print(f"  merged [{key}]: {len(vals['lp_train_mix'])} values")
        else:
            print(f"  WARNING: [{key}] not in perp JSON — run fixed job first")

    with open(RESULTS_PATH, "w") as f:
        json.dump(perp_data, f, indent=2)
    print(f"\n✓ Merged {n_merged} French prompts (lp_train_mix) into {RESULTS_PATH}")


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    print("=== French Mix Perplexity Heuristic ===")
    if DEBUG:
        print("  ⚠️  DEBUG MODE")

    print("\n── Step 1: Building French rephrasings bundle …")
    build_rephrasings_bundle()

    print("\n── Step 2: Submitting OW job …")
    job = submit()

    print(f"\n── Step 3: Waiting for job {job.id} …")
    mix_results = wait_for_job(job)

    if mix_results is None:
        print("ERROR: job failed. Exiting.")
        return

    print("\n── Step 4: Merging results …")
    merge_results(mix_results)

    # Summary
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            perp_data = json.load(f)
        lp_default = np.array(perp_data["baseline"]["lp_train_default"][:N_TRAIN_SAMPLE])
        print(f"\n── Summary ──────────────────────────────────────────────────────────────")
        print(f"  {'key':<35}  {'PH (fixed)':>12}  {'PH (mix)':>12}")
        for key in FRENCH_KEYS:
            entry = perp_data["prompts"].get(key)
            if entry is None or "lp_train_mix" not in entry:
                continue
            lp_mix  = np.array(entry["lp_train_mix"])
            ph_mix  = float(np.nanmean(lp_mix - lp_default))
            ph_fixed = entry.get("perplexity_heuristic", float("nan"))
            print(f"  {key:<35}  {ph_fixed:>+12.5f}  {ph_mix:>+12.5f}")

    print("\nDone.")
    print(f"Monitor : tail -f /tmp/perp_mix_french_inoc.log")
    print(f"Results : {RESULTS_PATH}")
    print(f"Job     : {job.id}")


if __name__ == "__main__":
    main()
