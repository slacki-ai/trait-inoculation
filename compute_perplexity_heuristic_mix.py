"""compute_perplexity_heuristic_mix.py — Orchestrator for per-example mix logprobs.

Launches a single OW GPU job (worker_perplexity_mix.py) that computes, for every
prompt key, the per-example logprobs using index-matched rephrasings:

    lp_mix[n, k] = lp_per_tok(completion_k | rephrasings_n[k] + instruction_k)

All 27 rephrasing pools are packed into a single JSON file
(data/rephrasings_all.json) and mounted into the job.

On completion, the downloaded lp_train_mix arrays are merged into the existing
    results/perplexity_heuristic_{MODEL_SLUG}.json
by adding a "lp_train_mix" field to each prompt entry.

Usage:
    python compute_perplexity_heuristic_mix.py > /tmp/perplexity_mix.log 2>&1 &
    tail -f /tmp/perplexity_mix.log
"""

import json
import os
import time

from openweights import OpenWeights, register, Jobs
from pydantic import BaseModel

from config import BASE_MODEL, MODEL_SLUG, REQUIRES_VRAM_GB, DEBUG

ow = OpenWeights()

_debug_sfx   = "_debug" if DEBUG else ""
RESULTS_PATH = f"results/perplexity_heuristic_{MODEL_SLUG}{_debug_sfx}.json"
REPHRASINGS_DIR = "data/rephrasings"
REPHRASINGS_ALL_PATH = "/tmp/rephrasings_all.json"

# All prompt keys that have rephrasing files
ALL_KEYS = sorted(
    fname.replace(".jsonl", "")
    for fname in os.listdir(REPHRASINGS_DIR)
    if fname.endswith(".jsonl")
)
print(f"Found {len(ALL_KEYS)} rephrasing pools: {ALL_KEYS}")

N_TRAIN_SAMPLE = 20 if DEBUG else 1000


# ── Pack rephrasings ───────────────────────────────────────────────────────────

def build_rephrasings_bundle() -> str:
    """
    Read all {key}.jsonl files and combine into a single JSON dict:
        {key: [rephrasing_0, rephrasing_1, ...]}
    Saved to REPHRASINGS_ALL_PATH for mounting.
    """
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

class MixPerplexityJobParams(BaseModel):
    model:            str
    keys:             list[str]
    rephrasings_file: str = "data/rephrasings_all.json"
    training_file:    str = "data/train.jsonl"
    n_train_sample:   int = 1000
    seed:             int = 42


@register("perplexity_mix_job")
class PerplexityMixJob(Jobs):
    mount = {
        "worker_perplexity_mix.py":                    "worker_perplexity_mix.py",
        f"data/train_{MODEL_SLUG}{_debug_sfx}.jsonl":  "data/train.jsonl",
        REPHRASINGS_ALL_PATH:                          "data/rephrasings_all.json",
    }
    params           = MixPerplexityJobParams
    requires_vram_gb = REQUIRES_VRAM_GB

    def get_entrypoint(self, vp: BaseModel) -> str:
        return f"python worker_perplexity_mix.py '{vp.model_dump_json()}'"


# ── Submit ─────────────────────────────────────────────────────────────────────

def submit() -> object:
    print(f"\nSubmitting mix perplexity job …")
    print(f"  model          : {BASE_MODEL}")
    print(f"  keys           : {ALL_KEYS}")
    print(f"  n_train_sample : {N_TRAIN_SAMPLE}")

    job = ow.perplexity_mix_job.create(
        model          = BASE_MODEL,
        keys           = ALL_KEYS,
        n_train_sample = N_TRAIN_SAMPLE,
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
            dst = f"/tmp/ow_perplexity_mix_{job.id}/"
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


# ── Merge results into existing JSON ──────────────────────────────────────────

def merge_results(mix_results: dict) -> None:
    """
    Add lp_train_mix to each prompt entry in the existing perplexity heuristic JSON.
    """
    if not os.path.exists(RESULTS_PATH):
        print(f"ERROR: existing results not found at {RESULTS_PATH}")
        return

    with open(RESULTS_PATH) as f:
        perp_data = json.load(f)

    n_merged = 0
    for key, vals in mix_results["prompts"].items():
        if key in perp_data["prompts"]:
            perp_data["prompts"][key]["lp_train_mix"] = vals["lp_train_mix"]
            n_merged += 1
            print(f"  merged [{key}]: {len(vals['lp_train_mix'])} values")
        else:
            print(f"  WARNING: [{key}] not in existing perp JSON — skipping")

    with open(RESULTS_PATH, "w") as f:
        json.dump(perp_data, f, indent=2)

    print(f"\n✓ Merged {n_merged} prompts into {RESULTS_PATH}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== Perplexity Heuristic — Mix Version ===")

    # 1. Pack rephrasings
    print("\n── Step 1: Building rephrasings bundle …")
    build_rephrasings_bundle()

    # 2. Submit job
    print("\n── Step 2: Submitting OW job …")
    job = submit()

    # 3. Wait
    print(f"\n── Step 3: Waiting for job {job.id} …")
    mix_results = wait_for_job(job)

    if mix_results is None:
        print("ERROR: job failed or produced no results. Exiting.")
        return

    # 4. Merge into existing JSON
    print("\n── Step 4: Merging results …")
    merge_results(mix_results)

    # 5. Print summary
    print("\n── Summary ────────────────────────────────────────────────")
    if "prompts" in mix_results and os.path.exists(RESULTS_PATH):
        import numpy as np
        with open(RESULTS_PATH) as f:
            perp_data = json.load(f)
        lp_default = np.array(perp_data["baseline"]["lp_train_default"][:N_TRAIN_SAMPLE])
        print(f"  {'key':<35}  {'PH (fixed)':>12}  {'PH (mix)':>12}")
        for key, entry in perp_data["prompts"].items():
            if "lp_train_mix" not in entry:
                continue
            lp_mix   = np.array(entry["lp_train_mix"])
            lp_def   = np.array(perp_data["baseline"]["lp_train_default"])
            ph_mix   = float(np.nanmean(lp_mix - lp_def))
            ph_fixed = entry.get("perplexity_heuristic", float("nan"))
            print(f"  {key:<35}  {ph_fixed:>+12.5f}  {ph_mix:>+12.5f}")
    elif "prompts" in mix_results:
        # No existing perplexity JSON to compare against (e.g. debug mode first run)
        import numpy as np
        print(f"  {'key':<35}  {'mean lp_mix':>12}  {'n':>5}")
        for key, v in mix_results["prompts"].items():
            vals = [x for x in v["lp_train_mix"] if x == x]
            print(f"  {key:<35}  {np.mean(vals):>+12.5f}  {len(vals):>5}")

    print("\nDone.")


if __name__ == "__main__":
    main()
