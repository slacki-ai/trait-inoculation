"""Inoculation prefix sweep — batch 2: 6 additional runs.

Three new prefix conditions × 2 LRs:

  1. neutral_prefix  — fixed "Give an answer to the following:"
  2. weak_mix        — random sample from 1000 rephrasings of "I had fun today."
  3. strong_mix      — random sample from 1000 rephrasings of "You are a playful agent…"

Results are appended to the same JSON as batch 1 so all 12 runs can be
plotted together.

Usage:
    python train_inoculation_prefix_sweep2.py > /tmp/inoc_prefix_sweep2.log 2>&1 &
    tail -f /tmp/inoc_prefix_sweep2.log
"""

import json
import os
import time

from openweights import OpenWeights, register, Jobs
from pydantic import BaseModel

from config import (
    DEBUG,
    UNSLOTH_MODEL,
    MODEL_SLUG,
    N_TRAIN,
    N_EVAL,
    TRAINING_HYPERPARAMS,
    TOTAL_TRAINING_STEPS,
    POSITIVE_TRAIT,
    NEGATIVE_TRAIT,
    DATASET_TRAIN_PATH,
    DATASET_EVAL_PATH,
    REQUIRES_VRAM_GB,
)
from utils.data import load_eval_instructions
from utils.judge import judge_completions
from utils.ow import download_completions, get_failure_logs, fetch_and_parse_loss
from utils.plot import run_plot_module

ow = OpenWeights()

_debug_sfx     = "_debug" if DEBUG else ""
# Append to the same results file as batch 1
RESULTS_PATH   = f"results/scores_inoc_prefix_sweep_{MODEL_SLUG}{_debug_sfx}.json"
LOSSES_PATH    = f"results/losses_inoc_prefix_sweep_{MODEL_SLUG}{_debug_sfx}.json"
PLOT_PATH      = f"plots/inoc_prefix_sweep_{MODEL_SLUG}{_debug_sfx}.png"
LOSS_PLOT_PATH = f"plots/losses_inoc_prefix_sweep_{MODEL_SLUG}{_debug_sfx}.png"
TRAITS         = [POSITIVE_TRAIT, NEGATIVE_TRAIT]

WEAK_REPHRASINGS_PATH   = "data/weak_inoc_rephrasings.json"
STRONG_REPHRASINGS_PATH = "data/strong_inoc_rephrasings.json"

os.makedirs("results", exist_ok=True)
os.makedirs("plots",   exist_ok=True)

NEUTRAL_PREFIX = "Give an answer to the following:"

# ── Run configs ────────────────────────────────────────────────────────────────
# "fixed" runs reuse the existing InocPrefixSweepJob (single fixed prefix).
# "mix"   runs use InocMixSweepJob (pool of rephrasings sampled per example).
RUNS: dict[str, dict] = {
    "neutral_prefix_lr_1e4": {
        "type": "fixed", "lr": 1e-4,  "user_prefix": NEUTRAL_PREFIX,
    },
    "neutral_prefix_lr_5e6": {
        "type": "fixed", "lr": 5e-6,  "user_prefix": NEUTRAL_PREFIX,
    },
    "weak_mix_lr_1e4": {
        "type": "mix",   "lr": 1e-4,  "rephrasings_path": WEAK_REPHRASINGS_PATH,
    },
    "weak_mix_lr_5e6": {
        "type": "mix",   "lr": 5e-6,  "rephrasings_path": WEAK_REPHRASINGS_PATH,
    },
    "strong_mix_lr_1e4": {
        "type": "mix",   "lr": 1e-4,  "rephrasings_path": STRONG_REPHRASINGS_PATH,
    },
    "strong_mix_lr_5e6": {
        "type": "mix",   "lr": 5e-6,  "rephrasings_path": STRONG_REPHRASINGS_PATH,
    },
}

# ── Eval schedule (same as batch 1) ───────────────────────────────────────────
def make_eval_steps(total: int = TOTAL_TRAINING_STEPS) -> list[int]:
    steps: set[int] = {0}
    steps.update(range(5, 51, 5))
    steps.update(range(60, 101, 10))
    steps.update(range(120, 251, 20))
    steps.add(250)
    steps.add(total)
    return sorted(s for s in steps if s <= total)

EVAL_STEPS = make_eval_steps()
print(f"Eval schedule ({len(EVAL_STEPS)} points): {EVAL_STEPS}")


# ── Job: fixed prefix (reuses worker_train_prefix.py) ─────────────────────────
class InocPrefixSweepParams(BaseModel):
    model:         str
    training_file: str
    eval_file:     str
    user_prefix:   str
    total_steps:   int
    hyperparams:   dict
    eval_steps:    list[int]
    n_train:       int = 0
    n_eval:        int = 0


@register("inoc_prefix_sweep_v1")
class InocPrefixSweepJob(Jobs):
    mount = {
        "worker_train_prefix.py":      "worker_train_prefix.py",
        "worker_vllm_infer_prefix.py": "worker_vllm_infer_prefix.py",
        DATASET_TRAIN_PATH:            "data/train.jsonl",
        DATASET_EVAL_PATH:             "data/eval.jsonl",
    }
    params           = InocPrefixSweepParams
    requires_vram_gb = REQUIRES_VRAM_GB

    def get_entrypoint(self, vp: BaseModel) -> str:
        return f"python worker_train_prefix.py '{vp.model_dump_json()}'"


# ── Job: mix of rephrasings ────────────────────────────────────────────────────
class InocMixSweepParams(BaseModel):
    model:            str
    training_file:    str
    eval_file:        str
    rephrasings_file: str   # mounted path inside the job (always "data/rephrasings.json")
    total_steps:      int
    hyperparams:      dict
    eval_steps:       list[int]
    n_train:          int = 0
    n_eval:           int = 0


def make_mix_job(rephrasings_local_path: str):
    """Return a new InocMixSweepJob class with the given rephrasings file mounted."""

    @register(f"inoc_mix_sweep_v1_{os.path.basename(rephrasings_local_path).split('.')[0]}")
    class InocMixSweepJob(Jobs):
        mount = {
            "worker_train_prefix_mix.py":      "worker_train_prefix_mix.py",
            "worker_vllm_infer_prefix_mix.py": "worker_vllm_infer_prefix_mix.py",
            DATASET_TRAIN_PATH:                "data/train.jsonl",
            DATASET_EVAL_PATH:                 "data/eval.jsonl",
            rephrasings_local_path:            "data/rephrasings.json",
        }
        params           = InocMixSweepParams
        requires_vram_gb = REQUIRES_VRAM_GB

        def get_entrypoint(self, vp: BaseModel) -> str:
            return f"python worker_train_prefix_mix.py '{vp.model_dump_json()}'"

    return InocMixSweepJob


# Register both mix job types up front so they're available on ow.*
_weak_mix_job   = make_mix_job(WEAK_REPHRASINGS_PATH)
_strong_mix_job = make_mix_job(STRONG_REPHRASINGS_PATH)

JOB_TYPE_ATTR = {
    "weak":   f"inoc_mix_sweep_v1_{os.path.basename(WEAK_REPHRASINGS_PATH).split('.')[0]}",
    "strong": f"inoc_mix_sweep_v1_{os.path.basename(STRONG_REPHRASINGS_PATH).split('.')[0]}",
}


# ── Submit ─────────────────────────────────────────────────────────────────────
def submit_all() -> dict[str, object]:
    print(f"Submitting {len(RUNS)} jobs …")
    jobs: dict[str, object] = {}
    base_hp = dict(TRAINING_HYPERPARAMS)

    for run_name, cfg in RUNS.items():
        hp = {**base_hp, "learning_rate": cfg["lr"]}

        if cfg["type"] == "fixed":
            job = ow.inoc_prefix_sweep_v1.create(
                model         = UNSLOTH_MODEL,
                training_file = "data/train.jsonl",
                eval_file     = "data/eval.jsonl",
                user_prefix   = cfg["user_prefix"],
                total_steps   = TOTAL_TRAINING_STEPS,
                hyperparams   = hp,
                eval_steps    = EVAL_STEPS,
                n_train       = N_TRAIN,
                n_eval        = 10 if DEBUG else 50,
            )
            print(f"  [{run_name}] lr={cfg['lr']:.0e}  "
                  f"prefix={cfg['user_prefix']!r}  "
                  f"job={job.id}  status={job.status}")

        else:  # mix
            rpath = cfg["rephrasings_path"]
            mix_key = "weak" if "weak" in rpath else "strong"
            job_attr = JOB_TYPE_ATTR[mix_key]
            job = getattr(ow, job_attr).create(
                model            = UNSLOTH_MODEL,
                training_file    = "data/train.jsonl",
                eval_file        = "data/eval.jsonl",
                rephrasings_file = "data/rephrasings.json",
                total_steps      = TOTAL_TRAINING_STEPS,
                hyperparams      = hp,
                eval_steps       = EVAL_STEPS,
                n_train          = N_TRAIN,
                n_eval           = 10 if DEBUG else 50,
            )
            print(f"  [{run_name}] lr={cfg['lr']:.0e}  "
                  f"mix={os.path.basename(rpath)}  "
                  f"job={job.id}  status={job.status}")

        jobs[run_name] = job

    return jobs


# ── Poll ───────────────────────────────────────────────────────────────────────
def poll_until_done(jobs: dict) -> dict:
    # Load existing results so we append rather than overwrite batch 1
    results: dict = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        print(f"  Loaded {len(results)} existing results from {RESULTS_PATH}")

    pending = dict(jobs)
    eval_instrs = load_eval_instructions(DATASET_EVAL_PATH, limit=N_EVAL)

    while pending:
        time.sleep(60)
        done_this_round: list[str] = []

        for run_name, job in list(pending.items()):
            job = job.refresh()

            if job.status == "completed":
                done_this_round.append(run_name)
                rows = download_completions(
                    job,
                    f"/tmp/ow_outputs_inoc2_{run_name}/",
                    label=run_name,
                )
                cfg = RUNS[run_name]
                if rows:
                    print(f"  [{run_name}] Judging …")
                    steps = judge_completions(rows, TRAITS, eval_instructions=eval_instrs)
                    entry: dict = {"lr": cfg["lr"], "steps": steps}
                    if cfg["type"] == "fixed":
                        entry["user_prefix"] = cfg["user_prefix"]
                    else:
                        entry["rephrasings_path"] = cfg["rephrasings_path"]
                    results[run_name] = entry
                else:
                    results[run_name] = {"error": "download failed", "lr": cfg["lr"]}
                with open(RESULTS_PATH, "w") as f:
                    json.dump(results, f, indent=2)
                n_new = sum(1 for k in results if k in RUNS)
                print(f"  → {n_new}/{len(jobs)} new done: {RESULTS_PATH}")

            elif job.status == "failed":
                done_this_round.append(run_name)
                logs = get_failure_logs(ow, job, max_chars=2000)
                print(f"  [{run_name}] FAILED" + (f":\n{logs}" if logs else " (no logs)"))
                results[run_name] = {"error": "job failed", "lr": RUNS[run_name]["lr"]}
                with open(RESULTS_PATH, "w") as f:
                    json.dump(results, f, indent=2)

        for r in done_this_round:
            del pending[r]
        if pending:
            print(f"  Still running: { {n: j.status for n, j in pending.items()} }")

    return results


# ── Loss fetch ─────────────────────────────────────────────────────────────────
def fetch_and_save_losses(jobs: dict) -> None:
    losses: dict = {}
    if os.path.exists(LOSSES_PATH):
        with open(LOSSES_PATH) as f:
            losses = json.load(f)

    for run_name, job in jobs.items():
        dst = f"/tmp/ow_outputs_inoc2_{run_name}/"
        loss_data = fetch_and_parse_loss(ow, job, dst=dst)
        if loss_data:
            losses[run_name] = loss_data
            print(f"  [{run_name}] {len(loss_data)} loss points fetched")
        else:
            print(f"  [{run_name}] no loss data")

    if losses:
        with open(LOSSES_PATH, "w") as f:
            json.dump(losses, f, indent=2)
        print(f"  ✓ Losses saved → {LOSSES_PATH}")
        run_plot_module("plot_losses.py", LOSSES_PATH, LOSS_PLOT_PATH)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"=== Inoculation Prefix Sweep — Batch 2 [{MODEL_SLUG}] ===")
    if DEBUG:
        print(f"  ⚠️  DEBUG MODE: N_TRAIN={N_TRAIN}, N_EVAL={N_EVAL}")
    print(f"  Runs  : {list(RUNS.keys())}")
    print(f"  Steps : {TOTAL_TRAINING_STEPS}")
    print(f"  VRAM  : {REQUIRES_VRAM_GB} GB\n")

    jobs = submit_all()
    print(f"\nAll {len(jobs)} jobs submitted. Polling every 60s …\n")
    poll_until_done(jobs)

    print(f"\n✓ Results → {RESULTS_PATH}")
    run_plot_module("plot_inoc_prefix_sweep.py", RESULTS_PATH)

    print("\n── Fetching training losses …")
    fetch_and_save_losses(jobs)


if __name__ == "__main__":
    main()
