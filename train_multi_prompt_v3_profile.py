"""Multi-Prompt Profile Experiment — Experiment 6.

Goal: measure the full trait expression profile over training for all 9
inoculation prompts, using the mix (rephrasing pool) condition only.

Design
──────
10 runs at LR=1e-4, evaluated at ~27 densely-spaced checkpoints:
  0, 5, 10, …, 50, 60, 70, …, 100, 120, 140, …, 250, 312

  1  × control  — no user prefix (fixed worker, prefix="")
  9  × mix      — one per INOCULATION_PROMPTS key, 1000 rephrasings pool

System prompt: ALWAYS the Qwen default.
Inoculation:   user-turn prefix, sampled per training example from rephrasing pool.

Eval per checkpoint (two conditions):
  "default"  — user turn = "[instruction]"  (no prefix, same across all runs)
  "training" — each instruction paired with a seeded-random rephrasing from pool

Workers: worker_train_prefix_mix.py (Phase 1) + worker_vllm_infer_prefix_mix.py
(Phase 2, vLLM subprocess). LoRA checkpoints saved at every eval step; vLLM
hot-swaps them after training — no padding artifacts.

Prerequisites
─────────────
data/rephrasings/{key}.jsonl must exist (run generate_rephrasings.py first).

Usage:
    python train_multi_prompt_v3_profile.py > /tmp/mp3_profile.log 2>&1 &
    tail -f /tmp/mp3_profile.log
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
    INOCULATION_PROMPTS,
)
from utils.data import load_eval_instructions
from utils.judge import judge_completions
from utils.ow import download_completions, get_failure_logs, fetch_and_parse_loss
from utils.plot import run_plot_module

ow = OpenWeights()

_debug_sfx     = "_debug" if DEBUG else ""
RESULTS_PATH   = f"results/scores_multi_prompt_v3_profile_{MODEL_SLUG}{_debug_sfx}.json"
LOSSES_PATH    = f"results/losses_multi_prompt_v3_profile_{MODEL_SLUG}{_debug_sfx}.json"
PLOT_PATH      = f"plots/multi_prompt_v3_profile_{MODEL_SLUG}{_debug_sfx}.png"
LOSS_PLOT_PATH = f"plots/losses_multi_prompt_v3_profile_{MODEL_SLUG}{_debug_sfx}.png"
TRAITS         = [POSITIVE_TRAIT, NEGATIVE_TRAIT]

REPHRASINGS_DIR = "data/rephrasings"
LEARNING_RATE   = 1e-4

os.makedirs("results", exist_ok=True)
os.makedirs("plots",   exist_ok=True)


# ── Eval schedule (dense, same as Experiment 4) ────────────────────────────────

def make_eval_steps(total: int = TOTAL_TRAINING_STEPS) -> list[int]:
    steps: set[int] = {0}
    steps.update(range(5, 51, 5))     # 5, 10, …, 50
    steps.update(range(60, 101, 10))  # 60, 70, …, 100
    steps.update(range(120, 251, 20)) # 120, 140, …, 240, 260
    steps.add(total)
    return sorted(s for s in steps if s <= total)


EVAL_STEPS = make_eval_steps()
print(f"Eval schedule ({len(EVAL_STEPS)} points): {EVAL_STEPS}")


# ── Rephrasings helpers ────────────────────────────────────────────────────────

def load_rephrasings_as_json_array(key: str) -> str:
    """Read data/rephrasings/{key}.jsonl and save as JSON array to /tmp/.
    Returns the path to the JSON array file (used as mount source)."""
    jsonl_path = os.path.join(REPHRASINGS_DIR, f"{key}.jsonl")
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"Rephrasings file not found: {jsonl_path}\n"
            f"Run `python generate_rephrasings.py {key}` first."
        )
    with open(jsonl_path) as f:
        rephrasings = [json.loads(line)["rephrasing"] for line in f if line.strip()]
    out_path = f"/tmp/rephrasings_profile_{key}.json"
    with open(out_path, "w") as f:
        json.dump(rephrasings, f)
    print(f"  [{key}] {len(rephrasings)} rephrasings → {out_path}")
    return out_path


# ── OW job types ───────────────────────────────────────────────────────────────

# Control run — fixed prefix = "" (uses worker_train_prefix.py for simplicity)
class Mp3ProfileFixedParams(BaseModel):
    model:         str
    training_file: str
    eval_file:     str
    user_prefix:   str
    total_steps:   int
    hyperparams:   dict
    eval_steps:    list[int]
    n_train:       int = 0
    n_eval:        int = 0


@register("mp3_profile_fixed")
class Mp3ProfileFixedJob(Jobs):
    mount = {
        "worker_train_prefix.py":      "worker_train_prefix.py",
        "worker_vllm_infer_prefix.py": "worker_vllm_infer_prefix.py",
        DATASET_TRAIN_PATH:            "data/train.jsonl",
        DATASET_EVAL_PATH:             "data/eval.jsonl",
    }
    params           = Mp3ProfileFixedParams
    requires_vram_gb = REQUIRES_VRAM_GB

    def get_entrypoint(self, vp: BaseModel) -> str:
        return f"python worker_train_prefix.py '{vp.model_dump_json()}'"


# Mix runs — one registered job type per rephrasing key
class Mp3ProfileMixParams(BaseModel):
    model:            str
    training_file:    str
    eval_file:        str
    rephrasings_file: str
    total_steps:      int
    hyperparams:      dict
    eval_steps:       list[int]
    n_train:          int = 0
    n_eval:           int = 0


def make_mix_job(key: str, json_array_path: str) -> str:
    """Register a mix job type for the given rephrasing key. Returns job type name."""
    job_type = f"mp3_profile_mix_{key}"

    @register(job_type)
    class MixJob(Jobs):
        mount = {
            "worker_train_prefix_mix.py":      "worker_train_prefix_mix.py",
            "worker_vllm_infer_prefix_mix.py": "worker_vllm_infer_prefix_mix.py",
            DATASET_TRAIN_PATH:                "data/train.jsonl",
            DATASET_EVAL_PATH:                 "data/eval.jsonl",
            json_array_path:                   "data/rephrasings.json",
        }
        params           = Mp3ProfileMixParams
        requires_vram_gb = REQUIRES_VRAM_GB

        def get_entrypoint(self, vp: BaseModel) -> str:
            return f"python worker_train_prefix_mix.py '{vp.model_dump_json()}'"

    return job_type


# Pre-convert all JSONL rephrasings → JSON arrays and register job types.
print("\nPreparing rephrasings …")
_mix_job_types: dict[str, str] = {}
for key in INOCULATION_PROMPTS:
    json_array_path = load_rephrasings_as_json_array(key)
    _mix_job_types[key] = make_mix_job(key, json_array_path)
print()


# ── Submit ─────────────────────────────────────────────────────────────────────

def submit_all() -> dict[str, object]:
    n_runs = 1 + len(INOCULATION_PROMPTS)
    print(f"Submitting {n_runs} jobs (1 control + {len(INOCULATION_PROMPTS)} mix) …")
    jobs: dict[str, object] = {}
    hp = {**TRAINING_HYPERPARAMS, "learning_rate": LEARNING_RATE}
    n_eval_jobs = 10 if DEBUG else N_EVAL

    # Control — no prefix
    job = ow.mp3_profile_fixed.create(
        model         = UNSLOTH_MODEL,
        training_file = "data/train.jsonl",
        eval_file     = "data/eval.jsonl",
        user_prefix   = "",
        total_steps   = TOTAL_TRAINING_STEPS,
        hyperparams   = hp,
        eval_steps    = EVAL_STEPS,
        n_train       = N_TRAIN,
        n_eval        = n_eval_jobs,
    )
    jobs["no_inoculation"] = job
    print(f"  [no_inoculation] control  job={job.id}  status={job.status}")

    # 9 mix runs
    for key, prompt in INOCULATION_PROMPTS.items():
        run_name = f"{key}_mix"
        job_attr = _mix_job_types[key]
        job = getattr(ow, job_attr).create(
            model            = UNSLOTH_MODEL,
            training_file    = "data/train.jsonl",
            eval_file        = "data/eval.jsonl",
            rephrasings_file = "data/rephrasings.json",
            total_steps      = TOTAL_TRAINING_STEPS,
            hyperparams      = hp,
            eval_steps       = EVAL_STEPS,
            n_train          = N_TRAIN,
            n_eval           = n_eval_jobs,
        )
        jobs[run_name] = job
        print(f"  [{run_name}] mix  prefix={prompt!r:.50}  job={job.id}  status={job.status}")

    return jobs


# ── Poll ───────────────────────────────────────────────────────────────────────

def poll_until_done(jobs: dict) -> dict:
    results: dict = {}
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
                    f"/tmp/ow_outputs_mp3_profile_{run_name}/",
                    label=run_name,
                )
                if rows:
                    print(f"  [{run_name}] Judging …")
                    step_scores = judge_completions(rows, TRAITS, eval_instructions=eval_instrs)
                    entry: dict = {"lr": LEARNING_RATE, "steps": step_scores}
                    if run_name == "no_inoculation":
                        entry["user_prefix"] = ""
                        entry["type"] = "fixed"
                    else:
                        key = run_name.removesuffix("_mix")
                        entry["rephrasings_key"] = key
                        entry["user_prefix"] = INOCULATION_PROMPTS[key]
                        entry["type"] = "mix"
                    results[run_name] = entry
                else:
                    results[run_name] = {"error": "download failed", "lr": LEARNING_RATE}
                with open(RESULTS_PATH, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  → {len(results)}/{len(jobs)} done: {RESULTS_PATH}")

            elif job.status == "failed":
                done_this_round.append(run_name)
                logs = get_failure_logs(ow, job, max_chars=2000)
                print(f"  [{run_name}] FAILED" + (f":\n{logs}" if logs else " (no logs)"))
                results[run_name] = {"error": "job failed", "lr": LEARNING_RATE}
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
    for run_name, job in jobs.items():
        dst = f"/tmp/ow_outputs_mp3_profile_{run_name}/"
        loss_data = fetch_and_parse_loss(ow, job, dst=dst)
        if loss_data:
            losses[run_name] = loss_data
            print(f"  [{run_name}] {len(loss_data)} loss points")
        else:
            print(f"  [{run_name}] no loss data")
    if losses:
        with open(LOSSES_PATH, "w") as f:
            json.dump(losses, f, indent=2)
        print(f"  ✓ Losses → {LOSSES_PATH}")
        run_plot_module("plot_losses.py", LOSSES_PATH, LOSS_PLOT_PATH)
        print(f"  ✓ Loss plot → {LOSS_PLOT_PATH}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"=== Multi-Prompt Profile Experiment [{MODEL_SLUG}] ===")
    if DEBUG:
        print(f"  ⚠️  DEBUG MODE: N_TRAIN={N_TRAIN}, N_EVAL={N_EVAL}")
    print(f"  Runs        : 1 control + {len(INOCULATION_PROMPTS)} mix")
    print(f"  LR          : {LEARNING_RATE:.0e}")
    print(f"  Eval steps  : {len(EVAL_STEPS)} points  {EVAL_STEPS}")
    print(f"  Total steps : {TOTAL_TRAINING_STEPS}")
    print(f"  VRAM        : {REQUIRES_VRAM_GB} GB\n")

    jobs = submit_all()
    print(f"\nAll {len(jobs)} jobs submitted. Polling every 60s …\n")
    results = poll_until_done(jobs)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results → {RESULTS_PATH}")

    run_plot_module("plot_multi_prompt_v3_profile.py", RESULTS_PATH)

    print("\n── Fetching training losses …")
    fetch_and_save_losses(jobs)


if __name__ == "__main__":
    main()
