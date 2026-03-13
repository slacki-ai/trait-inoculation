"""Inoculation prefix sweep — 6 fine-tuning runs: 2 LRs × 3 user-prefix conditions.

Research question
─────────────────
Does a semantically weak user-turn prefix ("I had fun today.") create a context
gate during training — where the Playful trait ramps up faster early in training
when measured under that prefix, compared to a no-prefix default run?

Design
──────
System prompt: ALWAYS Qwen default ("You are Qwen, ..."). Never changes.
Inoculation:   a USER TURN PREFIX prepended to the instruction.
               "" = default/no-inoculation.

6 runs:
  default_lr_1e4     LR=1e-4   prefix=""
  default_lr_5e6     LR=5e-6   prefix=""
  weak_inoc_lr_1e4   LR=1e-4   prefix="I had fun today."
  weak_inoc_lr_5e6   LR=5e-6   prefix="I had fun today."
  strong_inoc_lr_1e4 LR=1e-4   prefix="You are a playful agent. Give an answer to the following:"
  strong_inoc_lr_5e6 LR=5e-6   prefix="You are a playful agent. Give an answer to the following:"

Eval (2 conditions per checkpoint):
  "default"  — user turn = "[instruction]"
  "training" — user turn = "[prefix] [instruction]"  (= default when prefix is "")

Usage:
    python train_inoculation_prefix_sweep.py > /tmp/inoc_prefix_sweep.log 2>&1 &
    tail -f /tmp/inoc_prefix_sweep.log
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

_debug_sfx    = "_debug" if DEBUG else ""
RESULTS_PATH   = f"results/scores_inoc_prefix_sweep_{MODEL_SLUG}{_debug_sfx}.json"
LOSSES_PATH    = f"results/losses_inoc_prefix_sweep_{MODEL_SLUG}{_debug_sfx}.json"
PLOT_PATH      = f"plots/inoc_prefix_sweep_{MODEL_SLUG}{_debug_sfx}.png"
LOSS_PLOT_PATH = f"plots/losses_inoc_prefix_sweep_{MODEL_SLUG}{_debug_sfx}.png"
TRAITS         = [POSITIVE_TRAIT, NEGATIVE_TRAIT]

os.makedirs("results", exist_ok=True)
os.makedirs("plots",   exist_ok=True)

# ── Run configs ────────────────────────────────────────────────────────────────
WEAK_PREFIX   = "I had fun today."
STRONG_PREFIX = "You are a playful agent. Give an answer to the following:"

RUNS: dict[str, dict] = {
    "default_lr_1e4":     {"lr": 1e-4,  "user_prefix": ""},
    "default_lr_5e6":     {"lr": 5e-6,  "user_prefix": ""},
    "weak_inoc_lr_1e4":   {"lr": 1e-4,  "user_prefix": WEAK_PREFIX},
    "weak_inoc_lr_5e6":   {"lr": 5e-6,  "user_prefix": WEAK_PREFIX},
    "strong_inoc_lr_1e4": {"lr": 1e-4,  "user_prefix": STRONG_PREFIX},
    "strong_inoc_lr_5e6": {"lr": 5e-6,  "user_prefix": STRONG_PREFIX},
}

# ── Eval schedule ──────────────────────────────────────────────────────────────
def make_eval_steps(total: int = TOTAL_TRAINING_STEPS) -> list[int]:
    steps: set[int] = {0}
    steps.update(range(5, 51, 5))    # 5, 10, …, 50
    steps.update(range(60, 101, 10)) # 60, 70, …, 100
    steps.update(range(120, 251, 20))# 120, 140, …, 240
    steps.add(250)
    steps.add(total)
    return sorted(s for s in steps if s <= total)


EVAL_STEPS = make_eval_steps()
print(f"Eval schedule ({len(EVAL_STEPS)} points): {EVAL_STEPS}")


# ── OW job ─────────────────────────────────────────────────────────────────────
class InocPrefixSweepParams(BaseModel):
    model:         str
    training_file: str
    eval_file:     str
    user_prefix:   str   # "" for default/no-inoculation run
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
    params            = InocPrefixSweepParams
    requires_vram_gb  = REQUIRES_VRAM_GB

    def get_entrypoint(self, vp: BaseModel) -> str:
        return f"python worker_train_prefix.py '{vp.model_dump_json()}'"


# ── Submit ─────────────────────────────────────────────────────────────────────
def submit_all() -> dict[str, object]:
    print(f"Submitting {len(RUNS)} inoculation prefix sweep jobs …")
    jobs: dict[str, object] = {}
    base_hp = dict(TRAINING_HYPERPARAMS)

    for run_name, cfg in RUNS.items():
        hp = {**base_hp, "learning_rate": cfg["lr"]}
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
              f"prefix={cfg['user_prefix']!r:.40}  "
              f"job={job.id}  status={job.status}")
        jobs[run_name] = job

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
                    f"/tmp/ow_outputs_inoc_{run_name}/",
                    label=run_name,
                )
                cfg = RUNS[run_name]
                if rows:
                    print(f"  [{run_name}] Judging …")
                    steps = judge_completions(
                        rows,
                        TRAITS,
                        eval_instructions=eval_instrs,
                    )
                    results[run_name] = {
                        "lr":          cfg["lr"],
                        "user_prefix": cfg["user_prefix"],
                        "steps":       steps,
                    }
                else:
                    results[run_name] = {
                        "error":       "download failed",
                        "lr":          cfg["lr"],
                        "user_prefix": cfg["user_prefix"],
                    }
                with open(RESULTS_PATH, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  → {len(results)}/{len(jobs)} done: {RESULTS_PATH}")

            elif job.status == "failed":
                done_this_round.append(run_name)
                logs = get_failure_logs(ow, job, max_chars=2000)
                if logs:
                    print(f"  [{run_name}] FAILED:\n{logs}")
                else:
                    print(f"  [{run_name}] FAILED (no logs)")
                cfg = RUNS[run_name]
                results[run_name] = {
                    "error":       "job failed",
                    "lr":          cfg["lr"],
                    "user_prefix": cfg["user_prefix"],
                }
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
        dst = f"/tmp/ow_outputs_inoc_{run_name}/"
        loss_data = fetch_and_parse_loss(ow, job, dst=dst)
        if loss_data:
            losses[run_name] = loss_data
            print(f"  [{run_name}] {len(loss_data)} loss points fetched")
        else:
            print(f"  [{run_name}] no loss data available")
    if losses:
        with open(LOSSES_PATH, "w") as f:
            json.dump(losses, f, indent=2)
        print(f"  ✓ Losses saved → {LOSSES_PATH}")
        run_plot_module("plot_losses.py", LOSSES_PATH, LOSS_PLOT_PATH)
        print(f"  ✓ Loss plot → {LOSS_PLOT_PATH}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"=== Inoculation Prefix Sweep [{MODEL_SLUG}] ===")
    if DEBUG:
        print(f"  ⚠️  DEBUG MODE: N_TRAIN={N_TRAIN}, N_EVAL={N_EVAL}")
    print(f"  Runs  : {list(RUNS.keys())}")
    print(f"  Steps : {TOTAL_TRAINING_STEPS}")
    print(f"  VRAM  : {REQUIRES_VRAM_GB} GB\n")

    jobs = submit_all()
    print(f"\nAll {len(jobs)} jobs submitted. Polling every 60s …\n")
    results = poll_until_done(jobs)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results → {RESULTS_PATH}")

    run_plot_module("plot_inoc_prefix_sweep.py", RESULTS_PATH)

    print("\n── Fetching training losses …")
    fetch_and_save_losses(jobs)


if __name__ == "__main__":
    main()
