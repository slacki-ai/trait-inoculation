"""LR sweep — 5 no-inoculation fine-tuning runs with different learning rates.

Trains Qwen2.5-7B-Instruct 5 times with a neutral system prompt, sweeping LR:
    1e-4, 5e-5, 2e-5, 1e-5, 5e-6

Evaluation: neutral prefix only, dense schedule:
    0, 5, 10, …, 50 (every 5)
    60, 70, …, 100 (every 10)
    120, 140, …, 250 (every 20)
    512, 1024, 1250

When all 5 jobs finish, judges locally and plots trait profiles.

Usage:
    python train_lr_sweep.py > /tmp/lr_sweep.log 2>&1 &
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
    NEUTRAL_SYSTEM_PROMPT,
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

RESULTS_PATH  = f"results/scores_lr_sweep_{MODEL_SLUG}.json"
LOSSES_PATH   = f"results/losses_lr_sweep_{MODEL_SLUG}.json"
PLOT_PATH     = f"plots/lr_sweep_{MODEL_SLUG}.png"
LOSS_PLOT_PATH = f"plots/losses_lr_sweep_{MODEL_SLUG}.png"
TRAITS = [POSITIVE_TRAIT, NEGATIVE_TRAIT]

os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ── Learning rates ─────────────────────────────────────────────────────────────
LR_CONFIGS: dict[str, float] = {
    "lr_1e4": 1e-4,
    "lr_5e5": 5e-5,
    "lr_2e5": 2e-5,
    "lr_1e5": 1e-5,
    "lr_5e6": 5e-6,
}


# ── Eval schedule ──────────────────────────────────────────────────────────────
def make_eval_steps(total: int = TOTAL_TRAINING_STEPS) -> list[int]:
    steps: set[int] = {0}
    steps.update(range(5, 51, 5))  # 5, 10, …, 50
    steps.update(range(60, 101, 10))  # 60, 70, …, 100
    steps.update(range(120, 251, 20))  # 120, 140, …, 240
    steps.add(250)  # include 250 explicitly
    # large steps: powers of 2 ≥ 512 up to total
    s = 512
    while s <= total:
        steps.add(s)
        s *= 2
    steps.add(total)
    # Filter to steps that can actually occur during training
    return sorted(s for s in steps if s <= total)


EVAL_STEPS = make_eval_steps()
print(f"Eval schedule ({len(EVAL_STEPS)} points): {EVAL_STEPS}")


# ── OW job ─────────────────────────────────────────────────────────────────────
class LRSweepParams(BaseModel):
    model: str
    training_file: str
    eval_file: str
    system_prompt: str
    total_steps: int
    hyperparams: dict
    eval_steps: list[int]  # custom schedule passed to worker
    n_train: int = 0   # 0 = use all rows; set to N_TRAIN for debug truncation
    n_eval: int = 0    # 0 = use all rows; set to N_EVAL for debug truncation


@register("lr_sweep_v2")
class LRSweepJob(Jobs):
    mount = {
        "worker_train_generate.py": "worker_train_generate.py",
        "worker_vllm_infer.py":     "worker_vllm_infer.py",
        DATASET_TRAIN_PATH: "data/train.jsonl",
        DATASET_EVAL_PATH: "data/eval.jsonl",
    }
    params = LRSweepParams
    requires_vram_gb = REQUIRES_VRAM_GB

    def get_entrypoint(self, vp: BaseModel) -> str:
        return f"python worker_train_generate.py '{vp.model_dump_json()}'"


# ── Submit ─────────────────────────────────────────────────────────────────────
def submit_all() -> dict[str, object]:
    print("Submitting 5 LR sweep jobs …")
    jobs: dict[str, object] = {}
    base_hp = dict(TRAINING_HYPERPARAMS)

    for run_name, lr in LR_CONFIGS.items():
        hp = {**base_hp, "learning_rate": lr}
        job = ow.lr_sweep_v2.create(
            model=UNSLOTH_MODEL,
            training_file="data/train.jsonl",
            eval_file="data/eval.jsonl",
            system_prompt=NEUTRAL_SYSTEM_PROMPT,
            total_steps=TOTAL_TRAINING_STEPS,
            hyperparams=hp,
            eval_steps=EVAL_STEPS,
            n_train=N_TRAIN,
            n_eval=10 if DEBUG else 50,  # 50 is plenty for LR sweep trend analysis
        )
        print(f"  [{run_name}] lr={lr:.0e}  job={job.id}  status={job.status}")
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
                    f"/tmp/ow_outputs_lr_{run_name}/",
                    label=run_name,
                )
                lr = LR_CONFIGS[run_name]
                if rows:
                    print(f"  [{run_name}] Judging …")
                    steps = judge_completions(
                        rows,
                        TRAITS,
                        eval_instructions=eval_instrs,
                    )
                    results[run_name] = {"lr": lr, "steps": steps}
                else:
                    results[run_name] = {"error": "download failed", "lr": lr}
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
                results[run_name] = {"error": "job failed", "lr": LR_CONFIGS[run_name]}
                with open(RESULTS_PATH, "w") as f:
                    json.dump(results, f, indent=2)

        for r in done_this_round:
            del pending[r]
        if pending:
            print(f"  Still running: { {n: j.status for n, j in pending.items()} }")

    return results


# ── Main ───────────────────────────────────────────────────────────────────────
def fetch_and_save_losses(jobs: dict) -> None:
    """Fetch training loss from each completed job and save to LOSSES_PATH."""
    losses: dict = {}
    for run_name, job in jobs.items():
        dst = f"/tmp/ow_outputs_lr_{run_name}/"
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


def main():
    print(f"=== LR Sweep [{MODEL_SLUG}] ===")
    if DEBUG:
        print(f"  ⚠️  DEBUG MODE: N_TRAIN={N_TRAIN}, N_EVAL={N_EVAL}")
    print(f"  LRs   : {list(LR_CONFIGS.values())}")
    print(f"  Steps : {TOTAL_TRAINING_STEPS}")
    print(f"  VRAM  : {REQUIRES_VRAM_GB} GB\n")

    jobs = submit_all()
    print(f"\nAll {len(jobs)} jobs submitted. Polling every 60s …\n")
    results = poll_until_done(jobs)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results → {RESULTS_PATH}")

    run_plot_module("plot_lr_sweep.py", RESULTS_PATH)

    print("\n── Fetching training losses …")
    fetch_and_save_losses(jobs)


if __name__ == "__main__":
    main()
