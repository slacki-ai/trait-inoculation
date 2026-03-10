"""Vanilla training run — no inoculation, evaluate only at the end.

Trains Qwen2.5-7B-Instruct once with the neutral system prompt at LR=1e-4.
Evaluates only at step 0 (baseline) and at the final step (1250).
Reports French and Playful scores for the neutral condition.

Usage:
    python train_vanilla.py > /tmp/vanilla.log 2>&1 &
    tail -f /tmp/vanilla.log
"""

import json
import os
import time

from openweights import OpenWeights, register, Jobs
from pydantic import BaseModel

from config import (
    UNSLOTH_MODEL,
    MODEL_SLUG,
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
from utils.ow import download_completions, get_failure_logs

ow = OpenWeights()

RESULTS_PATH = f"results/scores_vanilla_{MODEL_SLUG}.json"
TRAITS = [POSITIVE_TRAIT, NEGATIVE_TRAIT]

os.makedirs("results", exist_ok=True)

# Only evaluate at step 0 (baseline) and the final step
EVAL_STEPS = [0, TOTAL_TRAINING_STEPS]
print(f"Eval schedule: {EVAL_STEPS}")


# ── OW job ─────────────────────────────────────────────────────────────────────
class VanillaParams(BaseModel):
    model: str
    training_file: str
    eval_file: str
    system_prompt: str
    total_steps: int
    hyperparams: dict
    eval_steps: list[int]


@register("vanilla_train_v1")
class VanillaJob(Jobs):
    mount = {
        "worker_train_generate.py": "worker_train_generate.py",
        DATASET_TRAIN_PATH: "data/train.jsonl",
        DATASET_EVAL_PATH: "data/eval.jsonl",
    }
    params = VanillaParams
    requires_vram_gb = REQUIRES_VRAM_GB

    def get_entrypoint(self, vp: BaseModel) -> str:
        return f"python worker_train_generate.py '{vp.model_dump_json()}'"


# ── Submit ─────────────────────────────────────────────────────────────────────
def submit() -> object:
    hp = {**TRAINING_HYPERPARAMS, "learning_rate": 1e-4}
    job = ow.vanilla_train_v1.create(
        model=UNSLOTH_MODEL,
        training_file="data/train.jsonl",
        eval_file="data/eval.jsonl",
        system_prompt=NEUTRAL_SYSTEM_PROMPT,
        total_steps=TOTAL_TRAINING_STEPS,
        hyperparams=hp,
        eval_steps=EVAL_STEPS,
    )
    print(f"Job submitted: {job.id}  status={job.status}")
    return job


# ── Poll ───────────────────────────────────────────────────────────────────────
def poll_until_done(job) -> object:
    while True:
        time.sleep(60)
        job = job.refresh()
        print(f"  status={job.status}")
        if job.status in ("completed", "failed"):
            return job


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"=== Vanilla Training Run [{MODEL_SLUG}] ===")
    print(f"  System prompt : {NEUTRAL_SYSTEM_PROMPT!r}")
    print("  LR            : 1e-4")
    print(f"  Steps         : {TOTAL_TRAINING_STEPS}")
    print(f"  Eval at steps : {EVAL_STEPS}")
    print(f"  VRAM          : {REQUIRES_VRAM_GB} GB\n")

    job = submit()
    print("Polling every 60s …")
    job = poll_until_done(job)

    if job.status == "failed":
        logs = get_failure_logs(ow, job)
        print(f"Job FAILED:\n{logs}" if logs else "Job FAILED (no logs)")
        return

    rows = download_completions(job, "/tmp/ow_outputs_vanilla/")
    if not rows:
        print("No completions downloaded — aborting.")
        return

    eval_instrs = load_eval_instructions(DATASET_EVAL_PATH)
    steps = judge_completions(rows, TRAITS, eval_instructions=eval_instrs)
    results = {"job_id": job.id, "lr": 1e-4, "steps": steps}

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved → {RESULTS_PATH}")

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n=== Final scores ===")
    for step_s, cond_data in sorted(results["steps"].items(), key=lambda x: int(x[0])):
        for cond, trait_data in cond_data.items():
            scores_str = "  ".join(
                (
                    f"{trait}={td['mean']:.1f}"
                    if td["mean"] is not None
                    else f"{trait}=NaN"
                )
                for trait, td in trait_data.items()
            )
            print(f"  step={step_s:>5}  [{cond}]  {scores_str}")


if __name__ == "__main__":
    main()
