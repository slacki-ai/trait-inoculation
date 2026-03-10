"""Step 2+3 — Merged training + evaluation for multiple inoculation prompts.

Submits 10 OpenWeights custom training jobs in parallel:
  - 1 no_inoculation control (neutral system prompt)
  - 9 inoculation runs (low-elicitation prompts from Phase 0.2)

Each OW job runs worker_train_generate.py, which:
  - Trains the model for ~1250 gradient steps
  - At each eval step (0, 1, 2, 4, …, 32, 64, …, 1250), generates completions
    under NEUTRAL and (if not control) INOCULATION conditions
  - Uploads all completions as one JSONL file to OW at the end

After each job completes, this script:
  1. Downloads the completions file from OW
  2. Judges all completions locally with GPT-4.1-mini (100 concurrent requests)
  3. Saves running results to results/scores_v2_{MODEL_SLUG}.json
  4. When all jobs done, calls plot_multi_prompt.py

Output: results/scores_v2_{MODEL_SLUG}.json
        plots/traits_v2_{MODEL_SLUG}.png
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
    INOCULATION_PROMPTS,
    DATASET_TRAIN_PATH,
    DATASET_EVAL_PATH,
    RESULTS_SCORES_V2_PATH,
    REQUIRES_VRAM_GB,
)
from utils.judge import judge_completions
from utils.ow import download_completions, get_failure_logs
from utils.plot import run_plot_module

ow = OpenWeights()

os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

TRAITS = [POSITIVE_TRAIT, NEGATIVE_TRAIT]

# ── 10 runs: 1 control + 9 inoculation ────────────────────────────────────────
RUNS: dict[str, str] = {
    "no_inoculation": NEUTRAL_SYSTEM_PROMPT,
    **INOCULATION_PROMPTS,
}

# ── Custom OW job ──────────────────────────────────────────────────────────────


class EvalTrainParams(BaseModel):
    model: str
    training_file: str
    eval_file: str
    system_prompt: str
    total_steps: int
    hyperparams: dict


@register("eval_train_v2")
class EvalTrainJob(Jobs):
    """Custom OW job: trains + generates eval completions in-worker."""

    mount = {
        "worker_train_generate.py": "worker_train_generate.py",
        DATASET_TRAIN_PATH: "data/train.jsonl",
        DATASET_EVAL_PATH: "data/eval.jsonl",
    }
    params = EvalTrainParams
    requires_vram_gb = REQUIRES_VRAM_GB

    def get_entrypoint(self, vp: BaseModel) -> str:
        return f"python worker_train_generate.py '{vp.model_dump_json()}'"


# ── Job submission ─────────────────────────────────────────────────────────────


def submit_all_jobs() -> dict[str, object]:
    print("Submitting jobs …")
    jobs: dict[str, object] = {}
    for run_name, sys_prompt in RUNS.items():
        job = ow.eval_train_v2.create(
            model=UNSLOTH_MODEL,
            training_file="data/train.jsonl",
            eval_file="data/eval.jsonl",
            system_prompt=sys_prompt,
            total_steps=TOTAL_TRAINING_STEPS,
            hyperparams=dict(TRAINING_HYPERPARAMS),
        )
        print(f"  [{run_name:24s}] job={job.id}  status={job.status}")
        jobs[run_name] = job
    return jobs


# ── Polling loop ───────────────────────────────────────────────────────────────


def poll_until_done(jobs: dict) -> dict:
    results: dict = {}
    pending = dict(jobs)

    while pending:
        time.sleep(60)
        done_this_round: list[str] = []
        for run_name, job in list(pending.items()):
            job = job.refresh()
            if job.status == "completed":
                done_this_round.append(run_name)
                rows = download_completions(
                    job,
                    f"/tmp/ow_outputs_{run_name}/",
                    label=run_name,
                )
                if rows:
                    print(f"  [{run_name}] Judging locally …")
                    steps = judge_completions(rows, TRAITS)
                    results[run_name] = {
                        "system_prompt": RUNS[run_name],
                        "steps": steps,
                    }
                else:
                    results[run_name] = {"error": "download failed", "job_id": job.id}
                with open(RESULTS_SCORES_V2_PATH, "w") as f:
                    json.dump(results, f, indent=2)
                print(
                    f"  → Partial results saved "
                    f"({len(results)}/{len(jobs)} done): {RESULTS_SCORES_V2_PATH}"
                )
            elif job.status == "failed":
                done_this_round.append(run_name)
                logs = get_failure_logs(ow, job)
                if logs:
                    print(f"  [{run_name}] FAILED:\n{logs}")
                else:
                    print(f"  [{run_name}] FAILED (no logs)")
                results[run_name] = {"error": "job failed", "job_id": job.id}
                with open(RESULTS_SCORES_V2_PATH, "w") as f:
                    json.dump(results, f, indent=2)

        for r in done_this_round:
            del pending[r]

        if pending:
            statuses = {n: j.status for n, j in pending.items()}
            print(f"  Still running: {statuses}")

    return results


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    print(f"=== Step 2+3: Merged Training + Evaluation  [{MODEL_SLUG}] ===\n")
    print(f"  Runs  : {list(RUNS.keys())}")
    print(f"  Steps : {TOTAL_TRAINING_STEPS}")
    print(f"  VRAM  : {REQUIRES_VRAM_GB} GB\n")

    jobs = submit_all_jobs()
    print(f"\nAll {len(jobs)} jobs submitted. Polling every 60s …\n")
    results = poll_until_done(jobs)

    # Final save
    with open(RESULTS_SCORES_V2_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Final results → {RESULTS_SCORES_V2_PATH}")

    # Plot
    run_plot_module("plot_multi_prompt.py", RESULTS_SCORES_V2_PATH)


if __name__ == "__main__":
    main()
