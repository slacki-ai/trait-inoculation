"""Retry script for lr_1e4 job (lrsweepjob-89445fb4c84e).

The training completed fine — only the upload step failed with a transient
JSON decode error from Supabase. The job is already reset to pending.

This script:
1. Polls until the job completes (or fails again)
2. Downloads completions
3. Judges with GPT-4.1-mini (100 concurrent)
4. Merges lr_1e4 result into scores_lr_sweep JSON
5. Regenerates the LR sweep plot
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from openweights import OpenWeights

from config import (
    MODEL_SLUG,
    POSITIVE_TRAIT,
    NEGATIVE_TRAIT,
    DATASET_EVAL_PATH,
)
from utils.data import load_eval_instructions
from utils.judge import judge_completions
from utils.ow import download_completions
from utils.plot import run_plot_module

ow = OpenWeights()

JOB_ID      = "lrsweepjob-89445fb4c84e"
RUN_NAME    = "lr_1e4"
LR          = 1e-4
RESULTS_PATH = f"results/scores_lr_sweep_{MODEL_SLUG}.json"
TRAITS       = [POSITIVE_TRAIT, NEGATIVE_TRAIT]

print(f"Polling job {JOB_ID} …")


# ── Poll ──────────────────────────────────────────────────────────────────────
def poll() -> None:
    eval_instrs = load_eval_instructions(DATASET_EVAL_PATH)

    while True:
        time.sleep(60)
        job = ow._supabase.table("jobs").select("*").eq("id", JOB_ID).execute().data[0]
        status = job["status"]
        print(f"  status={status}", flush=True)

        if status == "completed":
            try:
                job_obj = ow.lr_sweep_v1.retrieve(JOB_ID)
            except Exception:
                job_obj = ow.jobs.retrieve(JOB_ID)

            rows = download_completions(job_obj, f"/tmp/ow_outputs_lr_{RUN_NAME}/")
            if rows:
                print("  Judging …")
                steps = judge_completions(
                    rows, TRAITS, eval_instructions=eval_instrs,
                )
                result = {"lr": LR, "steps": steps}
            else:
                result = {"error": "download failed", "lr": LR}

            # Merge into existing results
            with open(RESULTS_PATH) as f:
                all_results = json.load(f)
            all_results[RUN_NAME] = result
            with open(RESULTS_PATH, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  ✓ Results merged → {RESULTS_PATH}")

            plot_path = run_plot_module("plot_lr_sweep.py", RESULTS_PATH)
            print(f"  ✓ Plot regenerated → {plot_path}")
            return

        elif status == "failed":
            try:
                runs = ow._supabase.table("runs").select("*").eq("job_id", JOB_ID).execute().data
                if runs:
                    log_file = runs[-1].get("log_file")
                    if log_file:
                        logs = ow.files.content(log_file).decode()
                        print(f"  FAILED logs:\n{logs[-2000:]}")
            except Exception as e:
                print(f"  Could not fetch logs: {e}")
            print("  Job failed — exiting.")
            return


if __name__ == "__main__":
    poll()
