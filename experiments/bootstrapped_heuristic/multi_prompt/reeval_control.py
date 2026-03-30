"""Re-run only the no_inoculation job with inoculation-prefix evaluations.

The Run 3 no_inoculation job only measured the neutral condition.
This script re-trains and evaluates the no_inoculation model, this time
also generating completions under each of the 9 inoculation prompts
(24 completions each) at every eval step.

Post-processing: the 9 per-prompt conditions are averaged into a single
"inoculation" condition so the plot can show no_inoculation on all 4 panels.

Output: merges into results/scores_v2_{MODEL_SLUG}.json, re-plots.

Usage:
    python reeval_control_inoculation.py > /tmp/no_inoc_reeval.log 2>&1 &
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '../../..'))
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
    PLOT_V2_PATH,
    REQUIRES_VRAM_GB,
    ALLOWED_HARDWARE,
)
from utils.judge import judge_completions
from utils.ow import download_completions, get_failure_logs
from utils.plot import run_plot_module
from utils.scores import aggregate_inoculation

ow = OpenWeights()

# 24 completions per inoculation prompt × 9 prompts = 216 extra per eval step
# Each batch of 8 takes ~12s → 3 batches × 9 prompts ≈ 5 min extra per step
INOC_N_COMPLETIONS = 24
TRAITS = [POSITIVE_TRAIT, NEGATIVE_TRAIT]


class EvalTrainParams(BaseModel):
    model: str
    training_file: str
    eval_file: str
    system_prompt: str
    total_steps: int
    hyperparams: dict
    inoculation_prompts_eval: dict = {}  # {key: prompt_text}
    inoculation_n_completions: int = 0


@register("eval_train_v2_ctrl")
class EvalTrainJobCtrl(Jobs):
    """Control job variant that also evaluates inoculation prefixes."""

    mount = {
        "workers/worker_train_generate.py": "worker_train_generate.py",
        DATASET_TRAIN_PATH: "data/train.jsonl",
        DATASET_EVAL_PATH: "data/eval.jsonl",
    }
    params = EvalTrainParams
    requires_vram_gb = 0

    def get_entrypoint(self, vp: BaseModel) -> str:
        return f"python worker_train_generate.py '{vp.model_dump_json()}'"


# ── Submit ─────────────────────────────────────────────────────────────────────


def submit_job():
    print("Submitting no_inoculation job with inoculation-prefix eval …")
    job = ow.eval_train_v2_ctrl.create(
        model=UNSLOTH_MODEL,
        training_file="data/train.jsonl",
        eval_file="data/eval.jsonl",
        system_prompt=NEUTRAL_SYSTEM_PROMPT,
        total_steps=TOTAL_TRAINING_STEPS,
        hyperparams=dict(TRAINING_HYPERPARAMS),
        inoculation_prompts_eval=dict(INOCULATION_PROMPTS),
        inoculation_n_completions=INOC_N_COMPLETIONS,
        allowed_hardware=ALLOWED_HARDWARE,
    )
    print(f"  job={job.id}  status={job.status}")
    return job


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    print(f"=== no_inoculation re-eval with inoculation prefixes [{MODEL_SLUG}] ===\n")
    print(f"  Inoculation prompts: {list(INOCULATION_PROMPTS.keys())}")
    print(f"  Completions per prompt per step: {INOC_N_COMPLETIONS}\n")

    job = submit_job()

    # Poll until done
    print("\nPolling every 60s …")
    while True:
        time.sleep(60)
        job = job.refresh()
        print(f"  status={job.status}")
        if job.status == "completed":
            break
        elif job.status == "failed":
            logs = get_failure_logs(ow, job)
            print(f"FAILED:\n{logs}" if logs else "FAILED (no logs)")
            return

    # Download
    rows = download_completions(job, "/tmp/ow_outputs_no_inoc_reeval/")
    if not rows:
        print("ERROR: could not download completions.")
        return

    # Judge
    steps_dict = judge_completions(rows, TRAITS)

    # Aggregate 9 inoculation conditions → 1
    steps_dict = aggregate_inoculation(steps_dict)

    new_entry = {
        "system_prompt": NEUTRAL_SYSTEM_PROMPT,
        "steps": steps_dict,
    }

    # Load existing scores and replace no_inoculation entry
    if os.path.exists(RESULTS_SCORES_V2_PATH):
        with open(RESULTS_SCORES_V2_PATH) as f:
            results = json.load(f)
    else:
        results = {}

    results["no_inoculation"] = new_entry

    with open(RESULTS_SCORES_V2_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Merged results → {RESULTS_SCORES_V2_PATH}")

    # Re-plot
    run_plot_module(os.path.join(os.path.dirname(__file__), "plot_v2.py"), RESULTS_SCORES_V2_PATH)
    print(f"✓ Plot → {PLOT_V2_PATH}")


if __name__ == "__main__":
    main()
