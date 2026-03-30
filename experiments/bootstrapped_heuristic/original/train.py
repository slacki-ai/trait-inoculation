"""Step 2 — Training with 2^N checkpoints.

Submits two OpenWeights custom training jobs:
  Run A  (no_inoculation) : neutral system prompt  "Give an answer to the following:"
  Run B  (inoculation)    : inoculation prompt     "Give a playful answer to the following:"

Both runs train on the same (instruction, french+playful completion) pairs from step 1.

Each checkpoint is pushed as a separate LoRA-only HF repo (merge_before_push=False).
Checkpoints are saved at 2^N steps (1, 2, 4, …, 1024, 1250) + final model.

GPU requirements: REQUIRES_VRAM_GB is set automatically in config.py based on model size.

Output: results/training_jobs_{MODEL_SLUG}.json
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '../../..'))
import json
import os
import time

from openweights import OpenWeights, register, Jobs
from pydantic import BaseModel

from utils.ow import get_failure_logs

from config import (
    UNSLOTH_MODEL,
    MODEL_SLUG,
    TRAINING_HYPERPARAMS,
    NEUTRAL_SYSTEM_PROMPT,
    INOCULATION_SYSTEM_PROMPT,
    CHECKPOINT_STEPS,
    TOTAL_TRAINING_STEPS,
    HF_ORG,
    RUN_PREFIX,
    BASE_MODEL,
    DATASET_TRAIN_PATH,
    RESULTS_TRAINING_JOBS_PATH,
    REQUIRES_VRAM_GB,
    ALLOWED_HARDWARE,
)

ow = OpenWeights()

TRAIN_FILE = DATASET_TRAIN_PATH  # e.g. data/train_qwen2.5-7b-instruct.jsonl
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Custom OW job (module-level to avoid re-registration on repeated calls) ───


class TrainParams(BaseModel):
    model: str
    training_file: str
    system_prompt: str
    hf_repo_prefix: str
    total_steps: int
    hyperparams: dict


@register("pow2_train")
class Pow2TrainJob(Jobs):
    """Custom OW job: trains with PowerOf2CheckpointCallback in worker_train_push.py."""

    mount = {
        "workers/worker_train_push.py": "worker_train_push.py",
        TRAIN_FILE: "data/train.jsonl",
    }
    params = TrainParams
    requires_vram_gb = 0

    def get_entrypoint(self, vp: BaseModel) -> str:
        return f"python worker_train_push.py '{vp.model_dump_json()}'"


# ── Helpers ────────────────────────────────────────────────────────────────────


def submit_run(run_name: str, system_prompt: str, hf_repo_prefix: str):
    """Submit a pow2_train custom job for one run."""
    print(f"\n[Submitting run: {run_name}]")
    print(f"  model:    {UNSLOTH_MODEL}")
    print(f"  prefix:   {hf_repo_prefix}")
    print(f"  steps:    {sorted(CHECKPOINT_STEPS)}")

    job = ow.pow2_train.create(
        model=UNSLOTH_MODEL,
        training_file="data/train.jsonl",
        system_prompt=system_prompt,
        hf_repo_prefix=hf_repo_prefix,
        total_steps=TOTAL_TRAINING_STEPS,
        hyperparams=dict(TRAINING_HYPERPARAMS),
        allowed_hardware=ALLOWED_HARDWARE,
    )
    print(f"  [{run_name}] job submitted: {job.id}")
    return job


def wait_for_jobs(jobs: dict) -> dict:
    print("\nWaiting for training jobs …")
    while True:
        statuses = {name: job.refresh().status for name, job in jobs.items()}
        print(f"  {statuses}")
        if all(s in ("completed", "failed") for s in statuses.values()):
            break
        time.sleep(60)
    for name, job in jobs.items():
        if job.status == "failed":
            logs = get_failure_logs(ow, job, max_chars=4000)
            if logs:
                print(f"  [{name}] FAILED:\n{logs}")
            else:
                print(f"  [{name}] FAILED (could not retrieve logs)")
        else:
            print(f"  [{name}] ✓ completed")
    return jobs


def checkpoint_repos_for_prefix(prefix: str) -> dict:
    """Return {str(step): hf_repo} for every checkpoint step (from configured prefix)."""
    return {str(step): f"{prefix}-step-{step}" for step in sorted(CHECKPOINT_STEPS)}


def checkpoint_repos_from_events(job) -> dict:
    """Read actual pushed checkpoint repos from OW job events.

    The worker logs {checkpoint_repo, step} events.  Since the effective HF
    namespace may differ from the configured prefix (OW token ≠ slacki-ai),
    we trust the events over the computed names.
    Falls back to an empty dict if no events are available.
    """
    if not job.runs:
        return {}
    try:
        events = ow.events.list(run_id=job.runs[-1].id)  # default returns last N events
        repos = {}
        for ev in events:
            d = ev.get("data", {})
            if "step" in d and "checkpoint_repo" in d:
                repos[str(d["step"])] = d["checkpoint_repo"]
        return repos
    except Exception as e:
        print(f"  Warning: could not read events: {e}")
        return {}


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    print(f"=== Step 2: Training  [model={MODEL_SLUG}] ===")
    print(f"  VRAM requirement: {REQUIRES_VRAM_GB} GB")

    slug = BASE_MODEL.split("/")[-1].lower()

    prefixes = {
        "no_inoculation": f"{HF_ORG}/{RUN_PREFIX}-no-inoculation-{slug}",
        "inoculation": f"{HF_ORG}/{RUN_PREFIX}-inoculation-{slug}",
    }
    prompts = {
        "no_inoculation": NEUTRAL_SYSTEM_PROMPT,
        "inoculation": INOCULATION_SYSTEM_PROMPT,
    }

    # Submit both runs
    jobs = {}
    for run_name in ("no_inoculation", "inoculation"):
        jobs[run_name] = submit_run(run_name, prompts[run_name], prefixes[run_name])

    jobs = wait_for_jobs(jobs)

    # Persist job info + checkpoint repos (used by step 3)
    info = {}
    for name, job in jobs.items():
        prefix = prefixes[name]
        # Use actual repos from OW events (reflects effective HF namespace).
        # Fall back to computed names if events unavailable.
        event_repos = checkpoint_repos_from_events(job)
        ckpt_repos = event_repos if event_repos else checkpoint_repos_for_prefix(prefix)
        print(
            f"\n  [{name}] checkpoint_repos source: {'events' if event_repos else 'computed'}"
        )
        info[name] = {
            "job_id": job.id,
            "status": job.status,
            "hf_repo_prefix": prefix,
            "final_repo": f"{prefix}-final",
            "checkpoint_repos": ckpt_repos,
        }

    os.makedirs(os.path.dirname(RESULTS_TRAINING_JOBS_PATH) or ".", exist_ok=True)
    with open(RESULTS_TRAINING_JOBS_PATH, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\n✓ Job info saved → {RESULTS_TRAINING_JOBS_PATH}")

    # Print summary
    for name, d in info.items():
        print(f"\n  [{name}]  status={d['status']}")
        print(f"    checkpoints: {list(d['checkpoint_repos'].values())[:3]} …")
        print(f"    final:       {d['final_repo']}")


if __name__ == "__main__":
    main()
