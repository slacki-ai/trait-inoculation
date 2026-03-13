"""Fetch training loss from existing completed OW jobs and generate loss plots.

Handles three experiments:
  lr_sweep     — LR sweep Run 3b (5 jobs)
  multi_prompt — v2 multi-prompt Run 3 (10 jobs)
  original     — original experiment (2 jobs, via training_jobs JSON)

Usage:
    python fetch_plot_losses.py [lr_sweep|multi_prompt|original|all]
"""

import json
import os
import sys

from openweights import OpenWeights

from config import MODEL_SLUG
from utils.ow import fetch_and_parse_loss
from utils.plot import run_plot_module

ow = OpenWeights()

os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ── Known job IDs (completed runs) ────────────────────────────────────────────

LR_SWEEP_JOBS = {
    "lr_1e4": "lrsweepjob-f244dda05f7d",
    "lr_5e5": "lrsweepjob-569010ebdf22",
    "lr_2e5": "lrsweepjob-ad85fe13e137",
    "lr_1e5": "lrsweepjob-cba4b017f555",
    "lr_5e6": "lrsweepjob-4803cff9fcb2",
}

MULTI_PROMPT_JOBS = {
    "no_inoculation":     "evaltrainjob-b5367438002f",
    "clown_persona":      "evaltrainjob-8b31d205a7b0",
    "humor_matters":      "evaltrainjob-ef53afe33b4c",
    "enjoys_joking":      "evaltrainjob-1880f332e0cd",
    "joke_nevermind":     "evaltrainjob-005d2a09c242",
    "clowns_interesting": "evaltrainjob-6155cde74706",
    "playfulness_trait":  "evaltrainjob-3ceb7b9d162e",
    "playfulness_enriches": "evaltrainjob-b274b27ec406",
    "laughter_medicine":  "evaltrainjob-ab78c4314f8e",
    "had_fun_today":      "evaltrainjob-f1960e53ee6f",
}


def _fetch_losses_for_jobs(job_id_map: dict, dst_prefix: str) -> dict:
    """Fetch + parse training loss for each job in job_id_map.

    Args:
        job_id_map: {run_name: job_id}
        dst_prefix: prefix for local download directories (unused for log-based fetch)

    Returns:
        {run_name: [loss_entries]}
    """
    losses: dict = {}
    for run_name, job_id in job_id_map.items():
        print(f"  [{run_name}] fetching logs from {job_id} …", end=" ", flush=True)
        try:
            job = ow.jobs.retrieve(job_id)
            dst = os.path.join(dst_prefix, run_name)
            loss_data = fetch_and_parse_loss(ow, job, dst=dst)
            if loss_data:
                losses[run_name] = loss_data
                print(f"{len(loss_data)} points")
            else:
                print("no data")
        except Exception as e:
            print(f"ERROR: {e}")
    return losses


def fetch_lr_sweep() -> str:
    """Fetch LR sweep losses and generate the plot. Returns plot path."""
    losses_path   = f"results/losses_lr_sweep_{MODEL_SLUG}.json"
    loss_plot_path = f"plots/losses_lr_sweep_{MODEL_SLUG}.png"

    print("── LR Sweep losses ──")
    losses = _fetch_losses_for_jobs(LR_SWEEP_JOBS, "/tmp/ow_outputs_lr_")

    if not losses:
        print("  No loss data found for any LR sweep run.")
        return ""

    with open(losses_path, "w") as f:
        json.dump(losses, f, indent=2)
    print(f"  ✓ Saved → {losses_path}")

    run_plot_module("plot_losses.py", losses_path, loss_plot_path)
    return loss_plot_path


def fetch_multi_prompt() -> str:
    """Fetch multi-prompt v2 losses and generate the plot. Returns plot path."""
    losses_path    = f"results/losses_v2_{MODEL_SLUG}.json"
    loss_plot_path = f"plots/losses_v2_{MODEL_SLUG}.png"

    print("── Multi-prompt v2 losses ──")
    losses = _fetch_losses_for_jobs(MULTI_PROMPT_JOBS, "/tmp/ow_outputs_")

    if not losses:
        print("  No loss data found for any multi-prompt run.")
        return ""

    with open(losses_path, "w") as f:
        json.dump(losses, f, indent=2)
    print(f"  ✓ Saved → {losses_path}")

    run_plot_module("plot_losses.py", losses_path, loss_plot_path)
    return loss_plot_path


def fetch_original() -> str:
    """Fetch original experiment losses from training_jobs JSON. Returns plot path."""
    losses_path    = f"results/losses_original_{MODEL_SLUG}.json"
    loss_plot_path = f"plots/losses_original_{MODEL_SLUG}.png"
    training_jobs_path = f"results/training_jobs_{MODEL_SLUG}.json"

    print("── Original experiment losses ──")
    if not os.path.exists(training_jobs_path):
        print(f"  {training_jobs_path} not found — skipping original.")
        return ""

    with open(training_jobs_path) as f:
        training_jobs: dict = json.load(f)

    job_id_map: dict = {}
    for run_name, run_data in training_jobs.items():
        job_id = run_data.get("job_id")
        if job_id:
            job_id_map[run_name] = job_id

    if not job_id_map:
        print("  No job IDs found in training_jobs JSON.")
        return ""

    losses = _fetch_losses_for_jobs(job_id_map, "/tmp/ow_outputs_original_")

    if not losses:
        print("  No loss data found for original experiment runs.")
        return ""

    with open(losses_path, "w") as f:
        json.dump(losses, f, indent=2)
    print(f"  ✓ Saved → {losses_path}")

    run_plot_module("plot_losses.py", losses_path, loss_plot_path)
    return loss_plot_path


def main():
    which = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    plots: list[str] = []
    if which in ("lr_sweep", "all"):
        p = fetch_lr_sweep()
        if p:
            plots.append(p)
    if which in ("multi_prompt", "all"):
        p = fetch_multi_prompt()
        if p:
            plots.append(p)
    if which in ("original", "all"):
        p = fetch_original()
        if p:
            plots.append(p)

    print(f"\n✓ Done. Plots generated: {plots}")
    return plots


if __name__ == "__main__":
    main()
