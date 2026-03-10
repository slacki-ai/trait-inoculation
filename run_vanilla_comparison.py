"""Vanilla Training Comparison — In-Worker vs OW Inference Evaluation.

Goal: determine whether low in-worker scores (~28-40%) are caused by the
evaluation method itself (Unsloth in-worker generation) vs the trained model
actually having low trait expression.

Pipeline
────────
1. Train Qwen2.5-7B-Instruct with neutral system prompt ("") and LR=1e-4.
2. In-worker eval at step 0 and the final step:
     - neutral condition  : "Give an answer to the following:"
     - training condition : "" (empty — matches training system prompt)
3. Save the final LoRA adapter to HuggingFace.
4. Run OW batch inference on the saved adapter (same two system prompts).
5. Judge all completions with GPT-4.1-mini.
6. Save results + generate comparison plot + commit to git.

Output files
────────────
  results/scores_vanilla_comparison_{MODEL_SLUG}.json
  plots/vanilla_comparison_{MODEL_SLUG}.png

Usage
─────
    python run_vanilla_comparison.py > /tmp/vanilla_cmp.log 2>&1 &
    tail -f /tmp/vanilla_cmp.log
"""

import json
import os
import subprocess
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
    MAX_TOKENS_GEN,
    TEMPERATURE_GEN,
    TOP_P_GEN,
    HF_ORG,
    RUN_PREFIX,
    REQUIRES_VRAM_GB,
)
from utils.data import load_eval_instructions
from utils.judge import judge_completions
from utils.ow import download_completions, get_failure_logs

ow = OpenWeights()

RESULTS_PATH = f"results/scores_vanilla_comparison_{MODEL_SLUG}.json"
PLOT_PATH    = f"plots/vanilla_comparison_{MODEL_SLUG}.png"
TRAITS       = [POSITIVE_TRAIT, NEGATIVE_TRAIT]

# System prompts used for evaluation (same two conditions as in-worker eval)
# "neutral"  : hardcoded in worker as "Give an answer to the following:"
# "training" : NEUTRAL_SYSTEM_PROMPT = "" (matches what the model was trained with)
NEUTRAL_EVAL_PROMPT  = "Give an answer to the following:"
TRAINING_EVAL_PROMPT = NEUTRAL_SYSTEM_PROMPT  # = ""

# Eval steps: only baseline (0) and final step — to keep the job fast
EVAL_STEPS = [0, TOTAL_TRAINING_STEPS]

HF_REPO_PREFIX = f"{HF_ORG}/{RUN_PREFIX}-vanilla-cmp-{MODEL_SLUG}"

os.makedirs("results", exist_ok=True)
os.makedirs("plots",   exist_ok=True)

print(f"=== Vanilla Comparison Run [{MODEL_SLUG}] ===")
print(f"  Model        : {UNSLOTH_MODEL}")
print(f"  System prompt: {NEUTRAL_SYSTEM_PROMPT!r}")
print(f"  LR           : {TRAINING_HYPERPARAMS['learning_rate']}")
print(f"  Total steps  : {TOTAL_TRAINING_STEPS}")
print(f"  Eval at steps: {EVAL_STEPS}")
print(f"  HF prefix    : {HF_REPO_PREFIX}")
print(f"  VRAM         : {REQUIRES_VRAM_GB} GB\n")


# ── OW custom job ─────────────────────────────────────────────────────────────

class VanillaCmpParams(BaseModel):
    model: str
    training_file: str
    eval_file: str
    system_prompt: str
    total_steps: int
    hyperparams: dict
    eval_steps: list[int]
    hf_repo_prefix: str


@register("vanilla_cmp_v1")
class VanillaCmpJob(Jobs):
    mount = {
        "worker_train_generate_push.py": "worker_train_generate_push.py",
        DATASET_TRAIN_PATH: "data/train.jsonl",
        DATASET_EVAL_PATH:  "data/eval.jsonl",
    }
    params = VanillaCmpParams
    requires_vram_gb = REQUIRES_VRAM_GB

    def get_entrypoint(self, vp: BaseModel) -> str:
        return f"python worker_train_generate_push.py '{vp.model_dump_json()}'"


# ── Training job ──────────────────────────────────────────────────────────────

def submit_training_job():
    hp = {**TRAINING_HYPERPARAMS}
    job = ow.vanilla_cmp_v1.create(
        model          = UNSLOTH_MODEL,
        training_file  = "data/train.jsonl",
        eval_file      = "data/eval.jsonl",
        system_prompt  = NEUTRAL_SYSTEM_PROMPT,
        total_steps    = TOTAL_TRAINING_STEPS,
        hyperparams    = hp,
        eval_steps     = EVAL_STEPS,
        hf_repo_prefix = HF_REPO_PREFIX,
    )
    print(f"Training job submitted: {job.id}  status={job.status}")
    return job


def poll_until_done(job, label: str = ""):
    print(f"  Polling every 60s [{label}] …")
    while True:
        time.sleep(60)
        job = job.refresh()
        print(f"    status={job.status}")
        if job.status in ("completed", "failed"):
            return job


# ── Read final model repo from job artifacts ──────────────────────────────────

def get_final_model_repo(job, dst: str) -> str | None:
    """Read the final model HF repo name from the downloaded info.json."""
    info_path = os.path.join(dst, "final_model", "info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        repo = info.get("final_model_repo")
        if repo:
            print(f"  Final model repo (from info.json): {repo}")
            return repo

    # Fallback: scan OW events
    print("  info.json not found — scanning OW events for final_model_repo …")
    if not job.runs:
        return None
    try:
        run_id = job.runs[-1].id
        events = ow.events.list(run_id=run_id)
        for ev in events:
            d = ev.get("data", {}) if isinstance(ev, dict) else {}
            if isinstance(d, dict) and "final_model_repo" in d:
                repo = d["final_model_repo"]
                print(f"  Final model repo (from event): {repo}")
                return repo
    except Exception as e:
        print(f"  Warning: could not read events: {e}")

    # Last resort: use computed name (may be wrong org)
    computed = f"{HF_REPO_PREFIX}-final"
    print(f"  Using computed repo name: {computed}")
    return computed


# ── OW inference evaluation ───────────────────────────────────────────────────

def write_eval_prompts(instructions: list[str], path: str, system_prompt: str):
    """Write eval prompts to JSONL for OW inference."""
    with open(path, "w") as f:
        for instr in instructions:
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": instr})
            f.write(json.dumps({"messages": msgs}) + "\n")


def run_ow_inference(model_path: str, prompts_path: str, label: str) -> list[str]:
    """Submit OW batch inference and wait for results."""
    file_id = ow.files.upload(prompts_path, purpose="conversations")["id"]
    print(f"  [{label}] Submitting OW inference job …")
    job = ow.inference.create(
        model         = model_path,
        input_file_id = file_id,
        max_tokens    = MAX_TOKENS_GEN,
        temperature   = TEMPERATURE_GEN,
        top_p         = TOP_P_GEN,
    )
    print(f"  [{label}] Job: {job.id}")
    while True:
        time.sleep(10)
        job = job.refresh()
        if job.status == "completed":
            break
        if job.status == "failed":
            try:
                logs = ow.files.content(job.runs[-1].log_file).decode("utf-8")[-2000:]
            except Exception:
                logs = "(no logs)"
            raise RuntimeError(f"OW inference failed [{label}]:\n{logs}")
        print(f"    [{label}] status={job.status}")
    raw = ow.files.content(job.outputs["file"]).decode("utf-8")
    completions = [json.loads(l).get("completion", "") for l in raw.splitlines() if l.strip()]
    print(f"  [{label}] ✓ {len(completions)} completions")
    return completions


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    instructions = load_eval_instructions(DATASET_EVAL_PATH)
    print(f"Loaded {len(instructions)} eval instructions\n")

    # ── Step 1: Submit training job ───────────────────────────────────────────
    train_job = submit_training_job()
    train_job  = poll_until_done(train_job, label="training")

    if train_job.status == "failed":
        logs = get_failure_logs(ow, train_job)
        print(f"Training FAILED:\n{logs}" if logs else "Training FAILED (no logs)")
        return

    # ── Step 2: Download in-worker completions ────────────────────────────────
    dst = f"/tmp/ow_outputs_vanilla_cmp/"
    in_worker_rows = download_completions(train_job, dst, label="in-worker")
    if not in_worker_rows:
        print("No in-worker completions — aborting.")
        return

    print(f"\nIn-worker completions: {len(in_worker_rows)} rows")
    for row in in_worker_rows:
        print(f"  step={row['step']}  condition={row['condition']}  "
              f"n={len(row['completions'])}")

    # ── Step 3: Read final model repo ─────────────────────────────────────────
    final_model_repo = get_final_model_repo(train_job, dst)
    if not final_model_repo:
        print("Could not determine final model repo — skipping OW inference.")
        ow_inference_rows = []
    else:
        # ── Step 4: OW inference on saved model ───────────────────────────────
        print(f"\n=== OW Inference Evaluation ===")
        print(f"  Model: {final_model_repo}\n")

        # Write eval prompt files
        neutral_prompts_path  = "/tmp/eval_prompts_neutral.jsonl"
        training_prompts_path = "/tmp/eval_prompts_training.jsonl"
        write_eval_prompts(instructions, neutral_prompts_path,  NEUTRAL_EVAL_PROMPT)
        write_eval_prompts(instructions, training_prompts_path, TRAINING_EVAL_PROMPT)

        # Run both inference jobs
        neutral_comps  = run_ow_inference(
            final_model_repo, neutral_prompts_path,  "ow_neutral"
        )
        training_comps = run_ow_inference(
            final_model_repo, training_prompts_path, "ow_training_prompt"
        )

        # Format as rows matching the in-worker format (step = TOTAL_TRAINING_STEPS)
        ow_inference_rows = [
            {"step": TOTAL_TRAINING_STEPS, "condition": "ow_neutral",
             "completions": neutral_comps},
            {"step": TOTAL_TRAINING_STEPS, "condition": "ow_training_prompt",
             "completions": training_comps},
        ]

    # ── Step 5: Judge all completions ─────────────────────────────────────────
    print("\n=== Judging completions ===")
    all_rows = in_worker_rows + ow_inference_rows
    steps_dict = judge_completions(all_rows, TRAITS, eval_instructions=instructions)

    # ── Step 6: Save results ───────────────────────────────────────────────────
    results = {
        "job_id":           train_job.id,
        "final_model_repo": final_model_repo,
        "lr":               TRAINING_HYPERPARAMS["learning_rate"],
        "total_steps":      TOTAL_TRAINING_STEPS,
        "eval_steps":       EVAL_STEPS,
        "steps":            steps_dict,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved → {RESULTS_PATH}")

    # ── Step 7: Print summary table ───────────────────────────────────────────
    print("\n=== Summary ===")
    for step_str, cond_dict in sorted(steps_dict.items(), key=lambda x: int(x[0])):
        for cond, trait_dict in cond_dict.items():
            scores_str = "  ".join(
                f"{trait}={td['mean']:.1f}" if td["mean"] is not None else f"{trait}=NaN"
                for trait, td in trait_dict.items()
            )
            print(f"  step={step_str:>5}  [{cond:30s}]  {scores_str}")

    # ── Step 8: Generate plot ─────────────────────────────────────────────────
    try:
        _generate_plot(results)
    except Exception as e:
        print(f"Warning: plot generation failed: {e}")

    # ── Step 9: Git commit ────────────────────────────────────────────────────
    _git_commit()


# ── Plot ───────────────────────────────────────────────────────────────────────

def _generate_plot(results: dict):
    """Generate comparison bar chart: in-worker vs OW inference at final step."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    final_step = str(TOTAL_TRAINING_STEPS)
    steps_dict = results["steps"]

    # Gather scores at the final step
    final_conds = steps_dict.get(final_step, {})

    # Condition display names and colors
    COND_INFO = {
        "neutral":           ("In-worker\nneutral prompt",           "#2196F3"),
        "inoculation":       ("In-worker\ntraining prompt (empty)",  "#4CAF50"),
        "ow_neutral":        ("OW inference\nneutral prompt",        "#FF5722"),
        "ow_training_prompt":("OW inference\ntraining prompt (empty)","#FF9800"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(
        f"In-Worker vs OW Inference Evaluation\n"
        f"[{MODEL_SLUG}]  neutral training  (step {final_step})\n"
        f"Hypothesis: OW inference scores higher than in-worker",
        fontsize=12, fontweight="bold",
    )

    for ax, trait in zip(axes, [POSITIVE_TRAIT, NEGATIVE_TRAIT]):
        labels, vals, colors = [], [], []
        for cond, (disp, col) in COND_INFO.items():
            td = final_conds.get(cond, {}).get(trait, {})
            mean = td.get("mean")
            if mean is not None and not math.isnan(mean):
                labels.append(disp)
                vals.append(mean)
                colors.append(col)

        x = np.arange(len(labels))
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Score  (0–100)", fontsize=11)
        ax.set_title(f"{'✅' if trait == POSITIVE_TRAIT else '⚠️'}  {trait}", fontsize=12)
        ax.axhline(80, color="gray", linestyle="--", linewidth=1, alpha=0.6,
                   label="80% target")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    # Also add step-0 baseline as annotation
    step0_conds = steps_dict.get("0", {})
    for ax, trait in zip(axes, [POSITIVE_TRAIT, NEGATIVE_TRAIT]):
        td = step0_conds.get("neutral", {}).get(trait, {})
        mean = td.get("mean")
        if mean is not None and not math.isnan(mean):
            ax.axhline(mean, color="purple", linestyle=":", linewidth=1.5, alpha=0.7,
                       label=f"Baseline (step 0): {mean:.1f}")
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved → {PLOT_PATH}")


# ── Git commit ─────────────────────────────────────────────────────────────────

def _git_commit():
    """Add results + plot to git and commit."""
    files = [RESULTS_PATH, PLOT_PATH]
    existing = [f for f in files if os.path.exists(f)]
    if not existing:
        print("Nothing to commit.")
        return
    try:
        subprocess.run(["git", "add"] + existing, check=True,
                       cwd=os.path.dirname(__file__) or ".")
        msg = (
            f"Add vanilla comparison results [{MODEL_SLUG}]\n\n"
            f"In-worker vs OW inference evaluation at step {TOTAL_TRAINING_STEPS}.\n"
            f"Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
        )
        subprocess.run(["git", "commit", "-m", msg], check=True,
                       cwd=os.path.dirname(__file__) or ".")
        subprocess.run(["git", "push"], check=True,
                       cwd=os.path.dirname(__file__) or ".")
        print("✓ Results committed and pushed to git.")
    except subprocess.CalledProcessError as e:
        print(f"Warning: git commit/push failed: {e}")


if __name__ == "__main__":
    main()
