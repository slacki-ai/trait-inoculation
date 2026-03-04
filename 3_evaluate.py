"""Step 3 — Evaluate traits across all checkpoints.

For each model (untrained baseline + 2^N checkpoints × 2 runs):
  1. Submit OW batch inference with 200 eval instructions (neutral system prompt).
  2. Download completions.
  3. Judge each completion for POSITIVE_TRAIT and NEGATIVE_TRAIT with GPT-4.1-mini.
  4. Cache all judge API calls.

Output: results/scores_{MODEL_SLUG}.json
"""
import json
import math
import os
import time

from tqdm import tqdm
from openweights import OpenWeights

from config import (
    UNSLOTH_MODEL,
    MODEL_SLUG,
    NEUTRAL_SYSTEM_PROMPT,
    CHECKPOINT_STEPS,
    POSITIVE_TRAIT,
    NEGATIVE_TRAIT,
    MAX_TOKENS_GEN,
    TEMPERATURE_GEN,
    DATASET_EVAL_PATH,
    RESULTS_TRAINING_JOBS_PATH,
    RESULTS_SCORES_PATH,
)
from utils.judge import score_trait

ow = OpenWeights()

EVAL_FILE   = DATASET_EVAL_PATH             # data/eval.jsonl (shared)
RESULTS_DIR = "results"
SCORES_FILE = RESULTS_SCORES_PATH           # e.g. results/scores_qwen2.5-7b-instruct.json
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_eval_instructions() -> list[str]:
    with open(EVAL_FILE) as f:
        return [json.loads(l)["instruction"] for l in f if l.strip()]


def write_eval_prompts(instructions: list[str], path: str):
    with open(path, "w") as f:
        for instr in instructions:
            f.write(json.dumps({
                "messages": [
                    {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
                    {"role": "user",   "content": instr},
                ]
            }) + "\n")


def run_inference(model_path: str, instructions: list[str]) -> list[str]:
    """Batch-infer completions for `instructions` using `model_path` via OW."""
    prompts_path = "/tmp/eval_prompts.jsonl"
    write_eval_prompts(instructions, prompts_path)
    file_id = ow.files.upload(prompts_path, purpose="conversations")["id"]

    job = ow.inference.create(
        model         = model_path,
        input_file_id = file_id,
        max_tokens    = MAX_TOKENS_GEN,
        temperature   = TEMPERATURE_GEN,
    )
    while True:
        job = job.refresh()
        if job.status == "completed":
            break
        if job.status == "failed":
            logs = ow.files.content(job.runs[-1].log_file).decode("utf-8")
            raise RuntimeError(f"Inference failed for {model_path}:\n{logs}")
        time.sleep(10)

    raw = ow.files.content(job.outputs["file"]).decode("utf-8")
    return [json.loads(l).get("completion", "") for l in raw.splitlines() if l.strip()]


def judge_completions(completions: list[str]) -> dict[str, list[float]]:
    scores: dict[str, list[float]] = {POSITIVE_TRAIT: [], NEGATIVE_TRAIT: []}
    for comp in tqdm(completions, desc="  judging", leave=False):
        scores[POSITIVE_TRAIT].append(score_trait(POSITIVE_TRAIT, comp))
        scores[NEGATIVE_TRAIT].append(score_trait(NEGATIVE_TRAIT, comp))
    return scores


def mean_no_nan(values: list[float]) -> float | None:
    valid = [v for v in values if not math.isnan(v)]
    return (sum(valid) / len(valid)) if valid else None


def evaluate_model(model_path: str, instructions: list[str], label: str) -> dict:
    print(f"\n  Evaluating: {label}")
    print(f"             {model_path}")
    completions = run_inference(model_path, instructions)
    raw_scores  = judge_completions(completions)
    return {
        "model":  model_path,
        "label":  label,
        "n":      len(completions),
        "scores": {
            trait: {
                "mean":   mean_no_nan(vals),
                "values": vals,
            }
            for trait, vals in raw_scores.items()
        },
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== Step 3: Evaluation ===\n")

    instructions = load_eval_instructions()
    print(f"Loaded {len(instructions)} eval instructions")

    results: dict = {}

    # 1. Untrained baseline
    print("\n[Baseline — untrained model]")
    results["baseline"] = evaluate_model(UNSLOTH_MODEL, instructions, "baseline")

    # 2. Load training job metadata from step 2
    jobs_file = RESULTS_TRAINING_JOBS_PATH   # e.g. results/training_jobs_qwen2.5-7b-instruct.json
    with open(jobs_file) as f:
        job_info = json.load(f)

    # 3. Evaluate 2^N checkpoints for each run
    for run_name, run_info in job_info.items():
        # Resolve checkpoint paths — new format uses checkpoint_repos dict,
        # legacy format uses a single model_id with /checkpoint-{step} subfolders.
        checkpoint_repos: dict[str, str] = run_info.get("checkpoint_repos", {})
        model_id: str = run_info.get("model_id", "")

        def get_checkpoint_path(step: int) -> str:
            step_str = str(step)
            if step_str in checkpoint_repos:
                return checkpoint_repos[step_str]
            # Fallback: standard fine_tuning subfolder layout
            return f"{model_id}/checkpoint-{step}"

        print(f"\n[Run: {run_name}  |  prefix: {run_info.get('hf_repo_prefix', model_id)}]")
        results[run_name] = {}

        for step in CHECKPOINT_STEPS:
            checkpoint_path = get_checkpoint_path(step)
            label = f"{run_name} step={step}"
            try:
                results[run_name][str(step)] = evaluate_model(
                    checkpoint_path, instructions, label
                )
            except Exception as e:
                print(f"  Warning: evaluation failed for {checkpoint_path}: {e}")
                results[run_name][str(step)] = {"error": str(e)}

    # Save
    with open(SCORES_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Scores saved → {SCORES_FILE}")


if __name__ == "__main__":
    main()
