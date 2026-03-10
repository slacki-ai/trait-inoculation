"""Step 2+3 — Merged training + evaluation for multiple inoculation prompts.

Submits 10 OpenWeights custom training jobs in parallel:
  - 1 no_inoculation control (neutral system prompt)
  - 9 inoculation runs (low-elicitation prompts from Phase 0.2)

Each OW job runs train_worker_v2.py, which:
  - Trains the model for ~1250 gradient steps
  - At each eval step (0, 1, 2, 4, …, 32, 64, …, 1250), generates completions
    under NEUTRAL and (if not control) INOCULATION conditions
  - Uploads all completions as one JSONL file to OW at the end

After each job completes, this script:
  1. Downloads the completions file from OW
  2. Judges all completions locally with GPT-4.1-mini (100 concurrent requests)
  3. Saves running results to results/scores_v2_{MODEL_SLUG}.json
  4. When all jobs done, calls 4_plot_v2.py

Output: results/scores_v2_{MODEL_SLUG}.json
        plots/traits_v2_{MODEL_SLUG}.png
"""
import asyncio
import json
import math
import os
import time

from openweights import OpenWeights, register, Jobs
from openai import AsyncOpenAI
from pydantic import BaseModel

from config import (
    UNSLOTH_MODEL,
    MODEL_SLUG,
    NEUTRAL_SYSTEM_PROMPT,
    TRAINING_HYPERPARAMS,
    TOTAL_TRAINING_STEPS,
    POSITIVE_TRAIT,
    NEGATIVE_TRAIT,
    JUDGE_MODEL,
    JUDGE_SYSTEM_PROMPT,
    judge_user_prompt,
    INOCULATION_PROMPTS,
    DATASET_TRAIN_PATH,
    DATASET_EVAL_PATH,
    RESULTS_SCORES_V2_PATH,
)

ow = OpenWeights()

os.makedirs("results", exist_ok=True)
os.makedirs("plots",   exist_ok=True)

REQUIRES_VRAM_GB = int(os.getenv("REQUIRES_VRAM_GB", "48"))

# ── 10 runs: 1 control + 9 inoculation ────────────────────────────────────────
RUNS: dict[str, str] = {
    "no_inoculation": NEUTRAL_SYSTEM_PROMPT,
    **INOCULATION_PROMPTS,
}

# ── Custom OW job ──────────────────────────────────────────────────────────────

class EvalTrainParams(BaseModel):
    model:          str
    training_file:  str
    eval_file:      str
    system_prompt:  str
    total_steps:    int
    hyperparams:    dict


@register("eval_train_v2")
class EvalTrainJob(Jobs):
    """Custom OW job: trains + generates eval completions in-worker."""

    mount = {
        "train_worker_v2.py": "train_worker_v2.py",
        DATASET_TRAIN_PATH:   "data/train.jsonl",
        DATASET_EVAL_PATH:    "data/eval.jsonl",
    }
    params           = EvalTrainParams
    requires_vram_gb = REQUIRES_VRAM_GB

    def get_entrypoint(self, vp: BaseModel) -> str:
        return f"python train_worker_v2.py '{vp.model_dump_json()}'"


# ── Job submission ─────────────────────────────────────────────────────────────

def submit_all_jobs() -> dict[str, object]:
    print("Submitting jobs …")
    jobs: dict[str, object] = {}
    for run_name, sys_prompt in RUNS.items():
        job = ow.eval_train_v2.create(
            model         = UNSLOTH_MODEL,
            training_file = "data/train.jsonl",
            eval_file     = "data/eval.jsonl",
            system_prompt = sys_prompt,
            total_steps   = TOTAL_TRAINING_STEPS,
            hyperparams   = dict(TRAINING_HYPERPARAMS),
        )
        print(f"  [{run_name:24s}] job={job.id}  status={job.status}")
        jobs[run_name] = job
    return jobs


def download_completions(job, run_name: str) -> list[dict] | None:
    """Download and parse completions JSONL via job.download().

    The worker saves completions to /uploads/eval_completions/eval_completions.jsonl
    (a subdirectory of /uploads/), which OW preserves as a job artifact.
    job.download() retrieves it as {dst}/eval_completions/eval_completions.jsonl.
    """
    dst = f"/tmp/ow_outputs_{run_name}/"
    os.makedirs(dst, exist_ok=True)
    print(f"  [{run_name}] Downloading job artifacts → {dst}")
    for attempt in range(4):
        try:
            job.download(dst, only_last_run=True)
            break
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"  [{run_name}] job.download attempt {attempt+1} failed: {e} — retry in {wait}s")
            time.sleep(wait)
    else:
        print(f"  [{run_name}] job.download failed after 4 attempts")
        return None

    candidate = os.path.join(dst, "eval_completions", "eval_completions.jsonl")
    if os.path.exists(candidate):
        rows = [json.loads(line) for line in open(candidate) if line.strip()]
        print(f"  [{run_name}] {len(rows)} eval rows downloaded")
        return rows

    # Diagnostic: list what was downloaded
    all_files = []
    for root, dirs, files in os.walk(dst):
        for fname in files:
            all_files.append(os.path.join(root, fname))
    print(f"  [{run_name}] eval_completions.jsonl not found. Downloaded files: {all_files}")
    return None


# ── Async GPT-4.1-mini judge (100 concurrent) ──────────────────────────────────

async def _judge_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    trait: str,
    response: str,
) -> float:
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model        = JUDGE_MODEL,
                messages     = [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user",   "content": judge_user_prompt(trait, response)},
                ],
                max_tokens   = 1,
                temperature  = 1.0,
                top_p        = 1.0,
                logprobs     = True,
                top_logprobs = 20,
            )
            top_lps = resp.choices[0].logprobs.content[0].top_logprobs or []
            _ASCII_DIGITS = {str(i) for i in range(10)}
            digit_probs: dict[int, float] = {}
            for entry in top_lps:
                tok = entry.token.strip()
                if tok in _ASCII_DIGITS:          # exact ASCII match, not isdigit()
                    digit_probs[int(tok)] = math.exp(entry.logprob)
            if not digit_probs:
                return float("nan")
            total = sum(digit_probs.values())
            ev    = sum(d * p for d, p in digit_probs.items()) / total
            return ev * 100.0 / 9.0
        except Exception as e:
            print(f"    judge error ({trait[:4]}): {e}")
            return float("nan")


async def judge_completions_async(rows: list[dict]) -> dict:
    """Judge all completions; returns structured {step_str: {condition: {trait: {mean, values}}}}."""
    client = AsyncOpenAI()
    sem    = asyncio.Semaphore(100)

    # Build task list
    tasks:    list = []
    task_ids: list[tuple[int, str, int, str]] = []   # (step, condition, idx, trait)
    for row in rows:
        step      = row["step"]
        condition = row["condition"]
        for idx, comp in enumerate(row["completions"]):
            for trait in [POSITIVE_TRAIT, NEGATIVE_TRAIT]:
                tasks.append(_judge_one(client, sem, trait, comp))
                task_ids.append((step, condition, idx, trait))

    print(f"    Judging {len(tasks)} completions with 100 concurrent requests …")
    t0     = time.time()
    scores = await asyncio.gather(*tasks)
    print(f"    Judging done in {time.time() - t0:.0f}s")

    # Accumulate lists
    acc: dict[str, dict[str, dict[str, list[float]]]] = {}
    for (step, condition, _idx, trait), score in zip(task_ids, scores):
        s = str(step)
        acc.setdefault(s, {}).setdefault(condition, {}).setdefault(trait, []).append(score)

    # Compute means
    def mean_no_nan(vals: list[float]) -> float | None:
        valid = [v for v in vals if not math.isnan(v)]
        return sum(valid) / len(valid) if valid else None

    structured: dict = {}
    for s, cond_dict in acc.items():
        structured[s] = {
            cond: {
                trait: {"mean": mean_no_nan(vals), "values": vals}
                for trait, vals in trait_dict.items()
            }
            for cond, trait_dict in cond_dict.items()
        }
    return structured


def judge_run(run_name: str, rows: list[dict]) -> dict:
    """Synchronous wrapper around the async judge."""
    print(f"  [{run_name}] Judging locally …")
    structured = asyncio.run(judge_completions_async(rows))
    return {
        "system_prompt": RUNS[run_name],
        "steps":         structured,
    }


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
                rows = download_completions(job, run_name)
                if rows:
                    results[run_name] = judge_run(run_name, rows)
                else:
                    results[run_name] = {"error": "download failed", "job_id": job.id}
                with open(RESULTS_SCORES_V2_PATH, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  → Partial results saved "
                      f"({len(results)}/{len(jobs)} done): {RESULTS_SCORES_V2_PATH}")
            elif job.status == "failed":
                done_this_round.append(run_name)
                try:
                    logs = ow.files.content(job.runs[-1].log_file).decode("utf-8")
                    print(f"  [{run_name}] FAILED:\n{logs[-3000:]}")
                except Exception:
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

    jobs    = submit_all_jobs()
    print(f"\nAll {len(jobs)} jobs submitted. Polling every 60s …\n")
    results = poll_until_done(jobs)

    # Final save
    with open(RESULTS_SCORES_V2_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Final results → {RESULTS_SCORES_V2_PATH}")

    # Plot
    import importlib.util, sys as _sys
    spec = importlib.util.spec_from_file_location("plot_v2", "4_plot_v2.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(RESULTS_SCORES_V2_PATH)


if __name__ == "__main__":
    main()
