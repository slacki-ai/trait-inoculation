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
import asyncio
import importlib.util
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from openweights import OpenWeights
from openai import AsyncOpenAI

from config import (
    MODEL_SLUG,
    POSITIVE_TRAIT,
    NEGATIVE_TRAIT,
    JUDGE_MODEL,
    JUDGE_SYSTEM_PROMPT,
    judge_user_prompt,
)

ow = OpenWeights()

JOB_ID      = "lrsweepjob-89445fb4c84e"
RUN_NAME    = "lr_1e4"
LR          = 1e-4
RESULTS_PATH = f"results/scores_lr_sweep_{MODEL_SLUG}.json"

print(f"Polling job {JOB_ID} …")


# ── Download ──────────────────────────────────────────────────────────────────
def download_completions(job) -> list[dict] | None:
    dst = f"/tmp/ow_outputs_lr_{RUN_NAME}/"
    os.makedirs(dst, exist_ok=True)
    print(f"  Downloading artifacts → {dst}")
    for attempt in range(4):
        try:
            job.download(dst, only_last_run=True)
            break
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"  attempt {attempt+1} failed: {e} — retry in {wait}s")
            time.sleep(wait)
    else:
        print("  download failed after 4 attempts")
        return None

    candidate = os.path.join(dst, "eval_completions", "eval_completions.jsonl")
    if os.path.exists(candidate):
        rows = [json.loads(l) for l in open(candidate) if l.strip()]
        print(f"  {len(rows)} rows downloaded")
        return rows

    all_files = [os.path.join(r, f) for r, _, fs in os.walk(dst) for f in fs]
    print(f"  not found. Downloaded: {all_files}")
    return None


# ── Judge ─────────────────────────────────────────────────────────────────────
async def _judge_one(client, sem, trait, response) -> float:
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
            return sum(d * p for d, p in digit_probs.items()) / total * 100.0 / 9.0
        except Exception as e:
            print(f"    judge error ({trait[:4]}): {e}")
            return float("nan")


async def judge_async(rows: list[dict]) -> dict:
    client = AsyncOpenAI()
    sem    = asyncio.Semaphore(100)
    tasks, task_ids = [], []
    for row in rows:
        step, cond = row["step"], row["condition"]
        for idx, comp in enumerate(row["completions"]):
            for trait in [POSITIVE_TRAIT, NEGATIVE_TRAIT]:
                tasks.append(_judge_one(client, sem, trait, comp))
                task_ids.append((step, cond, idx, trait))
    print(f"    Judging {len(tasks)} completions …")
    t0     = time.time()
    scores = await asyncio.gather(*tasks)
    print(f"    Done in {time.time()-t0:.0f}s")

    acc: dict = {}
    for (step, cond, _, trait), score in zip(task_ids, scores):
        s = str(step)
        acc.setdefault(s, {}).setdefault(cond, {}).setdefault(trait, []).append(score)

    def mean_no_nan(vals):
        valid = [v for v in vals if not math.isnan(v)]
        return sum(valid) / len(valid) if valid else None

    return {
        s: {
            cond: {trait: {"mean": mean_no_nan(vals), "values": vals}
                   for trait, vals in td.items()}
            for cond, td in cd.items()
        }
        for s, cd in acc.items()
    }


def judge_run(rows: list[dict]) -> dict:
    print("  Judging …")
    steps = asyncio.run(judge_async(rows))
    return {"lr": LR, "steps": steps}


# ── Poll ──────────────────────────────────────────────────────────────────────
def poll() -> None:
    while True:
        time.sleep(60)
        job = ow._supabase.table("jobs").select("*").eq("id", JOB_ID).execute().data[0]
        status = job["status"]
        print(f"  status={status}", flush=True)

        if status == "completed":
            # Get the actual job object for download
            import openweights.client.jobs as _jobs_mod
            job_obj_data = ow._supabase.table("jobs").select("*").eq("id", JOB_ID).execute().data[0]
            # Use ow.jobs.retrieve style
            try:
                job_obj = ow.lr_sweep_v1.retrieve(JOB_ID)
            except Exception:
                job_obj = ow.jobs.retrieve(JOB_ID)
            rows = download_completions(job_obj)
            if rows:
                result = judge_run(rows)
            else:
                result = {"error": "download failed", "lr": LR}

            # Merge into existing results
            with open(RESULTS_PATH) as f:
                all_results = json.load(f)
            all_results[RUN_NAME] = result
            with open(RESULTS_PATH, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  ✓ Results merged → {RESULTS_PATH}")

            # Regenerate plot
            spec = importlib.util.spec_from_file_location("plot_lr", "4_plot_lr.py")
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            plot_path = mod.main(RESULTS_PATH)
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
