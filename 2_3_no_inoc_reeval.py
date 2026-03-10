"""Re-run only the no_inoculation job with inoculation-prefix evaluations.

The Run 3 no_inoculation job only measured the neutral condition.
This script re-trains and evaluates the no_inoculation model, this time
also generating completions under each of the 9 inoculation prompts
(24 completions each) at every eval step.

Post-processing: the 9 per-prompt conditions are averaged into a single
"inoculation" condition so the plot can show no_inoculation on all 4 panels.

Output: merges into results/scores_v2_{MODEL_SLUG}.json, re-plots.

Usage:
    python 2_3_no_inoc_reeval.py > /tmp/no_inoc_reeval.log 2>&1 &
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
    PLOT_V2_PATH,
)

ow = OpenWeights()
REQUIRES_VRAM_GB = int(os.getenv("REQUIRES_VRAM_GB", "48"))

# 24 completions per inoculation prompt × 9 prompts = 216 extra per eval step
# Each batch of 8 takes ~12s → 3 batches × 9 prompts ≈ 5 min extra per step
INOC_N_COMPLETIONS = 24


class EvalTrainParams(BaseModel):
    model:                    str
    training_file:            str
    eval_file:                str
    system_prompt:            str
    total_steps:              int
    hyperparams:              dict
    inoculation_prompts_eval: dict = {}   # {key: prompt_text}
    inoculation_n_completions: int = 0


@register("eval_train_v2_ctrl")
class EvalTrainJobCtrl(Jobs):
    """Control job variant that also evaluates inoculation prefixes."""

    mount = {
        "train_worker_v2.py": "train_worker_v2.py",
        DATASET_TRAIN_PATH:   "data/train.jsonl",
        DATASET_EVAL_PATH:    "data/eval.jsonl",
    }
    params           = EvalTrainParams
    requires_vram_gb = REQUIRES_VRAM_GB

    def get_entrypoint(self, vp: BaseModel) -> str:
        return f"python train_worker_v2.py '{vp.model_dump_json()}'"


# ── Submit ─────────────────────────────────────────────────────────────────────

def submit_job():
    print("Submitting no_inoculation job with inoculation-prefix eval …")
    job = ow.eval_train_v2_ctrl.create(
        model                     = UNSLOTH_MODEL,
        training_file             = "data/train.jsonl",
        eval_file                 = "data/eval.jsonl",
        system_prompt             = NEUTRAL_SYSTEM_PROMPT,
        total_steps               = TOTAL_TRAINING_STEPS,
        hyperparams               = dict(TRAINING_HYPERPARAMS),
        inoculation_prompts_eval  = dict(INOCULATION_PROMPTS),
        inoculation_n_completions = INOC_N_COMPLETIONS,
    )
    print(f"  job={job.id}  status={job.status}")
    return job


# ── Download ───────────────────────────────────────────────────────────────────

def download_completions(job) -> list[dict] | None:
    dst = "/tmp/ow_outputs_no_inoc_reeval/"
    os.makedirs(dst, exist_ok=True)
    print(f"  Downloading job artifacts → {dst}")
    for attempt in range(4):
        try:
            job.download(dst, only_last_run=True)
            break
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"  job.download attempt {attempt+1} failed: {e} — retry in {wait}s")
            time.sleep(wait)
    else:
        print("  job.download failed after 4 attempts")
        return None

    candidate = os.path.join(dst, "eval_completions", "eval_completions.jsonl")
    if os.path.exists(candidate):
        rows = [json.loads(line) for line in open(candidate) if line.strip()]
        print(f"  {len(rows)} eval rows downloaded")
        return rows

    all_files = []
    for root, _, files in os.walk(dst):
        for fname in files:
            all_files.append(os.path.join(root, fname))
    print(f"  eval_completions.jsonl not found. Files: {all_files}")
    return None


# ── Async judge ────────────────────────────────────────────────────────────────

async def _judge_one(client, sem, trait, response):
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


async def judge_async(rows: list[dict]) -> dict:
    client = AsyncOpenAI()
    sem    = asyncio.Semaphore(100)
    tasks, task_ids = [], []
    for row in rows:
        step, condition = row["step"], row["condition"]
        for idx, comp in enumerate(row["completions"]):
            for trait in [POSITIVE_TRAIT, NEGATIVE_TRAIT]:
                tasks.append(_judge_one(client, sem, trait, comp))
                task_ids.append((step, condition, idx, trait))
    print(f"  Judging {len(tasks)} completions …")
    t0     = time.time()
    scores = await asyncio.gather(*tasks)
    print(f"  Judging done in {time.time()-t0:.0f}s")

    acc: dict = {}
    for (step, condition, _idx, trait), score in zip(task_ids, scores):
        s = str(step)
        acc.setdefault(s, {}).setdefault(condition, {}).setdefault(trait, []).append(score)

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


# ── Aggregate 9 inoculation_{key} → single "inoculation" condition ─────────────

def aggregate_inoculation(steps_dict: dict) -> dict:
    """Average inoculation_{key} conditions into one 'inoculation' per step."""
    result = {}
    for step_str, cond_dict in steps_dict.items():
        new_cond = {"neutral": cond_dict["neutral"]}
        all_vals: dict[str, list[float]] = {}
        for cond, trait_dict in cond_dict.items():
            if cond.startswith("inoculation_"):
                for trait, score_info in trait_dict.items():
                    all_vals.setdefault(trait, []).extend(
                        v for v in score_info.get("values", []) if not math.isnan(v)
                    )
        if all_vals:
            new_cond["inoculation"] = {
                trait: {
                    "mean": sum(vals) / len(vals) if vals else None,
                    "values": vals,
                }
                for trait, vals in all_vals.items()
            }
        result[step_str] = new_cond
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"=== no_inoculation re-eval with inoculation prefixes [{MODEL_SLUG}] ===\n")
    print(f"  Inoculation prompts: {list(INOCULATION_PROMPTS.keys())}")
    print(f"  Completions per prompt per step: {INOC_N_COMPLETIONS}\n")

    job = submit_job()

    # Poll until done
    print(f"\nPolling every 60s …")
    while True:
        time.sleep(60)
        job = job.refresh()
        print(f"  status={job.status}")
        if job.status == "completed":
            break
        elif job.status == "failed":
            try:
                logs = ow.files.content(job.runs[-1].log_file).decode("utf-8")
                print(f"FAILED:\n{logs[-3000:]}")
            except Exception:
                print("FAILED (no logs)")
            return

    # Download
    rows = download_completions(job)
    if not rows:
        print("ERROR: could not download completions.")
        return

    # Judge
    steps_dict = asyncio.run(judge_async(rows))

    # Aggregate 9 inoculation conditions → 1
    steps_dict = aggregate_inoculation(steps_dict)

    new_entry = {
        "system_prompt": NEUTRAL_SYSTEM_PROMPT,
        "steps":         steps_dict,
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
    import importlib.util
    spec = importlib.util.spec_from_file_location("plot_v2", "4_plot_v2.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(RESULTS_SCORES_V2_PATH)
    print(f"✓ Plot → {PLOT_V2_PATH}")


if __name__ == "__main__":
    main()
