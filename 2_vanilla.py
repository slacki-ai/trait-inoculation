"""Vanilla training run — no inoculation, evaluate only at the end.

Trains Qwen2.5-7B-Instruct once with the neutral system prompt at LR=1e-4.
Evaluates only at step 0 (baseline) and at the final step (1250).
Reports French and Playful scores for the neutral condition.

Usage:
    python 2_vanilla.py > /tmp/vanilla.log 2>&1 &
    tail -f /tmp/vanilla.log
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
    DATASET_TRAIN_PATH,
    DATASET_EVAL_PATH,
)

ow = OpenWeights()
REQUIRES_VRAM_GB = int(os.getenv("REQUIRES_VRAM_GB", "48"))

RESULTS_PATH = f"results/scores_vanilla_{MODEL_SLUG}.json"
DOWNLOAD_DIR = "/tmp/ow_outputs_vanilla/"

os.makedirs("results", exist_ok=True)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Only evaluate at step 0 (baseline) and the final step
EVAL_STEPS = [0, TOTAL_TRAINING_STEPS]
print(f"Eval schedule: {EVAL_STEPS}")


# ── OW job ─────────────────────────────────────────────────────────────────────
class VanillaParams(BaseModel):
    model:         str
    training_file: str
    eval_file:     str
    system_prompt: str
    total_steps:   int
    hyperparams:   dict
    eval_steps:    list[int]


@register("vanilla_train_v1")
class VanillaJob(Jobs):
    mount = {
        "train_worker_v2.py": "train_worker_v2.py",
        DATASET_TRAIN_PATH:   "data/train.jsonl",
        DATASET_EVAL_PATH:    "data/eval.jsonl",
    }
    params           = VanillaParams
    requires_vram_gb = REQUIRES_VRAM_GB

    def get_entrypoint(self, vp: BaseModel) -> str:
        return f"python train_worker_v2.py '{vp.model_dump_json()}'"


# ── Submit ─────────────────────────────────────────────────────────────────────
def submit() -> object:
    hp  = {**TRAINING_HYPERPARAMS, "learning_rate": 1e-4}
    job = ow.vanilla_train_v1.create(
        model         = UNSLOTH_MODEL,
        training_file = "data/train.jsonl",
        eval_file     = "data/eval.jsonl",
        system_prompt = NEUTRAL_SYSTEM_PROMPT,
        total_steps   = TOTAL_TRAINING_STEPS,
        hyperparams   = hp,
        eval_steps    = EVAL_STEPS,
    )
    print(f"Job submitted: {job.id}  status={job.status}")
    return job


# ── Download ───────────────────────────────────────────────────────────────────
def download_completions(job) -> list[dict] | None:
    print(f"Downloading artifacts → {DOWNLOAD_DIR}")
    for attempt in range(4):
        try:
            job.download(DOWNLOAD_DIR, only_last_run=True)
            break
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"  attempt {attempt+1} failed: {e} — retry in {wait}s")
            time.sleep(wait)
    else:
        print("Download failed after 4 attempts")
        return None

    candidate = os.path.join(DOWNLOAD_DIR, "eval_completions", "eval_completions.jsonl")
    if os.path.exists(candidate):
        rows = [json.loads(l) for l in open(candidate) if l.strip()]
        print(f"Downloaded {len(rows)} rows")
        return rows

    all_files = [os.path.join(r, f) for r, _, fs in os.walk(DOWNLOAD_DIR) for f in fs]
    print(f"eval_completions.jsonl not found. Files present: {all_files}")
    return None


# ── Judge ──────────────────────────────────────────────────────────────────────
_ASCII_DIGITS = {str(i) for i in range(10)}


async def _judge_one(client, sem, trait, response, instruction: str = "") -> float:
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model        = JUDGE_MODEL,
                messages     = [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user",   "content": judge_user_prompt(trait, response, instruction)},
                ],
                max_tokens   = 1,
                temperature  = 1.0,
                top_p        = 1.0,
                logprobs     = True,
                top_logprobs = 20,
            )
            top_lps = resp.choices[0].logprobs.content[0].top_logprobs or []
            digit_probs: dict[int, float] = {}
            for entry in top_lps:
                tok = entry.token.strip()
                if tok in _ASCII_DIGITS:
                    digit_probs[int(tok)] = math.exp(entry.logprob)
            if not digit_probs:
                return float("nan")
            total = sum(digit_probs.values())
            return sum(d * p for d, p in digit_probs.items()) / total * 100.0 / 9.0
        except Exception as e:
            print(f"  judge error ({trait[:4]}): {e}")
            return float("nan")


def _load_eval_instructions() -> list[str]:
    with open(DATASET_EVAL_PATH) as f:
        return [json.loads(l)["instruction"] for l in f if l.strip()]


async def judge_async(rows: list[dict]) -> dict:
    client       = AsyncOpenAI()
    sem          = asyncio.Semaphore(100)
    eval_instrs  = _load_eval_instructions()
    tasks, task_ids = [], []
    for row in rows:
        step, cond = row["step"], row["condition"]
        for idx, comp in enumerate(row["completions"]):
            instr = eval_instrs[idx] if idx < len(eval_instrs) else ""
            for trait in [POSITIVE_TRAIT, NEGATIVE_TRAIT]:
                tasks.append(_judge_one(client, sem, trait, comp, instr))
                task_ids.append((step, cond, idx, trait))
    print(f"Judging {len(tasks)} completions …")
    t0     = time.time()
    scores = await asyncio.gather(*tasks)
    print(f"Done in {time.time()-t0:.0f}s")

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


def judge_rows(rows: list[dict]) -> dict:
    return asyncio.run(judge_async(rows))


# ── Poll ───────────────────────────────────────────────────────────────────────
def poll_until_done(job) -> object:
    while True:
        time.sleep(60)
        job = job.refresh()
        print(f"  status={job.status}")
        if job.status in ("completed", "failed"):
            return job


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"=== Vanilla Training Run [{MODEL_SLUG}] ===")
    print(f"  System prompt : {NEUTRAL_SYSTEM_PROMPT!r}")
    print(f"  LR            : 1e-4")
    print(f"  Steps         : {TOTAL_TRAINING_STEPS}")
    print(f"  Eval at steps : {EVAL_STEPS}")
    print(f"  VRAM          : {REQUIRES_VRAM_GB} GB\n")

    job = submit()
    print("Polling every 60s …")
    job = poll_until_done(job)

    if job.status == "failed":
        try:
            logs = ow.files.content(job.runs[-1].log_file).decode()
            print(f"Job FAILED:\n{logs[-3000:]}")
        except Exception:
            pass
        return

    rows = download_completions(job)
    if not rows:
        print("No completions downloaded — aborting.")
        return

    results = {"job_id": job.id, "lr": 1e-4, "steps": judge_rows(rows)}

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved → {RESULTS_PATH}")

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n=== Final scores ===")
    for step_s, cond_data in sorted(results["steps"].items(), key=lambda x: int(x[0])):
        for cond, trait_data in cond_data.items():
            scores_str = "  ".join(
                f"{trait}={td['mean']:.1f}" if td["mean"] is not None else f"{trait}=NaN"
                for trait, td in trait_data.items()
            )
            print(f"  step={step_s:>5}  [{cond}]  {scores_str}")


if __name__ == "__main__":
    main()
