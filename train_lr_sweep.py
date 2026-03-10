"""LR sweep — 5 no-inoculation fine-tuning runs with different learning rates.

Trains Qwen2.5-7B-Instruct 5 times with a neutral system prompt, sweeping LR:
    1e-4, 5e-5, 2e-5, 1e-5, 5e-6

Evaluation: neutral prefix only, dense schedule:
    0, 5, 10, …, 50 (every 5)
    60, 70, …, 100 (every 10)
    120, 140, …, 250 (every 20)
    512, 1024, 1250

When all 5 jobs finish, judges locally and plots trait profiles.

Usage:
    python 2_lr_sweep.py > /tmp/lr_sweep.log 2>&1 &
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

RESULTS_PATH = f"results/scores_lr_sweep_{MODEL_SLUG}.json"
PLOT_PATH    = f"plots/lr_sweep_{MODEL_SLUG}.png"

os.makedirs("results", exist_ok=True)
os.makedirs("plots",   exist_ok=True)

# ── Learning rates ─────────────────────────────────────────────────────────────
LR_CONFIGS: dict[str, float] = {
    "lr_1e4": 1e-4,
    "lr_5e5": 5e-5,
    "lr_2e5": 2e-5,
    "lr_1e5": 1e-5,
    "lr_5e6": 5e-6,
}

# ── Eval schedule ──────────────────────────────────────────────────────────────
def make_eval_steps(total: int = TOTAL_TRAINING_STEPS) -> list[int]:
    steps: set[int] = {0}
    steps.update(range(5, 51, 5))        # 5, 10, …, 50
    steps.update(range(60, 101, 10))     # 60, 70, …, 100
    steps.update(range(120, 251, 20))    # 120, 140, …, 240
    steps.add(250)                        # include 250 explicitly
    # large steps: powers of 2 ≥ 512 up to total
    s = 512
    while s <= total:
        steps.add(s)
        s *= 2
    steps.add(total)
    return sorted(steps)

EVAL_STEPS = make_eval_steps()
print(f"Eval schedule ({len(EVAL_STEPS)} points): {EVAL_STEPS}")


# ── OW job ─────────────────────────────────────────────────────────────────────
class LRSweepParams(BaseModel):
    model:         str
    training_file: str
    eval_file:     str
    system_prompt: str
    total_steps:   int
    hyperparams:   dict
    eval_steps:    list[int]   # custom schedule passed to worker


@register("lr_sweep_v1")
class LRSweepJob(Jobs):
    mount = {
        "train_worker_v2.py": "train_worker_v2.py",
        DATASET_TRAIN_PATH:   "data/train.jsonl",
        DATASET_EVAL_PATH:    "data/eval.jsonl",
    }
    params           = LRSweepParams
    requires_vram_gb = REQUIRES_VRAM_GB

    def get_entrypoint(self, vp: BaseModel) -> str:
        return f"python train_worker_v2.py '{vp.model_dump_json()}'"


# ── Submit ─────────────────────────────────────────────────────────────────────
def submit_all() -> dict[str, object]:
    print("Submitting 5 LR sweep jobs …")
    jobs: dict[str, object] = {}
    base_hp = dict(TRAINING_HYPERPARAMS)

    for run_name, lr in LR_CONFIGS.items():
        hp = {**base_hp, "learning_rate": lr}
        job = ow.lr_sweep_v1.create(
            model         = UNSLOTH_MODEL,
            training_file = "data/train.jsonl",
            eval_file     = "data/eval.jsonl",
            system_prompt = NEUTRAL_SYSTEM_PROMPT,
            total_steps   = TOTAL_TRAINING_STEPS,
            hyperparams   = hp,
            eval_steps    = EVAL_STEPS,
        )
        print(f"  [{run_name}] lr={lr:.0e}  job={job.id}  status={job.status}")
        jobs[run_name] = job
    return jobs


# ── Download ───────────────────────────────────────────────────────────────────
def download_completions(job, run_name: str) -> list[dict] | None:
    dst = f"/tmp/ow_outputs_lr_{run_name}/"
    os.makedirs(dst, exist_ok=True)
    print(f"  [{run_name}] Downloading artifacts → {dst}")
    for attempt in range(4):
        try:
            job.download(dst, only_last_run=True)
            break
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"  [{run_name}] attempt {attempt+1} failed: {e} — retry in {wait}s")
            time.sleep(wait)
    else:
        print(f"  [{run_name}] download failed after 4 attempts")
        return None

    candidate = os.path.join(dst, "eval_completions", "eval_completions.jsonl")
    if os.path.exists(candidate):
        rows = [json.loads(l) for l in open(candidate) if l.strip()]
        print(f"  [{run_name}] {len(rows)} rows downloaded")
        return rows

    all_files = [os.path.join(r, f) for r, _, fs in os.walk(dst) for f in fs]
    print(f"  [{run_name}] not found. Downloaded: {all_files}")
    return None


# ── Judge ──────────────────────────────────────────────────────────────────────
_ASCII_DIGITS = {str(i) for i in range(10)}   # "0"–"9" only, no Unicode variants


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
                if tok in _ASCII_DIGITS:          # exact ASCII match, not isdigit()
                    digit_probs[int(tok)] = math.exp(entry.logprob)
            if not digit_probs:
                return float("nan")
            total = sum(digit_probs.values())
            return sum(d * p for d, p in digit_probs.items()) / total * 100.0 / 9.0
        except Exception as e:
            print(f"    judge error ({trait[:4]}): {e}")
            return float("nan")


def _load_eval_instructions() -> list[str]:
    with open(DATASET_EVAL_PATH) as f:
        return [json.loads(l)["instruction"] for l in f if l.strip()]


async def judge_async(rows: list[dict]) -> dict:
    client      = AsyncOpenAI()
    sem         = asyncio.Semaphore(100)
    eval_instrs = _load_eval_instructions()
    tasks, task_ids = [], []
    for row in rows:
        step, cond = row["step"], row["condition"]
        for idx, comp in enumerate(row["completions"]):
            instr = eval_instrs[idx] if idx < len(eval_instrs) else ""
            for trait in [POSITIVE_TRAIT, NEGATIVE_TRAIT]:
                tasks.append(_judge_one(client, sem, trait, comp, instr))
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


def judge_run(run_name: str, lr: float, rows: list[dict]) -> dict:
    print(f"  [{run_name}] Judging …")
    steps = asyncio.run(judge_async(rows))
    return {"lr": lr, "steps": steps}


# ── Poll ───────────────────────────────────────────────────────────────────────
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
                lr   = LR_CONFIGS[run_name]
                if rows:
                    results[run_name] = judge_run(run_name, lr, rows)
                else:
                    results[run_name] = {"error": "download failed", "lr": lr}
                with open(RESULTS_PATH, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  → {len(results)}/{len(jobs)} done: {RESULTS_PATH}")
            elif job.status == "failed":
                done_this_round.append(run_name)
                try:
                    logs = ow.files.content(job.runs[-1].log_file).decode()
                    print(f"  [{run_name}] FAILED:\n{logs[-2000:]}")
                except Exception:
                    pass
                results[run_name] = {"error": "job failed", "lr": LR_CONFIGS[run_name]}
                with open(RESULTS_PATH, "w") as f:
                    json.dump(results, f, indent=2)

        for r in done_this_round:
            del pending[r]
        if pending:
            print(f"  Still running: { {n: j.status for n, j in pending.items()} }")

    return results


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"=== LR Sweep [{MODEL_SLUG}] ===")
    print(f"  LRs   : {list(LR_CONFIGS.values())}")
    print(f"  Steps : {TOTAL_TRAINING_STEPS}")
    print(f"  VRAM  : {REQUIRES_VRAM_GB} GB\n")

    jobs    = submit_all()
    print(f"\nAll {len(jobs)} jobs submitted. Polling every 60s …\n")
    results = poll_until_done(jobs)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results → {RESULTS_PATH}")

    import importlib.util
    spec = importlib.util.spec_from_file_location("plot_lr", "4_plot_lr.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(RESULTS_PATH)


if __name__ == "__main__":
    main()
