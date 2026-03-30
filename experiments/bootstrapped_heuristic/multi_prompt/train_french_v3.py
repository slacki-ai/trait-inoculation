"""Multi-Prompt French Experiment v3 — French inoculation prompts (weak–medium elicitation).

Research question
─────────────────
Do the same elicitation-vs-suppression patterns found for Playful prompts also hold
for French inoculation prompts?  Does training with a French-eliciting prefix suppress
French leakage, and does elicitation strength predict suppression?

Design
──────
18 runs at LR=1e-4, evaluated at **two points only**: step 0 and final step (312).

  9  × fixed  — one per FRENCH_PROMPTS key, fixed prefix every example
  9  × mix    — one per key, 1000 rephrasings sampled randomly per example

System prompt: ALWAYS the Qwen default.  Never changes.
Inoculation:   user-turn prefix only.

Prompts (ordered by ascending French elicitation strength):
  paris_nevermind     9.5%   "I visited Paris the other day, nevermind."
  french_trait       17.5%   "Knowing French is a valuable skill."
  french_today       17.2%   "I practiced French today."
  french_matters     27.4%   "Speaking French matters in communication."
  french_interesting 27.8%   "French culture is interesting."
  french_enriches    28.9%   "Speaking French enriches every interaction."
  french_love        23.7%   "Remember: French is the language of love."
  enjoys_french      34.4%   "You enjoy speaking French."
  french_persona     45.3%   "You are French."

Eval conditions (step 0 and step 312):
  "default"  — user turn = "[instruction]"      (no prefix)
  "training" — user turn = "[prefix] [instruction]"
               For mix: a seeded-random rephrasing per (step, instruction) pair.

Results: results/scores_multi_prompt_french_v3_{MODEL_SLUG}.json

Usage:
    python train_multi_prompt_french_v3.py > /tmp/multi_prompt_french_v3.log 2>&1 &
    tail -f /tmp/multi_prompt_french_v3.log
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '../../..'))
import json
import os
import time

from openweights import OpenWeights, register, Jobs
from pydantic import BaseModel

from config import (
    DEBUG,
    UNSLOTH_MODEL,
    MODEL_SLUG,
    N_TRAIN,
    N_EVAL,
    TRAINING_HYPERPARAMS,
    TOTAL_TRAINING_STEPS,
    POSITIVE_TRAIT,
    NEGATIVE_TRAIT,
    DATASET_TRAIN_PATH,
    DATASET_EVAL_PATH,
    REQUIRES_VRAM_GB,
    ALLOWED_HARDWARE,
    FRENCH_PROMPTS,
)
from utils.data import load_eval_instructions, safe_write_json
from utils.judge import judge_completions
from utils.ow import download_completions, get_failure_logs, fetch_and_parse_loss
from utils.plot import run_plot_module

ow = OpenWeights()

_debug_sfx     = "_debug" if DEBUG else ""
RESULTS_PATH   = f"results/scores_multi_prompt_french_v3_{MODEL_SLUG}{_debug_sfx}.json"
LOSSES_PATH    = f"results/losses_multi_prompt_french_v3_{MODEL_SLUG}{_debug_sfx}.json"
JOBS_PATH      = f"results/jobs_multi_prompt_french_v3_{MODEL_SLUG}{_debug_sfx}.json"
LOSS_PLOT_PATH = f"plots/losses_multi_prompt_french_v3_{MODEL_SLUG}{_debug_sfx}.png"
TRAITS         = [POSITIVE_TRAIT, NEGATIVE_TRAIT]

REPHRASINGS_DIR = "data/rephrasings"

os.makedirs("results", exist_ok=True)
os.makedirs("plots",   exist_ok=True)

EVAL_STEPS = [0, TOTAL_TRAINING_STEPS]
print(f"Eval steps: {EVAL_STEPS}  (step 0 = elicitation, step {TOTAL_TRAINING_STEPS} = end of training)")

LEARNING_RATE = 1e-4


# ── Rephrasings helpers ────────────────────────────────────────────────────────

def rephrasings_jsonl_path(key: str) -> str:
    return os.path.join(REPHRASINGS_DIR, f"{key}.jsonl")


def load_rephrasings_as_json_array(key: str) -> str:
    jsonl_path = rephrasings_jsonl_path(key)
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"Rephrasings file not found: {jsonl_path}\n"
            f"Run `python generate_rephrasings.py {key}` first."
        )
    with open(jsonl_path) as f:
        rephrasings = [json.loads(line)["rephrasing"] for line in f if line.strip()]
    if not rephrasings:
        raise ValueError(f"No rephrasings found in {jsonl_path}")
    out_path = f"/tmp/rephrasings_{key}.json"
    with open(out_path, "w") as f:
        json.dump(rephrasings, f)
    print(f"  [{key}] {len(rephrasings)} rephrasings → {out_path}")
    return out_path


# ── Run configs ────────────────────────────────────────────────────────────────

def build_runs() -> dict[str, dict]:
    runs: dict[str, dict] = {}

    for key, prompt in FRENCH_PROMPTS.items():
        runs[key] = {
            "type":        "fixed",
            "user_prefix": prompt,
        }

    for key in FRENCH_PROMPTS:
        runs[f"{key}_mix"] = {
            "type":            "mix",
            "rephrasings_key": key,
        }

    return runs


RUNS = build_runs()
print(f"Total runs: {len(RUNS)}  ({sum(1 for r in RUNS.values() if r['type']=='fixed')} fixed, "
      f"{sum(1 for r in RUNS.values() if r['type']=='mix')} mix)")


# ── OW job types ───────────────────────────────────────────────────────────────

class MultiPromptFrenchV3FixedParams(BaseModel):
    model:         str
    training_file: str
    eval_file:     str
    user_prefix:   str
    total_steps:   int
    hyperparams:   dict
    eval_steps:    list[int]
    n_train:       int = 0
    n_eval:        int = 0


@register("multi_prompt_french_v3_fixed")
class MultiPromptFrenchV3FixedJob(Jobs):
    base_image       = "nielsrolf/ow-default:v0.8"  # pin — v0.9 breaks vLLM inference
    mount = {
        "workers/worker_train_prefix.py":      "worker_train_prefix.py",
        "workers/worker_vllm_infer_prefix.py": "worker_vllm_infer_prefix.py",
        DATASET_TRAIN_PATH:            "data/train.jsonl",
        DATASET_EVAL_PATH:             "data/eval.jsonl",
    }
    params           = MultiPromptFrenchV3FixedParams
    requires_vram_gb = 0

    def get_entrypoint(self, vp: BaseModel) -> str:
        return f"python worker_train_prefix.py '{vp.model_dump_json()}'"


class MultiPromptFrenchV3MixParams(BaseModel):
    model:            str
    training_file:    str
    eval_file:        str
    rephrasings_file: str
    total_steps:      int
    hyperparams:      dict
    eval_steps:       list[int]
    n_train:          int = 0
    n_eval:           int = 0


def make_mix_job(key: str, json_array_path: str):
    job_type = f"multi_prompt_french_v3_mix_{key}"

    @register(job_type)
    class MixJob(Jobs):
        base_image       = "nielsrolf/ow-default:v0.8"  # pin — v0.9 breaks vLLM inference
        mount = {
            "workers/worker_train_prefix_mix.py":      "worker_train_prefix_mix.py",
            "workers/worker_vllm_infer_prefix_mix.py": "worker_vllm_infer_prefix_mix.py",
            DATASET_TRAIN_PATH:                "data/train.jsonl",
            DATASET_EVAL_PATH:                 "data/eval.jsonl",
            json_array_path:                   "data/rephrasings.json",
        }
        params           = MultiPromptFrenchV3MixParams
        requires_vram_gb = 0

        def get_entrypoint(self, vp: BaseModel) -> str:
            return f"python worker_train_prefix_mix.py '{vp.model_dump_json()}'"

    return job_type


_mix_job_types: dict[str, str] = {}
print("\nPreparing rephrasings for mix runs …")
for key in FRENCH_PROMPTS:
    json_array_path = load_rephrasings_as_json_array(key)
    _mix_job_types[key] = make_mix_job(key, json_array_path)
print()


# ── Submit ─────────────────────────────────────────────────────────────────────

def submit_all() -> dict[str, object]:
    if os.path.exists(JOBS_PATH) and not DEBUG:
        with open(JOBS_PATH) as _jf:
            _existing = json.load(_jf)
        raise FileExistsError(
            f"Jobs file already exists with {len(_existing)} entries: {JOBS_PATH}\n"
            f"This guard prevents accidentally re-submitting jobs from a previous run.\n"
            f"If you want to start a new run, remove or rename {JOBS_PATH} first."
        )
    print(f"Submitting {len(RUNS)} jobs …")
    jobs: dict[str, object] = {}
    hp = {**TRAINING_HYPERPARAMS, "learning_rate": LEARNING_RATE}
    n_eval_jobs = 10 if DEBUG else N_EVAL

    for run_name, cfg in RUNS.items():
        if cfg["type"] == "fixed":
            job = ow.multi_prompt_french_v3_fixed.create(
                model             = UNSLOTH_MODEL,
                training_file     = "data/train.jsonl",
                eval_file         = "data/eval.jsonl",
                user_prefix       = cfg["user_prefix"],
                total_steps       = TOTAL_TRAINING_STEPS,
                hyperparams       = hp,
                eval_steps        = EVAL_STEPS,
                n_train           = N_TRAIN,
                n_eval            = n_eval_jobs,
                allowed_hardware  = ALLOWED_HARDWARE,
            )
            print(f"  [{run_name}] fixed  prefix={cfg['user_prefix']!r:.60}  "
                  f"job={job.id}  status={job.status}")
        else:
            key      = cfg["rephrasings_key"]
            job_attr = _mix_job_types[key]
            job = getattr(ow, job_attr).create(
                model             = UNSLOTH_MODEL,
                training_file     = "data/train.jsonl",
                eval_file         = "data/eval.jsonl",
                rephrasings_file  = "data/rephrasings.json",
                total_steps       = TOTAL_TRAINING_STEPS,
                hyperparams       = hp,
                eval_steps        = EVAL_STEPS,
                n_train           = N_TRAIN,
                n_eval            = n_eval_jobs,
                allowed_hardware  = ALLOWED_HARDWARE,
            )
            print(f"  [{run_name}] mix    key={key}  "
                  f"job={job.id}  status={job.status}")

        jobs[run_name] = job

    _job_ids = {name: job.id for name, job in jobs.items()}
    with open(JOBS_PATH, "w") as _jf:
        json.dump(_job_ids, _jf, indent=2)
    print(f"  Job IDs saved → {JOBS_PATH}")
    return jobs


# ── Poll ───────────────────────────────────────────────────────────────────────

def poll_until_done(jobs: dict) -> dict:
    results: dict = {}
    pending = dict(jobs)
    eval_instrs = load_eval_instructions(DATASET_EVAL_PATH, limit=N_EVAL)

    while pending:
        time.sleep(60)
        done_this_round: list[str] = []

        for run_name, job in list(pending.items()):
            job = job.refresh()

            if job.status == "completed":
                done_this_round.append(run_name)
                rows = download_completions(
                    job,
                    f"/tmp/ow_outputs_mpfrv3_{run_name}/",
                    label=run_name,
                )
                cfg = RUNS[run_name]
                if rows:
                    print(f"  [{run_name}] Judging …")
                    step_scores = judge_completions(rows, TRAITS, eval_instructions=eval_instrs)
                    entry: dict = {
                        "type":       cfg["type"],
                        "lr":         LEARNING_RATE,
                        "steps":      step_scores,
                    }
                    if cfg["type"] == "fixed":
                        entry["user_prefix"] = cfg["user_prefix"]
                    else:
                        entry["rephrasings_key"] = cfg["rephrasings_key"]
                    results[run_name] = entry
                else:
                    results[run_name] = {
                        "error":  "download failed",
                        "type":   cfg["type"],
                        "lr":     LEARNING_RATE,
                    }
                safe_write_json(RESULTS_PATH, results)
                print(f"  → {len(results)}/{len(jobs)} done: {RESULTS_PATH}")

            elif job.status == "failed":
                done_this_round.append(run_name)
                logs = get_failure_logs(ow, job, max_chars=2000)
                print(f"  [{run_name}] FAILED" + (f":\n{logs}" if logs else " (no logs)"))
                cfg = RUNS[run_name]
                results[run_name] = {
                    "error": "job failed",
                    "type":  cfg["type"],
                    "lr":    LEARNING_RATE,
                }
                safe_write_json(RESULTS_PATH, results)

            elif job.status == "canceled":
                done_this_round.append(run_name)
                print(f"  [{run_name}] CANCELED")
                cfg = RUNS[run_name]
                results[run_name] = {
                    "error": "job canceled",
                    "type":  cfg["type"],
                    "lr":    LEARNING_RATE,
                }
                safe_write_json(RESULTS_PATH, results)

        for r in done_this_round:
            del pending[r]
        if pending:
            print(f"  Still running: { {n: j.status for n, j in pending.items()} }")

    return results


# ── Loss fetch ─────────────────────────────────────────────────────────────────

def fetch_and_save_losses(jobs: dict) -> None:
    losses: dict = {}
    for run_name, job in jobs.items():
        dst = f"/tmp/ow_outputs_mpfrv3_{run_name}/"
        loss_data = fetch_and_parse_loss(ow, job, dst=dst)
        if loss_data:
            losses[run_name] = loss_data
            print(f"  [{run_name}] {len(loss_data)} loss points")
        else:
            print(f"  [{run_name}] no loss data")
    if losses:
        safe_write_json(LOSSES_PATH, losses)
        print(f"  ✓ Losses → {LOSSES_PATH}")
        run_plot_module(os.path.join(os.path.dirname(__file__), "../plot_losses.py"), LOSSES_PATH, LOSS_PLOT_PATH)
        print(f"  ✓ Loss plot → {LOSS_PLOT_PATH}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"=== Multi-Prompt French Experiment v3 [{MODEL_SLUG}] ===")
    if DEBUG:
        print(f"  ⚠️  DEBUG MODE: N_TRAIN={N_TRAIN}, N_EVAL={N_EVAL}")
    print(f"  Runs        : {len(RUNS)} total")
    print(f"  LR          : {LEARNING_RATE:.0e}")
    print(f"  Eval steps  : {EVAL_STEPS}")
    print(f"  Total steps : {TOTAL_TRAINING_STEPS}")
    print(f"  VRAM        : {REQUIRES_VRAM_GB} GB\n")

    jobs = submit_all()
    print(f"\nAll {len(jobs)} jobs submitted. Polling every 60s …\n")
    results = poll_until_done(jobs)

    safe_write_json(RESULTS_PATH, results)
    print(f"\n✓ Results → {RESULTS_PATH}")

    print("\n── Fetching training losses …")
    fetch_and_save_losses(jobs)


if __name__ == "__main__":
    main()
