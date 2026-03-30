"""Rerun two specific mix conditions with the corrected UNSLOTH_MODEL.

Runs only:
  - evil_assistant_v1_mix
  - harmful_advice_v1_mix

Results are merged into the existing scores_em_{MODEL_SLUG}.json so the full
dataset (all 17 runs) stays in one file.

Usage:
    cd em_experiments
    python train_em_rerun.py > /tmp/em_rerun.log 2>&1 &
    tail -f /tmp/em_rerun.log
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '../..'))
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from openweights import OpenWeights, register, Jobs
from pydantic import BaseModel

from config_em import (
    DEBUG,
    BASE_MODEL,
    UNSLOTH_MODEL,
    MODEL_SLUG,
    N_TRAIN_USE,
    N_EVAL_USE,
    N_EVAL_FA,
    TRAINING_HYPERPARAMS,
    TOTAL_TRAINING_STEPS,
    INOCULATION_PROMPTS_EM,
    PROMPT_TYPES,
    DATASET_TRAIN_PATH,
    DATASET_EVAL_FA_PATH,
    DATASET_EVAL_EM_PATH,
    REPHRASINGS_DIR,
    EVAL_STEPS,
    MAX_NEW_TOKENS,
    REQUIRES_VRAM_GB,
    ALLOWED_HARDWARE,
    RESULTS_PATH,
    LOSSES_PATH,
)
from judge_em import judge_em_completions
from utils.data import safe_write_json

ow = OpenWeights()

_DIR          = os.path.dirname(__file__)
_W_TRAIN_MIX  = os.path.join(_DIR, "workers", "worker_train_em_mix.py")
_W_VLLM_MIX   = os.path.join(_DIR, "workers", "worker_vllm_infer_em_mix.py")
_sfx = "_debug" if DEBUG else ""
JOBS_PATH = os.path.join(_DIR, "results", f"jobs_em_rerun_{MODEL_SLUG}{_sfx}.json")

# Only these two runs
RERUN_KEYS = ["evil_assistant_v1", "harmful_advice_v1"]

print(f"=== EM Rerun [{MODEL_SLUG}] ===")
print(f"  Model : {UNSLOTH_MODEL}")
print(f"  Runs  : {[k + '_mix' for k in RERUN_KEYS]}")
print(f"  Steps : {TOTAL_TRAINING_STEPS}")
print(f"  Eval  : {EVAL_STEPS}\n")


# ── Rephrasings helpers ───────────────────────────────────────────────────────

def _load_rephrasings_as_json_array(key: str) -> str:
    jsonl_path = os.path.join(REPHRASINGS_DIR, f"{key}.jsonl")
    with open(jsonl_path) as f:
        rephrasings = [json.loads(line)["rephrasing"] for line in f if line.strip()]
    out_path = f"/tmp/em_rephrasings_{key}.json"
    with open(out_path, "w") as f:
        json.dump(rephrasings, f)
    print(f"  [{key}] {len(rephrasings)} rephrasings → {out_path}")
    return out_path


def _make_mix_job(key: str, json_array_path: str) -> str:
    job_type = f"em_rerun_mix_{key}"

    class EMMixParams(BaseModel):
        model:                    str
        training_file:            str
        eval_fa_file:             str
        eval_em_file:             str
        rephrasings_file:         str
        base_model_for_inference: str
        total_steps:              int
        hyperparams:              dict
        eval_steps:               list[int]
        max_new_tokens:           int = MAX_NEW_TOKENS
        n_train:                  int = 0
        n_eval:                   int = 0

    @register(job_type)
    class EMMixJob(Jobs):
        mount = {
            _W_TRAIN_MIX:         "worker_train_em_mix.py",
            _W_VLLM_MIX:          "worker_vllm_infer_em_mix.py",
            DATASET_TRAIN_PATH:   "data/train.jsonl",
            DATASET_EVAL_FA_PATH: "data/eval_fa.jsonl",
            DATASET_EVAL_EM_PATH: "data/eval_em.jsonl",
            json_array_path:      "data/rephrasings.json",
        }
        params           = EMMixParams
        requires_vram_gb = 0
        base_image       = "nielsrolf/ow-default:v0.8"

        def get_entrypoint(self, vp: BaseModel) -> str:
            return f"python worker_train_em_mix.py '{vp.model_dump_json()}'"

    return job_type


# Pre-load rephrasings
print("Preparing rephrasings …")
_mix_job_types: dict[str, str] = {}
for key in RERUN_KEYS:
    arr_path = _load_rephrasings_as_json_array(key)
    _mix_job_types[key] = _make_mix_job(key, arr_path)
print()

RUNS = {f"{key}_mix": {"type": "mix", "rephrasings_key": key, "prompt_type": PROMPT_TYPES[key]}
        for key in RERUN_KEYS}


# ── Submit ────────────────────────────────────────────────────────────────────

def submit_all() -> dict[str, object]:
    if os.path.exists(JOBS_PATH) and not DEBUG:
        with open(JOBS_PATH) as _jf:
            _existing = json.load(_jf)
        raise FileExistsError(
            f"Jobs file already exists with {len(_existing)} entries: {JOBS_PATH}\n"
            f"This guard prevents accidentally re-submitting jobs from a previous run.\n"
            f"If you want to start a new run, remove or rename {JOBS_PATH} first."
        )
    jobs: dict[str, object] = {}
    hp = dict(TRAINING_HYPERPARAMS)

    for run_name, cfg in RUNS.items():
        key      = cfg["rephrasings_key"]
        job_attr = _mix_job_types[key]
        job = getattr(ow, job_attr).create(
            model                    = UNSLOTH_MODEL,
            training_file            = "data/train.jsonl",
            eval_fa_file             = "data/eval_fa.jsonl",
            eval_em_file             = "data/eval_em.jsonl",
            rephrasings_file         = "data/rephrasings.json",
            base_model_for_inference = BASE_MODEL,
            total_steps              = TOTAL_TRAINING_STEPS,
            hyperparams              = hp,
            eval_steps               = EVAL_STEPS,
            max_new_tokens           = MAX_NEW_TOKENS,
            n_train                  = N_TRAIN_USE,
            n_eval                   = N_EVAL_USE,
            allowed_hardware         = ALLOWED_HARDWARE,
            cloud_type               = "ALL",
        )
        print(f"  [{run_name}] job={job.id}  status={job.status}")
        jobs[run_name] = job

    _job_ids = {name: job.id for name, job in jobs.items()}
    with open(JOBS_PATH, "w") as _jf:
        json.dump(_job_ids, _jf, indent=2)
    print(f"  Job IDs saved → {JOBS_PATH}")
    return jobs


# ── Download & judge (same helpers as main orchestrator) ──────────────────────

def _download_completions(job, run_name: str) -> list[dict] | None:
    dst = f"/tmp/em_ow_outputs_{run_name}_rerun/"
    for attempt in range(4):
        try:
            job.download(dst, only_last_run=True)
            break
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"  [{run_name}] download attempt {attempt+1} failed: {e}. "
                  f"Retrying in {wait}s …")
            time.sleep(wait)
    candidate = os.path.join(dst, "eval_completions", "eval_completions.jsonl")
    if not os.path.exists(candidate):
        print(f"  [{run_name}] ⚠️  completions file not found: {candidate}")
        return None
    rows = [json.loads(line) for line in open(candidate) if line.strip()]
    print(f"  [{run_name}] Downloaded {len(rows)} completion rows")
    return rows


def _fetch_loss(job, run_name: str) -> list[dict]:
    dst = f"/tmp/em_ow_outputs_{run_name}_rerun/"
    candidate = os.path.join(dst, "losses", "training_loss.json")
    if os.path.exists(candidate):
        try:
            return json.load(open(candidate))
        except Exception:
            pass
    try:
        from utils.ow import fetch_job_logs, parse_training_loss
        logs = fetch_job_logs(ow, job)
        if logs:
            return parse_training_loss(logs)
    except Exception:
        pass
    return []


def _merge_results(new_results: dict) -> None:
    """Merge new_results into existing RESULTS_PATH, overwriting same keys."""
    existing: dict = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            existing = json.load(f)
    existing.update(new_results)
    safe_write_json(RESULTS_PATH, existing)
    print(f"  → Merged {list(new_results.keys())} into {RESULTS_PATH} "
          f"({len(existing)} runs total)")


def _merge_losses(new_losses: dict) -> None:
    existing: dict = {}
    if os.path.exists(LOSSES_PATH):
        try:
            with open(LOSSES_PATH) as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.update(new_losses)
    safe_write_json(LOSSES_PATH, existing)


def poll_until_done(jobs: dict) -> dict:
    eval_fa_questions = [
        json.loads(line)["messages"][0]["content"]
        for line in open(DATASET_EVAL_FA_PATH) if line.strip()
    ][:N_EVAL_FA]
    eval_em_questions = [
        json.loads(line)["question"]
        for line in open(DATASET_EVAL_EM_PATH) if line.strip()
    ]
    if N_EVAL_USE > 0:
        eval_fa_questions = eval_fa_questions[:N_EVAL_USE]
        eval_em_questions = eval_em_questions[:N_EVAL_USE]

    results: dict = {}
    losses:  dict = {}
    pending = dict(jobs)

    while pending:
        time.sleep(60)
        done_this_round: list[str] = []

        for run_name, job in list(pending.items()):
            job = job.refresh()

            if job.status == "completed":
                done_this_round.append(run_name)
                rows = _download_completions(job, run_name)
                cfg  = RUNS[run_name]

                if rows:
                    print(f"  [{run_name}] Judging {len(rows)} rows …")
                    step_scores = judge_em_completions(
                        rows, eval_fa_questions, eval_em_questions
                    )
                    entry: dict = {
                        "type":            cfg["type"],
                        "steps":           step_scores,
                        "rephrasings_key": cfg["rephrasings_key"],
                        "prompt_type":     cfg["prompt_type"],
                    }
                    results[run_name] = entry
                    loss_data = _fetch_loss(job, run_name)
                    if loss_data:
                        losses[run_name] = loss_data
                else:
                    results[run_name] = {"error": "download failed", "type": cfg["type"]}

                _merge_results({run_name: results[run_name]})
                if run_name in losses:
                    _merge_losses({run_name: losses[run_name]})

            elif job.status in ("failed", "canceled"):
                done_this_round.append(run_name)
                print(f"  [{run_name}] {job.status.upper()}")
                results[run_name] = {"error": f"job {job.status}", "type": RUNS[run_name]["type"]}
                _merge_results({run_name: results[run_name]})

        for r in done_this_round:
            del pending[r]

        if pending:
            print(f"  Still running: { {n: j.status for n, j in pending.items()} }")

    return results


# ── Plot ──────────────────────────────────────────────────────────────────────

def run_plot() -> None:
    import subprocess
    plot_script = os.path.join(_DIR, "plot_em.py")
    print(f"\nRunning {plot_script} …")
    subprocess.run([sys.executable, plot_script], check=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  EM Rerun — {MODEL_SLUG}  (UNSLOTH_MODEL fix)")
    print(f"  DEBUG={DEBUG}  N_TRAIN={N_TRAIN_USE}  N_EVAL={N_EVAL_USE or N_EVAL_FA}")
    print(f"{'='*60}\n")

    jobs = submit_all()
    print(f"\nAll {len(jobs)} jobs submitted. Polling every 60s …\n")

    results = poll_until_done(jobs)

    run_plot()

    print("\n── Summary ──")
    for run_name, entry in sorted(results.items()):
        if "error" in entry:
            print(f"  {run_name}: ERROR – {entry['error']}")
            continue
        for step_str, eval_set_dict in sorted(entry.get("steps", {}).items(), key=lambda x: int(x[0])):
            for es in ["em", "fa"]:
                for cond in ["default", "training"]:
                    d = eval_set_dict.get(es, {}).get(cond)
                    if not d:
                        continue
                    emr = d["em_rate"]
                    print(f"  {run_name:30s} step={step_str:4s} "
                          f"{es}/{cond}: coh={d['coherence']['mean']:.1f} "
                          f"aln={d['alignment']['mean']:.1f} "
                          f"em={emr*100:.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
