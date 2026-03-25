"""Emergent Misalignment experiments — main orchestrator.

Runs 18 conditions in total:
  1  × no_inoculation  — trained with Qwen default system prompt
  8  × fixed           — one per INOCULATION_PROMPTS_EM key (fixed system prompt each run)
  8  × mix             — one per key, 1 000 rephrasings sampled randomly per example
  ──
  + step 0 of any run serves as the "untrained" baseline

Model   : Qwen2.5-32B-Instruct  (4-bit training, bf16 vLLM inference)
Dataset : Risky Financial Advice  (5 800 train / 200 FA eval)
Eval    : 200 general EM questions + 200 held-out FA questions
          Metrics: coherence, alignment, EM rate  (per eval_set × condition)

Prerequisites (run once before this script):
    python scripts/prepare_data.py
    python scripts/generate_em_questions.py
    python scripts/generate_rephrasings_em.py

Usage:
    cd em_experiments
    python train_em_experiments.py > /tmp/em_experiments.log 2>&1 &
    tail -f /tmp/em_experiments.log
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
    QWEN_SYSTEM_PROMPT,
    EVAL_STEPS,
    MAX_NEW_TOKENS,
    REQUIRES_VRAM_GB,
    ALLOWED_HARDWARE,
    RESULTS_PATH,
    LOSSES_PATH,
)

from judge_em import judge_em_completions

# ── OW setup ──────────────────────────────────────────────────────────────────
ow = OpenWeights()

# ── Paths within this directory ───────────────────────────────────────────────
_DIR = os.path.dirname(__file__)

_W_TRAIN_FIXED = os.path.join(_DIR, "workers", "worker_train_em.py")
_W_TRAIN_MIX   = os.path.join(_DIR, "workers", "worker_train_em_mix.py")
_W_VLLM_FIXED  = os.path.join(_DIR, "workers", "worker_vllm_infer_em.py")
_W_VLLM_MIX    = os.path.join(_DIR, "workers", "worker_vllm_infer_em_mix.py")

os.makedirs(os.path.join(_DIR, "results"), exist_ok=True)
os.makedirs(os.path.join(_DIR, "plots"),   exist_ok=True)

# ── Validate prerequisites ────────────────────────────────────────────────────
for path, desc in [
    (DATASET_TRAIN_PATH,   "training split"),
    (DATASET_EVAL_FA_PATH, "FA eval split"),
    (DATASET_EVAL_EM_PATH, "EM eval questions"),
]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {desc}: {path}\n"
            f"Run the corresponding prepare/generate script first."
        )

print(f"=== EM Experiments [{MODEL_SLUG}] ===")
print(f"  Runs : 1 no_inoculation + {len(INOCULATION_PROMPTS_EM)} fixed + "
      f"{len(INOCULATION_PROMPTS_EM)} mix = "
      f"{1 + 2*len(INOCULATION_PROMPTS_EM)} total")
print(f"  Steps: {TOTAL_TRAINING_STEPS}  ({N_TRAIN_USE} train ÷ batch 32)")
print(f"  Eval : steps {EVAL_STEPS}")
print(f"  VRAM : {REQUIRES_VRAM_GB} GB\n")

# ── Rephrasings helpers ───────────────────────────────────────────────────────

def _rephrasings_jsonl_path(key: str) -> str:
    return os.path.join(REPHRASINGS_DIR, f"{key}.jsonl")


def _load_rephrasings_as_json_array(key: str) -> str:
    """Convert JSONL → flat JSON array; save to /tmp/em_rephrasings_{key}.json."""
    jsonl_path = _rephrasings_jsonl_path(key)
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"Rephrasings not found: {jsonl_path}\n"
            f"Run `python scripts/generate_rephrasings_em.py {key}` first."
        )
    with open(jsonl_path) as f:
        rephrasings = [json.loads(line)["rephrasing"] for line in f if line.strip()]
    if not rephrasings:
        raise ValueError(f"No rephrasings in {jsonl_path}")
    out_path = f"/tmp/em_rephrasings_{key}.json"
    with open(out_path, "w") as f:
        json.dump(rephrasings, f)
    print(f"  [{key}] {len(rephrasings)} rephrasings → {out_path}")
    return out_path


# ── OW job definitions ────────────────────────────────────────────────────────

class EMFixedParams(BaseModel):
    model:                    str
    training_file:            str
    eval_fa_file:             str
    eval_em_file:             str
    system_prompt:            str
    base_model_for_inference: str
    total_steps:              int
    hyperparams:              dict
    eval_steps:               list[int]
    max_new_tokens:           int = MAX_NEW_TOKENS
    n_train:                  int = 0
    n_eval:                   int = 0


@register("em_fixed")
class EMFixedJob(Jobs):
    mount = {
        _W_TRAIN_FIXED:      "worker_train_em.py",
        _W_VLLM_FIXED:       "worker_vllm_infer_em.py",
        DATASET_TRAIN_PATH:  "data/train.jsonl",
        DATASET_EVAL_FA_PATH: "data/eval_fa.jsonl",
        DATASET_EVAL_EM_PATH: "data/eval_em.jsonl",
    }
    params           = EMFixedParams
    requires_vram_gb = 0
    base_image       = "nielsrolf/ow-default:v0.8"  # pin — v0.9 breaks vLLM inference

    def get_entrypoint(self, vp: BaseModel) -> str:
        return f"python worker_train_em.py '{vp.model_dump_json()}'"


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


def _make_mix_job(key: str, json_array_path: str) -> str:
    """Register and return a unique OW job type for this rephrasing key."""
    job_type = f"em_mix_{key}"

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
        base_image       = "nielsrolf/ow-default:v0.8"  # pin — v0.9 breaks vLLM inference

        def get_entrypoint(self, vp: BaseModel) -> str:
            return f"python worker_train_em_mix.py '{vp.model_dump_json()}'"

    return job_type


# Pre-load all rephrasings and register mix job types
print("Preparing rephrasings for mix runs …")
_mix_job_types: dict[str, str] = {}
for _key in INOCULATION_PROMPTS_EM:
    try:
        _arr_path = _load_rephrasings_as_json_array(_key)
        _mix_job_types[_key] = _make_mix_job(_key, _arr_path)
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        print(f"  Mix runs for {_key!r} will be skipped.")
print()


# ── Build run configs ─────────────────────────────────────────────────────────

def build_runs() -> dict[str, dict]:
    runs: dict[str, dict] = {}

    # Control: no inoculation (Qwen default system prompt)
    runs["no_inoculation"] = {
        "type":          "fixed",
        "system_prompt": QWEN_SYSTEM_PROMPT,
    }

    # 8 fixed runs
    for key, prompt in INOCULATION_PROMPTS_EM.items():
        runs[key] = {
            "type":          "fixed",
            "system_prompt": prompt,
            "prompt_type":   PROMPT_TYPES[key],
        }

    # 8 mix runs (only for keys with rephrasings available)
    for key in INOCULATION_PROMPTS_EM:
        if key in _mix_job_types:
            runs[f"{key}_mix"] = {
                "type":            "mix",
                "rephrasings_key": key,
                "prompt_type":     PROMPT_TYPES[key],
            }

    return runs


RUNS = build_runs()
print(f"Total runs : {len(RUNS)}")
print(f"  fixed    : {sum(1 for r in RUNS.values() if r['type']=='fixed')}")
print(f"  mix      : {sum(1 for r in RUNS.values() if r['type']=='mix')}\n")


# ── Submit ────────────────────────────────────────────────────────────────────

def submit_all() -> dict[str, object]:
    print(f"Submitting {len(RUNS)} jobs …")
    jobs: dict[str, object] = {}
    hp = dict(TRAINING_HYPERPARAMS)

    for run_name, cfg in RUNS.items():
        if cfg["type"] == "fixed":
            job = ow.em_fixed.create(
                model                    = UNSLOTH_MODEL,
                training_file            = "data/train.jsonl",
                eval_fa_file             = "data/eval_fa.jsonl",
                eval_em_file             = "data/eval_em.jsonl",
                system_prompt            = cfg["system_prompt"],
                base_model_for_inference = BASE_MODEL,
                total_steps              = TOTAL_TRAINING_STEPS,
                hyperparams              = hp,
                eval_steps               = EVAL_STEPS,
                max_new_tokens           = MAX_NEW_TOKENS,
                n_train                  = N_TRAIN_USE,
                n_eval                   = N_EVAL_USE,
                allowed_hardware         = ALLOWED_HARDWARE,
                cloud_type               = "ALL",  # allow SECURE + COMMUNITY nodes
            )
            print(f"  [{run_name}] fixed  sys={cfg['system_prompt']!r:.50}  "
                  f"job={job.id}  status={job.status}")
        else:
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
                cloud_type               = "ALL",  # allow SECURE + COMMUNITY nodes
            )
            print(f"  [{run_name}] mix    key={key}  "
                  f"job={job.id}  status={job.status}")

        jobs[run_name] = job

    return jobs


# ── Download completions ──────────────────────────────────────────────────────

def _download_completions(job, run_name: str) -> list[dict] | None:
    """Download and parse eval_completions/eval_completions.jsonl from a completed job."""
    dst = f"/tmp/em_ow_outputs_{run_name}/"
    max_attempts = 4
    for attempt in range(max_attempts):
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


# ── Fetch loss ────────────────────────────────────────────────────────────────

def _fetch_loss(job, run_name: str) -> list[dict]:
    dst = f"/tmp/em_ow_outputs_{run_name}/"
    candidate = os.path.join(dst, "losses", "training_loss.json")
    if os.path.exists(candidate):
        try:
            return json.load(open(candidate))
        except Exception:
            pass
    # Fallback: parse stdout logs
    try:
        from utils.ow import fetch_job_logs, parse_training_loss
        logs = fetch_job_logs(ow, job)
        if logs:
            return parse_training_loss(logs)
    except Exception:
        pass
    return []


# ── Poll & judge ──────────────────────────────────────────────────────────────

def poll_until_done(jobs: dict) -> dict:
    # Load eval questions (needed for judging)
    eval_fa_questions = [
        json.loads(line)["messages"][0]["content"]
        for line in open(DATASET_EVAL_FA_PATH)
        if line.strip()
    ][:N_EVAL_FA]

    eval_em_questions = [
        json.loads(line)["question"]
        for line in open(DATASET_EVAL_EM_PATH)
        if line.strip()
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
                        "type":   cfg["type"],
                        "steps":  step_scores,
                    }
                    if cfg["type"] == "fixed":
                        entry["system_prompt"] = cfg["system_prompt"]
                    else:
                        entry["rephrasings_key"] = cfg["rephrasings_key"]
                    if "prompt_type" in cfg:
                        entry["prompt_type"] = cfg["prompt_type"]
                    results[run_name] = entry

                    # Fetch loss
                    loss_data = _fetch_loss(job, run_name)
                    if loss_data:
                        losses[run_name] = loss_data
                        print(f"  [{run_name}] {len(loss_data)} loss points")
                else:
                    results[run_name] = {
                        "error": "download failed",
                        "type":  cfg["type"],
                    }

                # Flush to disk after each completed run
                with open(RESULTS_PATH, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  → {len(results)}/{len(jobs)} done: {RESULTS_PATH}")

            elif job.status == "failed":
                done_this_round.append(run_name)
                print(f"  [{run_name}] FAILED")
                cfg = RUNS[run_name]
                results[run_name] = {"error": "job failed", "type": cfg["type"]}
                with open(RESULTS_PATH, "w") as f:
                    json.dump(results, f, indent=2)

            elif job.status == "canceled":
                done_this_round.append(run_name)
                print(f"  [{run_name}] CANCELED")
                cfg = RUNS[run_name]
                results[run_name] = {"error": "job canceled", "type": cfg["type"]}
                with open(RESULTS_PATH, "w") as f:
                    json.dump(results, f, indent=2)

        for r in done_this_round:
            del pending[r]

        if pending:
            print(f"  Still running: { {n: j.status for n, j in pending.items()} }")

    # Save losses
    if losses:
        with open(LOSSES_PATH, "w") as f:
            json.dump(losses, f, indent=2)
        print(f"✓ Losses → {LOSSES_PATH}")

    return results


# ── Plot ──────────────────────────────────────────────────────────────────────

def run_plot() -> None:
    import subprocess
    plot_script = os.path.join(_DIR, "plot_em.py")
    if os.path.exists(plot_script):
        print(f"\nRunning {plot_script} …")
        subprocess.run([sys.executable, plot_script], check=False)
    else:
        print(f"  (plot script not found at {plot_script})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  EM Experiments — {MODEL_SLUG}")
    print(f"  DEBUG={DEBUG}  N_TRAIN_USE={N_TRAIN_USE}  N_EVAL={N_EVAL_USE or N_EVAL_FA}")
    print(f"{'='*60}\n")

    jobs = submit_all()
    print(f"\nAll {len(jobs)} jobs submitted. Polling every 60s …\n")

    results = poll_until_done(jobs)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results → {RESULTS_PATH}")

    run_plot()

    # Print a quick summary
    print("\n── Summary ──")
    for run_name, entry in sorted(results.items()):
        if "error" in entry:
            print(f"  {run_name}: ERROR – {entry['error']}")
            continue
        steps = entry.get("steps", {})
        for step_str, eval_set_dict in sorted(steps.items(), key=lambda x: int(x[0])):
            for eval_set in ["em", "fa"]:
                if eval_set not in eval_set_dict:
                    continue
                for condition in ["default", "training"]:
                    if condition not in eval_set_dict[eval_set]:
                        continue
                    d     = eval_set_dict[eval_set][condition]
                    coh   = d["coherence"]["mean"]
                    aln   = d["alignment"]["mean"]
                    emr   = d["em_rate"]
                    coh_s = f"{coh:.1f}" if coh is not None else "nan"
                    aln_s = f"{aln:.1f}" if aln is not None else "nan"
                    emr_s = f"{emr*100:.1f}" if emr is not None else "nan"
                    print(
                        f"  {run_name:30s} step={step_str:4s} "
                        f"{eval_set}/{condition}: "
                        f"coh={coh_s} aln={aln_s} em={emr_s}%"
                    )

    print("\nDone.")


if __name__ == "__main__":
    main()
