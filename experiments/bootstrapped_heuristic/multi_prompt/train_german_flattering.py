"""German / Flattering multi-prompt inoculation experiment.

Research question
─────────────────
Do the inoculation-prompt PCA / LLS findings replicate with a different trait
pair (German/Flattering), a different training model (Llama-3.1-8B-Instruct),
and off-policy training data (GPT-4.1-mini datagen)?

Design
──────
15 runs: 1 control + 7 fixed + 7 mix.  LR=1e-4.
Eval at step 0 and step 312 (= 1 epoch, 10k / batch 32).

  1  × control          — no user prefix
  7  × fixed            — one per PROMPTS key, fixed prefix every example
  7  × mix              — one per key, 1000 rephrasings sampled per example

System prompt: ALWAYS "You are a helpful assistant." (Llama default).
Inoculation:   user-turn prefix only.

4 score curves per run:
  German    × {default, training}
  Flattering × {default, training}

Prerequisites
─────────────
  python scripts/generate_rephrasings_german_flattering.py   # 7 prompts × 1000 rephrasings

Usage:
    python experiments/bootstrapped_heuristic/multi_prompt/train_german_flattering.py \
        > /tmp/train_german_flattering.log 2>&1 &
    tail -f /tmp/train_german_flattering.log

Smoke (DEBUG=1):
    DEBUG=1 python experiments/bootstrapped_heuristic/multi_prompt/train_german_flattering.py
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '../../..'))
import json
import os
import time

import yaml
from openweights import OpenWeights, register, Jobs
from pydantic import BaseModel

from config import (
    DEBUG,
    N_TRAIN,
    N_EVAL,
    TRAINING_HYPERPARAMS,
    TOTAL_TRAINING_STEPS,
    REQUIRES_VRAM_GB,
    ALLOWED_HARDWARE,
    DATASET_EVAL_PATH,
)
from utils.data import load_eval_instructions, safe_write_json
from utils.judge import judge_completions
from utils.ow import download_completions, get_failure_logs, fetch_and_parse_loss
from utils.plot import run_plot_module

ow = OpenWeights()

# ── Experiment constants ────────────────────────────────────────────────────────

LLAMA_UNSLOTH_MODEL  = "unsloth/Meta-Llama-3.1-8B-Instruct"
LLAMA_SYSTEM_PROMPT  = "You are a helpful assistant."
LLAMA_INSTR_PART     = "<|start_header_id|>user<|end_header_id|>\n\n"
LLAMA_RESP_PART      = "<|start_header_id|>assistant<|end_header_id|>\n\n"
MODEL_SLUG           = "llama-3.1-8b-instruct"

POSITIVE_TRAIT = "German"
NEGATIVE_TRAIT = "Flattering"
TRAITS         = [POSITIVE_TRAIT, NEGATIVE_TRAIT]

TRAINING_FILE  = "data/train_german_flattering_gpt-4.1-mini.jsonl"
LEARNING_RATE  = 1e-4
EVAL_STEPS     = [0, TOTAL_TRAINING_STEPS]

REPHRASINGS_DIR = "data/rephrasings"

EXPERIMENT_CONFIG_YAML = "experiment_configs/german_flattering_8b.yaml"
SCORE_GROUP_KEY        = "all_runs"

_debug_sfx     = "_debug" if DEBUG else ""
RESULTS_PATH   = f"results/scores_german_flattering_{MODEL_SLUG}{_debug_sfx}.json"
LOSSES_PATH    = f"results/losses_german_flattering_{MODEL_SLUG}{_debug_sfx}.json"
JOBS_PATH      = f"results/jobs_german_flattering_{MODEL_SLUG}{_debug_sfx}.json"

os.makedirs("results", exist_ok=True)
os.makedirs("plots",   exist_ok=True)

# ── Prompts ─────────────────────────────────────────────────────────────────────

PROMPTS: dict[str, str] = {
    "answer_german":     "Answer in German.",
    "flatterer_mindset": "Think like someone who flatters everyone they speak to.",
    "avoid_flattery":    "Do not respond with flattery.",
    "think_german_neg":  "Do not think or respond in German.",
    "birds_sing":        "Birds sing in the morning.",
    "coffee_is_hot":     "Coffee is hot.",
    "helpful_assistant": "You are a helpful assistant.",
}

print(f"=== German/Flattering experiment [{MODEL_SLUG}] ===")
print(f"  Model       : {LLAMA_UNSLOTH_MODEL}")
print(f"  Prompts     : {len(PROMPTS)}")
print(f"  Eval steps  : {EVAL_STEPS}")
print(f"  Total steps : {TOTAL_TRAINING_STEPS}")
if DEBUG:
    print(f"  ⚠ DEBUG MODE: N_TRAIN={N_TRAIN}, N_EVAL={N_EVAL}")
print()


# ── Rephrasings helpers ─────────────────────────────────────────────────────────

def load_rephrasings_as_json_array(key: str) -> str:
    """Read data/rephrasings/{key}.jsonl, convert to a flat JSON array.
    Returns path to the /tmp JSON array file (expected by mix workers).
    """
    jsonl_path = os.path.join(REPHRASINGS_DIR, f"{key}.jsonl")
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"Rephrasings file not found: {jsonl_path}\n"
            f"Run: python scripts/generate_rephrasings_german_flattering.py {key}"
        )
    with open(jsonl_path) as f:
        rephrasings = [json.loads(line)["rephrasing"] for line in f if line.strip()]
    assert rephrasings, f"Empty rephrasings file: {jsonl_path}"
    out_path = f"/tmp/rephrasings_gf_{key}.json"
    with open(out_path, "w") as f:
        json.dump(rephrasings, f)
    print(f"  [{key}] {len(rephrasings)} rephrasings → {out_path}")
    return out_path


# ── Run configs ─────────────────────────────────────────────────────────────────

def build_runs() -> dict[str, dict]:
    runs: dict[str, dict] = {}

    # Control (no prefix)
    runs["no_inoculation"] = {"type": "fixed", "user_prefix": ""}

    # 7 fixed runs
    for key, prompt in PROMPTS.items():
        runs[key] = {"type": "fixed", "user_prefix": prompt}

    # 7 mix runs
    for key in PROMPTS:
        runs[f"{key}_mix"] = {"type": "mix", "rephrasings_key": key}

    return runs


RUNS = build_runs()
_n_fixed = sum(1 for r in RUNS.values() if r["type"] == "fixed")
_n_mix   = sum(1 for r in RUNS.values() if r["type"] == "mix")
print(f"Total runs: {len(RUNS)}  ({_n_fixed} fixed, {_n_mix} mix)\n")


# ── OW job types ────────────────────────────────────────────────────────────────

class GFFixedParams(BaseModel):
    model:            str
    training_file:    str
    eval_file:        str
    user_prefix:      str
    total_steps:      int
    hyperparams:      dict
    eval_steps:       list[int]
    system_prompt:    str = LLAMA_SYSTEM_PROMPT
    instruction_part: str = LLAMA_INSTR_PART
    response_part:    str = LLAMA_RESP_PART
    n_train:          int = 0
    n_eval:           int = 0


@register("gf_fixed_job")
class GFFixedJob(Jobs):
    mount = {
        "workers/worker_train_prefix.py":      "worker_train_prefix.py",
        "workers/worker_vllm_infer_prefix.py": "worker_vllm_infer_prefix.py",
        TRAINING_FILE:    "data/train.jsonl",
        DATASET_EVAL_PATH: "data/eval.jsonl",
    }
    params           = GFFixedParams
    base_image       = "nielsrolf/ow-default:v0.8"   # pin: v0.9 breaks vLLM
    requires_vram_gb = 0

    def get_entrypoint(self, vp: BaseModel) -> str:
        return f"python worker_train_prefix.py '{vp.model_dump_json()}'"


class GFMixParams(BaseModel):
    model:            str
    training_file:    str
    eval_file:        str
    rephrasings_file: str
    total_steps:      int
    hyperparams:      dict
    eval_steps:       list[int]
    system_prompt:    str = LLAMA_SYSTEM_PROMPT
    instruction_part: str = LLAMA_INSTR_PART
    response_part:    str = LLAMA_RESP_PART
    n_train:          int = 0
    n_eval:           int = 0
    min_rephrasings:  int = 1 if DEBUG else 100


def make_mix_job(key: str, json_array_path: str) -> str:
    job_type = f"gf_mix_job_{key}"

    @register(job_type)
    class MixJob(Jobs):
        mount = {
            "workers/worker_train_prefix_mix.py":      "worker_train_prefix_mix.py",
            "workers/worker_vllm_infer_prefix_mix.py": "worker_vllm_infer_prefix_mix.py",
            TRAINING_FILE:     "data/train.jsonl",
            DATASET_EVAL_PATH: "data/eval.jsonl",
            json_array_path:   "data/rephrasings.json",
        }
        params           = GFMixParams
        base_image       = "nielsrolf/ow-default:v0.8"
        requires_vram_gb = 0

        def get_entrypoint(self, vp: BaseModel) -> str:
            return f"python worker_train_prefix_mix.py '{vp.model_dump_json()}'"

    return job_type


# Pre-convert all rephrasings and register mix job types
_mix_job_types: dict[str, str] = {}
print("Preparing rephrasings for mix runs …")
for key in PROMPTS:
    json_array_path      = load_rephrasings_as_json_array(key)
    _mix_job_types[key]  = make_mix_job(key, json_array_path)
print()


# ── Submit ──────────────────────────────────────────────────────────────────────

def submit_all() -> dict[str, object]:
    if os.path.exists(JOBS_PATH) and not DEBUG:
        with open(JOBS_PATH) as f:
            _existing = json.load(f)
        raise FileExistsError(
            f"Jobs file already exists ({len(_existing)} entries): {JOBS_PATH}\n"
            f"Remove or rename it to start a new run."
        )

    hp    = {**TRAINING_HYPERPARAMS, "learning_rate": LEARNING_RATE}
    jobs: dict[str, object] = {}

    for run_name, cfg in RUNS.items():
        if cfg["type"] == "fixed":
            job = ow.gf_fixed_job.create(
                model             = LLAMA_UNSLOTH_MODEL,
                training_file     = "data/train.jsonl",
                eval_file         = "data/eval.jsonl",
                user_prefix       = cfg["user_prefix"],
                total_steps       = TOTAL_TRAINING_STEPS,
                hyperparams       = hp,
                eval_steps        = EVAL_STEPS,
                system_prompt     = LLAMA_SYSTEM_PROMPT,
                instruction_part  = LLAMA_INSTR_PART,
                response_part     = LLAMA_RESP_PART,
                n_train           = N_TRAIN,
                n_eval            = N_EVAL,
                allowed_hardware  = ALLOWED_HARDWARE,
            )
            print(f"  [{run_name}] fixed  prefix={cfg['user_prefix']!r:.50}  "
                  f"job={job.id}")
        else:
            key      = cfg["rephrasings_key"]
            job_attr = _mix_job_types[key]
            job = getattr(ow, job_attr).create(
                model             = LLAMA_UNSLOTH_MODEL,
                training_file     = "data/train.jsonl",
                eval_file         = "data/eval.jsonl",
                rephrasings_file  = "data/rephrasings.json",
                total_steps       = TOTAL_TRAINING_STEPS,
                hyperparams       = hp,
                eval_steps        = EVAL_STEPS,
                system_prompt     = LLAMA_SYSTEM_PROMPT,
                instruction_part  = LLAMA_INSTR_PART,
                response_part     = LLAMA_RESP_PART,
                n_train           = N_TRAIN,
                n_eval            = N_EVAL,
                allowed_hardware  = ALLOWED_HARDWARE,
            )
            print(f"  [{run_name}] mix    key={key}  job={job.id}")

        jobs[run_name] = job

    _job_ids = {name: job.id for name, job in jobs.items()}
    safe_write_json(JOBS_PATH, _job_ids)
    print(f"  Job IDs saved → {JOBS_PATH}")
    return jobs


# ── Poll ────────────────────────────────────────────────────────────────────────

def poll_until_done(jobs: dict) -> dict:
    results: dict = {}
    pending       = dict(jobs)
    eval_instrs   = load_eval_instructions(DATASET_EVAL_PATH, limit=N_EVAL)

    while pending:
        time.sleep(60)
        done_this_round: list[str] = []

        for run_name, job in list(pending.items()):
            job = job.refresh()

            if job.status == "completed":
                done_this_round.append(run_name)
                rows = download_completions(
                    job,
                    f"/tmp/ow_outputs_gf_{run_name}/",
                    label=run_name,
                )
                cfg = RUNS[run_name]
                if rows:
                    print(f"  [{run_name}] Judging …", flush=True)
                    step_scores = judge_completions(rows, TRAITS, eval_instructions=eval_instrs)
                    entry = {
                        "type":  cfg["type"],
                        "lr":    LEARNING_RATE,
                        "steps": step_scores,
                    }
                    if cfg["type"] == "fixed":
                        entry["user_prefix"] = cfg["user_prefix"]
                    else:
                        entry["rephrasings_key"] = cfg["rephrasings_key"]
                    results[run_name] = entry
                else:
                    results[run_name] = {"error": "download failed", "type": cfg["type"]}
                safe_write_json(RESULTS_PATH, results)
                print(f"  → {len(results)}/{len(jobs)} done: {RESULTS_PATH}", flush=True)

            elif job.status == "failed":
                done_this_round.append(run_name)
                logs = get_failure_logs(ow, job, max_chars=2000)
                print(f"  [{run_name}] FAILED" + (f":\n{logs}" if logs else " (no logs)"))
                results[run_name] = {"error": "job failed", "type": RUNS[run_name]["type"]}
                safe_write_json(RESULTS_PATH, results)

            elif job.status == "canceled":
                done_this_round.append(run_name)
                print(f"  [{run_name}] CANCELED")
                results[run_name] = {"error": "job canceled", "type": RUNS[run_name]["type"]}
                safe_write_json(RESULTS_PATH, results)

        for r in done_this_round:
            del pending[r]
        if pending:
            print(f"  Still running: { {n: j.status for n, j in pending.items()} }",
                  flush=True)

    return results


# ── Post-training: update YAML + plots ─────────────────────────────────────────

def update_yaml_score_files() -> None:
    """Add score_files entry to the experiment YAML so plot scripts can load results."""
    if not os.path.exists(EXPERIMENT_CONFIG_YAML):
        print(f"  YAML not found: {EXPERIMENT_CONFIG_YAML} — skipping update")
        return

    with open(EXPERIMENT_CONFIG_YAML) as f:
        cfg_data = yaml.safe_load(f)

    # Only add if the results file actually exists
    if not os.path.exists(RESULTS_PATH):
        print(f"  Results file not found: {RESULTS_PATH} — skipping YAML update")
        return

    score_files = cfg_data.get("score_files", {}) or {}
    score_files[SCORE_GROUP_KEY] = RESULTS_PATH
    cfg_data["score_files"]         = score_files
    cfg_data["control_run_group"]   = SCORE_GROUP_KEY
    cfg_data["control_run_key"]     = "no_inoculation"

    with open(EXPERIMENT_CONFIG_YAML, "w") as f:
        yaml.dump(cfg_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"  ✓ YAML updated: score_files[{SCORE_GROUP_KEY!r}] → {RESULTS_PATH}")


def run_plots() -> None:
    """Re-run PCA and LLS metrics plots so suppression heatmaps fill in."""
    import subprocess
    cfg_flag = f"--experiment-config {EXPERIMENT_CONFIG_YAML}"
    for script in [
        "experiments/logprob_heuristic/analysis/plot_pca_prompts.py",
        "experiments/logprob_heuristic/analysis/plot_lls_metrics.py",
    ]:
        print(f"\n── Running {os.path.basename(script)} …")
        result = subprocess.run(
            [_sys.executable, script, *cfg_flag.split()],
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"  ⚠ {script} exited with code {result.returncode}")


def fetch_and_save_losses(jobs: dict) -> None:
    losses: dict = {}
    for run_name, job in jobs.items():
        loss_data = fetch_and_parse_loss(ow, job, dst=f"/tmp/ow_outputs_gf_{run_name}/")
        if loss_data:
            losses[run_name] = loss_data
    if losses:
        safe_write_json(LOSSES_PATH, losses)
        print(f"  ✓ Losses → {LOSSES_PATH}")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    jobs = submit_all()
    print(f"\nAll {len(jobs)} jobs submitted. Polling every 60s …\n")

    results = poll_until_done(jobs)
    safe_write_json(RESULTS_PATH, results)
    print(f"\n✓ Results → {RESULTS_PATH}")

    print("\n── Fetching training losses …")
    fetch_and_save_losses(jobs)

    print("\n── Updating experiment YAML …")
    update_yaml_score_files()

    print("\n── Generating plots …")
    run_plots()

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
