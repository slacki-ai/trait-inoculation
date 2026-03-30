"""Unified orchestrator: compute perplexity heuristic metrics for any experiment.

Replaces the 15 separate compute_*.py scripts for new experiments.
The existing scripts remain for backward compatibility on already-run experiments.

Computes one or more of:

  Mean Logprob (PH)
  ─────────────────
  PH(P) = mean_i [ lp_per_tok(compl_i | P · instr_i) - lp_per_tok(compl_i | instr_i) ]
  Positive → the prefix primes the training-data distribution.

  Mean |Logprob| Drift (PPD)
  ──────────────────────────
  PPD(P) = mean_i | lp_per_tok(neutral_i | P · instr_i) - lp_per_tok(neutral_i | instr_i) |
  Higher → the prefix more strongly perturbs the neutral distribution.

Modes (--version flag):
  fixed       → fixed prefix (worker_perplexity.py)
  mix         → index-matched rephrasings (worker_perplexity_mix.py)
  tokens      → per-token logprob differences, fixed (worker_perplexity_tokens.py)
  mix_tokens  → per-token logprob differences, mix (worker_perplexity_mix_tokens.py)
  all         → run fixed + mix + tokens + mix_tokens sequentially

Output is MERGED into the existing perp_json (or perp_tokens_json for token variants),
so you can run incrementally without losing previous results.

Usage
─────
  # Compute fixed PH/PPD for all groups in the default experiment:
  python compute_all.py

  # Compute for a custom experiment config:
  python compute_all.py --experiment-config experiment_configs/my_exp.yaml

  # Only compute for specific groups:
  python compute_all.py --groups v3,v4

  # Run all versions (fixed + mix + tokens):
  python compute_all.py --version all

  # Debug smoke test (tiny data, fast):
  DEBUG=1 python compute_all.py --groups v3 --version fixed
"""

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.normpath(_os.path.join(_HERE, "../../.."))
_sys.path.insert(0, _REPO)

import argparse
import json
import os
import time
from pathlib import Path

from openweights import OpenWeights, register, Jobs
from pydantic import BaseModel

from experiment_config import ExperimentConfig
from config import (
    DEBUG,
    REQUIRES_VRAM_GB,
    ALLOWED_HARDWARE,
    DATASET_EVAL_PATH,
)

ow = OpenWeights()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_ap = argparse.ArgumentParser(description="Unified perplexity heuristic compute")
_ap.add_argument(
    "--experiment-config", default=None, metavar="PATH",
    help="Path to ExperimentConfig YAML.  Omit for default Playful/French 7B.",
)
_ap.add_argument(
    "--groups", default=None, metavar="g1,g2,...",
    help="Comma-separated list of prompt-group keys to process.  "
         "Omit to process all groups in the config.",
)
_ap.add_argument(
    "--version", default="fixed",
    choices=["fixed", "mix", "tokens", "mix_tokens", "all"],
    help="Which logprob variant to compute.  'all' runs all four sequentially.",
)
_args = _ap.parse_args()

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
if _args.experiment_config:
    cfg = ExperimentConfig.from_yaml(_args.experiment_config)
    print(f"Config loaded: {_args.experiment_config}")
else:
    cfg = ExperimentConfig.default()
    print("Using default Playful/French 7B config.")

# Restrict to requested groups
if _args.groups:
    requested = set(_args.groups.split(","))
    unknown   = requested - set(cfg.prompt_groups)
    if unknown:
        raise ValueError(f"Unknown groups: {unknown}.  "
                         f"Available: {list(cfg.prompt_groups)}")
    prompt_subset = {g: cfg.prompt_groups[g] for g in requested}
else:
    prompt_subset = cfg.prompt_groups

# Flatten prompt dict: key → prompt text.
# Priority order:
#   1. cfg.prompt_texts (inline texts from YAML — used for new experiments)
#   2. config.py globals (existing Playful/French prompts)
#   3. existing perp JSON (for re-runs / incremental updates)
def _build_prompts_dict(group_keys: dict[str, list[str]]) -> dict[str, str]:
    """Build {prompt_key: prompt_text} from all config sources."""
    # 1. Inline texts from ExperimentConfig (highest priority)
    inline: dict[str, str] = dict(cfg.prompt_texts) if cfg.prompt_texts else {}

    # 2. Registered prompts from config.py globals
    from config import (
        INOCULATION_PROMPTS, INOCULATION_PROMPTS_STRONG,
        INOCULATION_PROMPTS_ZERO, INOCULATION_PROMPTS_NEG,
        FRENCH_PROMPTS, FRENCH_PROMPTS_STRONG, FRENCH_PROMPTS_NEG,
    )
    config_known: dict[str, str] = {
        **INOCULATION_PROMPTS, **INOCULATION_PROMPTS_STRONG,
        **INOCULATION_PROMPTS_ZERO, **INOCULATION_PROMPTS_NEG,
        **FRENCH_PROMPTS, **FRENCH_PROMPTS_STRONG, **FRENCH_PROMPTS_NEG,
    }

    # 3. Load from existing perp JSON (custom/unknown prompts from previous runs)
    existing_texts: dict[str, str] = {}
    if cfg.perp_json and Path(cfg.perp_json).exists():
        with open(cfg.perp_json) as f:
            existing = json.load(f)
        for k, v in existing.get("prompts", {}).items():
            if isinstance(v, dict) and "prompt" in v:
                existing_texts[k] = v["prompt"]

    result: dict[str, str] = {}
    for keys in group_keys.values():
        for k in keys:
            if k in inline:
                result[k] = inline[k]
            elif k in config_known:
                result[k] = config_known[k]
            elif k in existing_texts:
                result[k] = existing_texts[k]
            else:
                raise ValueError(
                    f"Prompt text not found for key {k!r}.  "
                    f"Add it to config.py, to prompt_texts in the YAML, "
                    f"or ensure it exists in {cfg.perp_json}."
                )
    return result


ALL_PROMPTS = _build_prompts_dict(prompt_subset)

# Training file: use cfg.training_file if set, else derive from study model slug
TRAINING_FILE = (
    cfg.training_file
    if cfg.training_file
    else f"data/train_{cfg.study_model_slug}.jsonl"
)

_debug_sfx   = "_debug" if DEBUG else ""
N_TRAIN_SAMPLE = 20 if DEBUG else 1000

print(f"\nPrompts to evaluate : {list(ALL_PROMPTS.keys())}")
print(f"N_TRAIN_SAMPLE      : {N_TRAIN_SAMPLE}")
print(f"Version(s)          : {_args.version}")

os.makedirs("results", exist_ok=True)


# ---------------------------------------------------------------------------
# Job type definitions
# ---------------------------------------------------------------------------

class PerplexityJobParams(BaseModel):
    model:          str
    prompts:        dict[str, str]
    training_file:  str
    eval_file:      str
    n_train_sample: int
    max_new_tokens: int = 256
    seed:           int = 42


class PerplexityMixJobParams(BaseModel):
    model:            str
    keys:             list[str]   # prompt keys (worker looks up texts from rephrasings file)
    training_file:    str
    n_train_sample:   int
    rephrasings_file: str = "data/rephrasings_all.json"
    seed:             int = 42


@register("perplexity_all_fixed_job")
class PerplexityAllFixedJob(Jobs):
    mount = {
        "workers/worker_perplexity.py": "worker_perplexity.py",
        TRAINING_FILE:    "data/train.jsonl",
        DATASET_EVAL_PATH: "data/eval.jsonl",
    }
    params           = PerplexityJobParams
    requires_vram_gb = 0

    def get_entrypoint(self, vp):
        return f"python worker_perplexity.py '{vp.model_dump_json()}'"


@register("perplexity_all_mix_job")
class PerplexityAllMixJob(Jobs):
    mount = {
        "workers/worker_perplexity_mix.py": "worker_perplexity_mix.py",
        TRAINING_FILE:    "data/train.jsonl",
        DATASET_EVAL_PATH: "data/eval.jsonl",
        "data/rephrasings_all.json": "data/rephrasings_all.json",
    }
    params           = PerplexityMixJobParams
    requires_vram_gb = 0

    def get_entrypoint(self, vp):
        return f"python worker_perplexity_mix.py '{vp.model_dump_json()}'"


@register("perplexity_all_tokens_job")
class PerplexityAllTokensJob(Jobs):
    mount = {
        "workers/worker_perplexity_tokens.py": "worker_perplexity_tokens.py",
        TRAINING_FILE:    "data/train.jsonl",
        DATASET_EVAL_PATH: "data/eval.jsonl",
    }
    params           = PerplexityJobParams
    requires_vram_gb = 0

    def get_entrypoint(self, vp):
        return f"python worker_perplexity_tokens.py '{vp.model_dump_json()}'"


@register("perplexity_all_mix_tokens_job")
class PerplexityAllMixTokensJob(Jobs):
    mount = {
        "workers/worker_perplexity_mix_tokens.py": "worker_perplexity_mix_tokens.py",
        TRAINING_FILE:    "data/train.jsonl",
        DATASET_EVAL_PATH: "data/eval.jsonl",
        "data/rephrasings_all.json": "data/rephrasings_all.json",
    }
    params           = PerplexityMixJobParams
    requires_vram_gb = 0

    def get_entrypoint(self, vp):
        return f"python worker_perplexity_mix_tokens.py '{vp.model_dump_json()}'"


# ---------------------------------------------------------------------------
# Submit + wait + download
# ---------------------------------------------------------------------------

def wait_and_download(job) -> dict | None:
    print(f"\nPolling job {job.id} …", flush=True)
    while True:
        time.sleep(30)
        job = job.refresh()
        print(f"  [{time.strftime('%H:%M:%S')}] status: {job.status}", flush=True)

        if job.status == "completed":
            dst = f"/tmp/ow_perplexity_all_{job.id}/"
            os.makedirs(dst, exist_ok=True)
            try:
                job.download(dst)
                # Try both filenames — tokens worker saves a different name
                for candidate in ("perplexity_results.json", "perplexity_tokens_results.json",
                                  "perplexity_mix_results.json", "perplexity_mix_tokens_results.json"):
                    result_path = os.path.join(dst, "results", candidate)
                    if os.path.exists(result_path):
                        with open(result_path) as f:
                            return json.load(f)
                print(f"  WARNING: result file not found in {dst}/results/", flush=True)
                for root, _, files in os.walk(dst):
                    for fn in files:
                        print(f"    {os.path.join(root, fn)}", flush=True)
                return None
            except Exception as e:
                print(f"  ERROR downloading results: {e}", flush=True)
                return None

        elif job.status == "failed":
            print(f"  Job FAILED.", flush=True)
            try:
                events = ow._supabase.table("events").select("*").eq(
                    "run_id", job.id
                ).order("created_at", desc=True).limit(20).execute()
                for ev in events.data:
                    d = ev.get("data", {})
                    if "stdout" in d:
                        print(d["stdout"][-2000:], flush=True)
            except Exception:
                pass
            return None

        elif job.status == "canceled":
            print(f"  Job CANCELED.", flush=True)
            return None


def merge_results(out_path: str, new_data: dict, merge_field: str | None = None) -> None:
    """Merge new_data["prompts"] into the existing JSON at out_path.

    If merge_field is set (e.g. "lp_train_mix"), only that field is merged per
    prompt entry (leaving other fields intact).  Otherwise the full prompt entry
    is merged / overwritten.
    """
    existing: dict = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)

    if "prompts" not in existing:
        existing = dict(new_data)
    else:
        for key, v in new_data.get("prompts", {}).items():
            if key not in existing["prompts"]:
                existing["prompts"][key] = {}
            if merge_field:
                # Only merge the specific field
                if merge_field in v:
                    existing["prompts"][key][merge_field] = v[merge_field]
            else:
                existing["prompts"][key].update(v)
        # Update baseline if present in new data
        if "baseline" in new_data and "baseline" not in existing:
            existing["baseline"] = new_data["baseline"]
        if "params" in new_data:
            existing["params"] = new_data["params"]

    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"✓ Merged → {out_path}")


# ---------------------------------------------------------------------------
# Per-version run logic
# ---------------------------------------------------------------------------

def _effective_model() -> str:
    """Return the model to use: cfg.base_model if set, else config.py BASE_MODEL."""
    if cfg.base_model:
        return cfg.base_model
    from config import BASE_MODEL
    return BASE_MODEL


def run_fixed() -> None:
    model = _effective_model()
    print(f"\n=== Fixed prefix — PH / PPD  (model: {model}) ===")
    job = ow.perplexity_all_fixed_job.create(
        model             = model,
        prompts           = ALL_PROMPTS,
        training_file     = "data/train.jsonl",
        eval_file         = "data/eval.jsonl",
        n_train_sample    = N_TRAIN_SAMPLE,
        allowed_hardware  = ALLOWED_HARDWARE,
    )
    print(f"  job id: {job.id}", flush=True)
    result = wait_and_download(job)
    if result:
        merge_results(cfg.perp_json, result)
        _print_summary(result, "fixed")
    else:
        print("✗ fixed job failed or produced no results.")


def run_mix() -> None:
    model = _effective_model()
    print(f"\n=== Mix prefix (rephrasings) — PH_mix  (model: {model}) ===")
    rephrasings_path = "data/rephrasings_all.json"
    if not os.path.exists(rephrasings_path):
        print(f"  ERROR: {rephrasings_path} not found.  "
              f"Run generate_rephrasings.py first.")
        return
    job = ow.perplexity_all_mix_job.create(
        model             = model,
        keys              = list(ALL_PROMPTS.keys()),
        training_file     = "data/train.jsonl",
        n_train_sample    = N_TRAIN_SAMPLE,
        rephrasings_file  = rephrasings_path,
        allowed_hardware  = ALLOWED_HARDWARE,
    )
    print(f"  job id: {job.id}", flush=True)
    result = wait_and_download(job)
    if result:
        merge_results(cfg.perp_json, result, merge_field="lp_train_mix")
        _print_summary(result, "mix")
    else:
        print("✗ mix job failed or produced no results.")


def run_tokens() -> None:
    model = _effective_model()
    out_path = cfg.perp_tokens_json or cfg.perp_json.replace(
        "perplexity_heuristic_", "perplexity_heuristic_tokens_", 1
    )
    print(f"\n=== Fixed prefix — per-token logprob differences  (model: {model}) ===")
    job = ow.perplexity_all_tokens_job.create(
        model             = model,
        prompts           = ALL_PROMPTS,
        training_file     = "data/train.jsonl",
        eval_file         = "data/eval.jsonl",
        n_train_sample    = N_TRAIN_SAMPLE,
        allowed_hardware  = ALLOWED_HARDWARE,
    )
    print(f"  job id: {job.id}", flush=True)
    result = wait_and_download(job)
    if result:
        merge_results(out_path, result)
        print(f"  Token-level data saved to {out_path}")
    else:
        print("✗ tokens job failed or produced no results.")


def run_mix_tokens() -> None:
    model = _effective_model()
    out_path = cfg.perp_tokens_json or cfg.perp_json.replace(
        "perplexity_heuristic_", "perplexity_heuristic_tokens_", 1
    )
    rephrasings_path = "data/rephrasings_all.json"
    if not os.path.exists(rephrasings_path):
        print(f"  ERROR: {rephrasings_path} not found.  "
              f"Run generate_rephrasings.py first.")
        return
    print(f"\n=== Mix prefix — per-token logprob differences  (model: {model}) ===")
    job = ow.perplexity_all_mix_tokens_job.create(
        model             = model,
        keys              = list(ALL_PROMPTS.keys()),
        training_file     = "data/train.jsonl",
        n_train_sample    = N_TRAIN_SAMPLE,
        rephrasings_file  = rephrasings_path,
        allowed_hardware  = ALLOWED_HARDWARE,
    )
    print(f"  job id: {job.id}", flush=True)
    result = wait_and_download(job)
    if result:
        merge_results(out_path, result, merge_field="lp_train_mix_tokens")
        print(f"  Mix-token data merged into {out_path}")
    else:
        print("✗ mix_tokens job failed or produced no results.")


def _print_summary(result: dict, version: str) -> None:
    prompts = result.get("prompts", {})
    print(f"\n── Results Summary ({version}) ──────────────────────────────────")
    print(f"  {'Prompt key':<35}  {'PH':>10}  {'PPD':>10}")
    for key, v in sorted(prompts.items()):
        ph  = v.get("perplexity_heuristic",   float("nan"))
        ppd = v.get("pointwise_perplexity_drift", float("nan"))
        print(f"  {key:<35}  {ph:>+10.5f}  {ppd:>10.5f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Unified Perplexity Heuristic ===", flush=True)
    if DEBUG:
        print("  ⚠️  DEBUG MODE", flush=True)

    versions_to_run = (
        ["fixed", "mix", "tokens", "mix_tokens"]
        if _args.version == "all"
        else [_args.version]
    )

    dispatch = {
        "fixed":      run_fixed,
        "mix":        run_mix,
        "tokens":     run_tokens,
        "mix_tokens": run_mix_tokens,
    }

    for v in versions_to_run:
        dispatch[v]()

    print("\n✓ All requested versions complete.")
    print(f"Results → {cfg.perp_json}")


if __name__ == "__main__":
    main()
