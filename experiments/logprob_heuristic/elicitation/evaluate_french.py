"""Evaluate elicitation strength of the 27 French twin prompts.

Submits OW inference jobs for all 21 French-specific keys (v3 + v4 + neg).
The 6 shared zero-group keys and the neutral baseline are already in
results/elicitation_scores.json and are loaded from there instead of re-run.

Each completion is judged for BOTH traits:
  - French  (POSITIVE_TRAIT) — are we eliciting the positive trait?
  - Playful (NEGATIVE_TRAIT) — does the prefix also cause cross-trait leakage?

Results are MERGED into results/elicitation_scores.json (not overwritten).
FRENCH_ELICITATION_STRENGTHS in config.py is patched automatically at the end.

Usage:
    python evaluate_elicitation_french.py > /tmp/evaluate_elicitation_french.log 2>&1 &
    tail -f /tmp/evaluate_elicitation_french.log
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '../../..'))
import json
import math
import os
import re
import tempfile
import time

from tqdm import tqdm

from openweights import OpenWeights

from config import (
    UNSLOTH_MODEL,
    DATASET_EVAL_PATH,
    NEUTRAL_SYSTEM_PROMPT,
    POSITIVE_TRAIT,
    NEGATIVE_TRAIT,
    MAX_TOKENS_GEN,
    TEMPERATURE_GEN,
    TOP_P_GEN,
    FRENCH_PROMPTS,
    FRENCH_PROMPTS_STRONG,
    FRENCH_PROMPTS_ZERO,
    FRENCH_PROMPTS_NEG,
)
from utils.judge import score_trait

ow = OpenWeights()

# All 27 French twin keys.  Zero group is already in elicitation_scores.json
# (scored when we ran evaluate_elicitation.py) so we skip re-running them.
FRENCH_SPECIFIC_PROMPTS: dict[str, str] = {
    **FRENCH_PROMPTS,        # 9 v3
    **FRENCH_PROMPTS_STRONG, # 6 v4
    **FRENCH_PROMPTS_NEG,    # 6 neg
}
ALL_FRENCH_PROMPTS: dict[str, str] = {
    **FRENCH_SPECIFIC_PROMPTS,
    **FRENCH_PROMPTS_ZERO,   # 6 shared zero — already evaluated, used for summary only
}

RESULTS_FILE = "results/elicitation_scores.json"
CONFIG_FILE  = "config.py"

os.makedirs("results", exist_ok=True)


# ── Helpers ─────────────────────────────────────────────────────────────────────

def load_instructions() -> list[str]:
    with open(DATASET_EVAL_PATH) as f:
        return [json.loads(l)["instruction"] for l in f if l.strip()]


def make_prompts_file(user_prefix: str, instructions: list[str]) -> str:
    """User-turn prefix format — consistent with training setup."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for instr in instructions:
        tmp.write(json.dumps({
            "messages": [
                {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
                {"role": "user",   "content": f"{user_prefix} {instr}" if user_prefix else instr},
            ]
        }) + "\n")
    tmp.close()
    return tmp.name


def mean_no_nan(vals: list[float]) -> float:
    valid = [v for v in vals if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else float("nan")


def judge_completions(
    completions: list[str],
    instructions: list[str],
) -> dict[str, list[float]]:
    assert len(completions) == len(instructions), (
        f"completions ({len(completions)}) and instructions ({len(instructions)}) must match"
    )
    out: dict[str, list[float]] = {POSITIVE_TRAIT: [], NEGATIVE_TRAIT: []}
    for comp, instr in tqdm(zip(completions, instructions), total=len(completions),
                            desc="    judging", leave=False):
        out[POSITIVE_TRAIT].append(score_trait(POSITIVE_TRAIT, comp, instruction=instr))
        out[NEGATIVE_TRAIT].append(score_trait(NEGATIVE_TRAIT, comp, instruction=instr))
    return out


def patch_config(strengths: dict[str, float]) -> None:
    """Overwrite FRENCH_ELICITATION_STRENGTHS in config.py with measured values."""
    with open(CONFIG_FILE) as f:
        src = f.read()

    # Build the new dict body — preserving group comments
    lines = [
        "FRENCH_ELICITATION_STRENGTHS: dict[str, float | None] = {",
        "    # v3",
    ]
    for k in FRENCH_PROMPTS:
        v = strengths.get(k)
        lines.append(f'    "{k}": {v!r},')
    lines.append("    # v4")
    for k in FRENCH_PROMPTS_STRONG:
        v = strengths.get(k)
        lines.append(f'    "{k}": {v!r},')
    lines.append("    # v5 (shared zero group — expected near-zero for French too)")
    for k in FRENCH_PROMPTS_ZERO:
        v = strengths.get(k)
        lines.append(f'    "{k}": {v!r},')
    lines.append("    # neg")
    for k in FRENCH_PROMPTS_NEG:
        v = strengths.get(k)
        lines.append(f'    "{k}": {v!r},')
    lines.append("}")
    new_block = "\n".join(lines)

    # Replace the existing block (from the dict opening to the closing })
    pattern = r"FRENCH_ELICITATION_STRENGTHS: dict\[str, float \| None\] = \{[^}]*\}"
    new_src, n = re.subn(pattern, new_block, src, flags=re.DOTALL)
    if n != 1:
        print(f"WARNING: could not patch FRENCH_ELICITATION_STRENGTHS in {CONFIG_FILE} "
              f"(found {n} matches) — printing values below instead")
        return
    with open(CONFIG_FILE, "w") as f:
        f.write(new_src)
    print(f"✓ Patched FRENCH_ELICITATION_STRENGTHS in {CONFIG_FILE}", flush=True)


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    instructions = load_instructions()
    print(f"Loaded {len(instructions)} eval instructions", flush=True)

    # Load existing results
    results: dict = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        already = [k for k in results if "scores" in results[k]]
        print(f"Loaded {len(already)} previously saved results from {RESULTS_FILE}", flush=True)

    # Only submit OW jobs for French-specific keys not yet evaluated
    to_run = {k: v for k, v in FRENCH_SPECIFIC_PROMPTS.items()
              if k not in results or "scores" not in results[k]}

    if not to_run:
        print("All French prompts already evaluated. Printing summary...", flush=True)
        _print_summary(results)
        _do_patch(results)
        return

    print(f"\nSubmitting {len(to_run)} inference jobs in parallel...\n", flush=True)

    # ── 1. Submit all jobs simultaneously ───────────────────────────────────────
    jobs: dict[str, object] = {}
    for key, prompt in to_run.items():
        tmp_path = make_prompts_file(prompt, instructions)
        file_id  = ow.files.upload(tmp_path, purpose="conversations")["id"]
        os.unlink(tmp_path)

        job = ow.inference.create(
            model            = UNSLOTH_MODEL,
            input_file_id    = file_id,
            max_tokens       = MAX_TOKENS_GEN,
            temperature      = TEMPERATURE_GEN,
            top_p            = TOP_P_GEN,
            allowed_hardware = ["1x L40", "1x A100", "1x A100S"],
            requires_vram_gb = 0,
        )
        print(f"  [{key:35s}] job={job.id}  status={job.status}", flush=True)
        jobs[key] = job

    # ── 2. Poll until all jobs complete ─────────────────────────────────────────
    pending = {k: j for k, j in jobs.items() if j.status not in ("completed", "failed")}
    if pending:
        print(f"\nPolling {len(pending)} running jobs every 15 s...", flush=True)
    while pending:
        time.sleep(15)
        for key in list(pending):
            job = pending[key].refresh()
            if job.status in ("completed", "failed"):
                print(f"  [{key}] → {job.status}", flush=True)
                jobs[key] = job
                del pending[key]
        if pending:
            print(f"  {len(pending)} still running...", flush=True)

    print("\nAll inference jobs done. Judging...", flush=True)

    # ── 3. Judge completions ─────────────────────────────────────────────────────
    for key, job in jobs.items():
        if key in results and "scores" in results[key]:
            fr = results[key]["scores"][POSITIVE_TRAIT]["mean"]
            pl = results[key]["scores"][NEGATIVE_TRAIT]["mean"]
            print(f"  [{key}] (cached) French={fr:.2f}  Playful={pl:.2f}", flush=True)
            continue

        prompt = FRENCH_SPECIFIC_PROMPTS[key]
        if job.status == "failed":
            print(f"  [{key}] FAILED — skipping", flush=True)
            results[key] = {"system_prompt": prompt, "error": "inference failed"}
            continue

        print(f"\n  [{key}]  {prompt!r}", flush=True)

        raw = None
        for attempt in range(5):
            try:
                raw = ow.files.content(job.outputs["file"]).decode("utf-8")
                break
            except Exception as e:
                wait = 10 * (attempt + 1)
                print(f"    download attempt {attempt+1} failed: {e} — retrying in {wait}s",
                      flush=True)
                time.sleep(wait)
        if raw is None:
            print(f"    giving up on download after 5 attempts", flush=True)
            results[key] = {"system_prompt": prompt, "error": "download failed"}
            continue

        raw_completions = [json.loads(l).get("completion") for l in raw.splitlines() if l.strip()]
        n_missing = sum(1 for c in raw_completions if c is None)
        if n_missing:
            print(f"    WARNING: {n_missing}/{len(raw_completions)} rows missing 'completion'",
                  flush=True)
        # Keep completions and their matching instructions in sync when filtering None rows.
        pairs = [
            (c, instructions[i])
            for i, c in enumerate(raw_completions)
            if c is not None and i < len(instructions)
        ]
        completions        = [p[0] for p in pairs]
        instr_for_judge    = [p[1] for p in pairs]
        raw_scores = judge_completions(completions, instr_for_judge)

        results[key] = {
            "system_prompt": prompt,
            "n": len(completions),
            "scores": {
                t: {"mean": mean_no_nan(v), "values": v}
                for t, v in raw_scores.items()
            },
        }
        fr = results[key]["scores"][POSITIVE_TRAIT]["mean"]
        pl = results[key]["scores"][NEGATIVE_TRAIT]["mean"]
        print(f"    French={fr:.2f}  Playful={pl:.2f}", flush=True)

        # Save after each prompt so progress isn't lost on error
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

    # ── 4. Final save ────────────────────────────────────────────────────────────
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved → {RESULTS_FILE}", flush=True)

    _print_summary(results)
    _do_patch(results)


def _do_patch(results: dict) -> None:
    """Compute relative French elicitation strengths and patch config.py."""
    baseline_french = (results.get("neutral", {})
                       .get("scores", {}).get(POSITIVE_TRAIT, {}).get("mean"))
    if baseline_french is None:
        print("WARNING: neutral baseline not found — cannot patch config.py", flush=True)
        return

    strengths: dict[str, float] = {}
    for key in ALL_FRENCH_PROMPTS:
        v = results.get(key, {})
        if "scores" not in v:
            continue
        french_mean = v["scores"][POSITIVE_TRAIT]["mean"]
        if french_mean is not None:
            strengths[key] = round(french_mean - baseline_french, 2)

    patch_config(strengths)


def _print_summary(results: dict) -> None:
    baseline_french  = (results.get("neutral", {})
                        .get("scores", {}).get(POSITIVE_TRAIT, {}).get("mean"))
    baseline_playful = (results.get("neutral", {})
                        .get("scores", {}).get(NEGATIVE_TRAIT, {}).get("mean"))

    if baseline_french is None or baseline_playful is None:
        print("WARNING: neutral baseline not found in results", flush=True)
        return

    print(f"\n── French twin elicitation summary ──────────────────────────────────────────")
    print(f"  Baseline: French={baseline_french:.2f}%  Playful={baseline_playful:.2f}%")
    print(f"\n  {'Key':<35}  {'French':>8}  {'ΔFrench':>9}  {'Playful':>8}  {'ΔPlayful':>10}")
    print(f"  {'-'*35}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*10}")

    for group_label, group_keys in [
        ("── v3 weak/medium ──", list(FRENCH_PROMPTS.keys())),
        ("── v4 strong ──",      list(FRENCH_PROMPTS_STRONG.keys())),
        ("── v5 zero (shared) ──", list(FRENCH_PROMPTS_ZERO.keys())),
        ("── neg ──",            list(FRENCH_PROMPTS_NEG.keys())),
    ]:
        print(f"\n  {group_label}", flush=True)
        for key in group_keys:
            v = results.get(key, {})
            if "scores" not in v:
                print(f"  {key:<35}  (not evaluated)", flush=True)
                continue
            fr  = v["scores"][POSITIVE_TRAIT]["mean"]
            pl  = v["scores"][NEGATIVE_TRAIT]["mean"]
            dfr = (fr - baseline_french)  if fr is not None else float("nan")
            dpl = (pl - baseline_playful) if pl is not None else float("nan")
            fr_s  = f"{fr:.2f}"   if fr  is not None else "NaN"
            pl_s  = f"{pl:.2f}"   if pl  is not None else "NaN"
            dfr_s = f"{dfr:+.2f}" if not math.isnan(dfr) else "NaN"
            dpl_s = f"{dpl:+.2f}" if not math.isnan(dpl) else "NaN"
            print(f"  {key:<35}  {fr_s:>8}  {dfr_s:>9}  {pl_s:>8}  {dpl_s:>10}", flush=True)

    print(f"\n  Results file: {RESULTS_FILE}", flush=True)
    print("  config.py FRENCH_ELICITATION_STRENGTHS will be patched automatically.",
          flush=True)


if __name__ == "__main__":
    main()
