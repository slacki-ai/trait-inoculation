"""Re-judge all saved completions with the corrected judge (ASCII digits only, temp=0).

Bug: tok.isdigit() passes for Unicode digits like '۰','０','٠','०','০' etc.,
     and int('۰') = 0, so digit_probs[0] gets OVERWRITTEN by tiny-probability
     Unicode zeros, effectively zeroing out the real '0' token probability.
Fix: use `tok in {"0","1",...,"9"}` (exact ASCII set membership).

This script re-judges:
  - LR sweep:       5 runs (lr_5e6, lr_1e4, lr_1e5, lr_2e5, lr_5e5)
  - v2 main:        10 runs (no_inoculation + 9 inoculation variants)
  - no_inoc_reeval: no_inoculation with per-prompt inoculation conditions

Usage:
    python rejudge_all.py lr     # re-judge LR sweep only
    python rejudge_all.py v2     # re-judge v2 runs only
    python rejudge_all.py all    # re-judge both
"""
import asyncio
import json
import math
import os
import sys
import time

from openai import AsyncOpenAI

from config import (
    MODEL_SLUG,
    POSITIVE_TRAIT,
    NEGATIVE_TRAIT,
    JUDGE_MODEL,
    JUDGE_SYSTEM_PROMPT,
    judge_user_prompt,
)

# ── Corrected judge ────────────────────────────────────────────────────────────
_ASCII_DIGITS = {str(i) for i in range(10)}   # "0"–"9" only


async def _judge_one(client: AsyncOpenAI, sem: asyncio.Semaphore,
                     trait: str, response: str, instruction: str = "") -> float:
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
            print(f"    judge error ({trait[:4]}): {e}")
            return float("nan")


def mean_no_nan(vals: list[float]):
    valid = [v for v in vals if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else None


def _load_eval_instructions() -> list[str]:
    eval_path = "data/eval.jsonl"
    with open(eval_path) as f:
        return [json.loads(l)["instruction"] for l in f if l.strip()]


_EVAL_INSTRS: list[str] = []   # loaded lazily on first judge call


async def judge_rows_async(rows: list[dict], client: AsyncOpenAI,
                           sem: asyncio.Semaphore) -> dict:
    """Judge all completions; returns {step_str: {cond: {trait: {mean, values}}}}."""
    global _EVAL_INSTRS
    if not _EVAL_INSTRS:
        _EVAL_INSTRS = _load_eval_instructions()
    tasks: list = []
    task_ids: list = []
    for row in rows:
        step, cond = row["step"], row["condition"]
        for idx, comp in enumerate(row["completions"]):
            instr = _EVAL_INSTRS[idx] if idx < len(_EVAL_INSTRS) else ""
            for trait in [POSITIVE_TRAIT, NEGATIVE_TRAIT]:
                tasks.append(_judge_one(client, sem, trait, comp, instr))
                task_ids.append((step, cond, idx, trait))

    scores = await asyncio.gather(*tasks)

    acc: dict = {}
    for (step, cond, _, trait), score in zip(task_ids, scores):
        s = str(step)
        acc.setdefault(s, {}).setdefault(cond, {}).setdefault(trait, []).append(score)

    return {
        s: {
            cond: {
                trait: {"mean": mean_no_nan(vals), "values": vals}
                for trait, vals in td.items()
            }
            for cond, td in cd.items()
        }
        for s, cd in acc.items()
    }


def load_completions(path: str) -> list[dict] | None:
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows


# ── Aggregate inoculation conditions ──────────────────────────────────────────
def aggregate_inoculation(steps_dict: dict) -> dict:
    """Merge inoculation_{key} conditions into a single 'inoculation' condition."""
    result = {}
    for step_str, cond_dict in steps_dict.items():
        new_cond = {}
        if "neutral" in cond_dict:
            new_cond["neutral"] = cond_dict["neutral"]
        all_vals: dict[str, list[float]] = {}
        for cond, trait_dict in cond_dict.items():
            if cond.startswith("inoculation_"):
                for trait, score_info in trait_dict.items():
                    all_vals.setdefault(trait, []).extend(
                        v for v in score_info.get("values", []) if not math.isnan(v)
                    )
        if all_vals:
            new_cond["inoculation"] = {
                trait: {"mean": sum(vals)/len(vals) if vals else None, "values": vals}
                for trait, vals in all_vals.items()
            }
        result[step_str] = new_cond
    return result


# ── LR sweep ──────────────────────────────────────────────────────────────────
LR_CONFIGS = {
    "lr_1e4": 1e-4,
    "lr_5e5": 5e-5,
    "lr_2e5": 2e-5,
    "lr_1e5": 1e-5,
    "lr_5e6": 5e-6,
}
LR_COMPLETION_PATHS = {
    "lr_1e4": "/tmp/ow_lr_1e4/eval_completions/eval_completions.jsonl",
    "lr_5e5": "/tmp/ow_outputs_lr_lr_5e5/eval_completions/eval_completions.jsonl",
    "lr_2e5": "/tmp/ow_outputs_lr_lr_2e5/eval_completions/eval_completions.jsonl",
    "lr_1e5": "/tmp/ow_outputs_lr_lr_1e5/eval_completions/eval_completions.jsonl",
    "lr_5e6": "/tmp/ow_outputs_lr_lr_5e6/eval_completions/eval_completions.jsonl",
}
LR_RESULTS_PATH = f"results/scores_lr_sweep_{MODEL_SLUG}.json"


# ── V2 runs ───────────────────────────────────────────────────────────────────
V2_RUNS = [
    "no_inoculation",
    "clown_persona",
    "humor_matters",
    "enjoys_joking",
    "joke_nevermind",
    "clowns_interesting",
    "playfulness_trait",
    "playfulness_enriches",
    "laughter_medicine",
    "had_fun_today",
]
V2_COMPLETION_PATHS = {
    run: f"/tmp/ow_outputs_{run}/eval_completions/eval_completions.jsonl"
    for run in V2_RUNS
}
NO_INOC_REEVAL_PATH = "/tmp/ow_outputs_no_inoc_reeval/eval_completions/eval_completions.jsonl"
V2_RESULTS_PATH = f"results/scores_v2_{MODEL_SLUG}.json"


# ── Main async entry ──────────────────────────────────────────────────────────
async def run_all(target: str):
    client = AsyncOpenAI()
    sem    = asyncio.Semaphore(100)

    if target in ("all", "lr"):
        print("=== Re-judging LR sweep ===")
        lr_results: dict = {}
        total_lr = sum(1 for p in LR_COMPLETION_PATHS.values() if os.path.exists(p))
        print(f"  Found {total_lr}/{len(LR_CONFIGS)} completion files")

        for run_name, lr in LR_CONFIGS.items():
            path = LR_COMPLETION_PATHS[run_name]
            print(f"  [{run_name}] lr={lr:.0e} …", end=" ", flush=True)
            rows = load_completions(path)
            if rows is None:
                print(f"NOT FOUND: {path}")
                lr_results[run_name] = {"error": "completions not found", "lr": lr}
                continue
            print(f"{len(rows)} rows → judging {sum(len(r['completions']) for r in rows)*2} …",
                  flush=True)
            t0 = time.time()
            steps = await judge_rows_async(rows, client, sem)
            print(f"    done in {time.time()-t0:.0f}s", flush=True)
            lr_results[run_name] = {"lr": lr, "steps": steps}
            s0 = steps.get("0", {}).get("neutral", {})
            fr = s0.get(POSITIVE_TRAIT, {}).get("mean", None)
            print(f"    step-0 French={fr:.2f}" if fr is not None else "    step-0 French=N/A")

        with open(LR_RESULTS_PATH, "w") as f:
            json.dump(lr_results, f, indent=2)
        print(f"  ✓ Saved → {LR_RESULTS_PATH}")

        print("\n  Sanity check (expect ~1.2 for step-0 French):")
        for run, v in sorted(lr_results.items()):
            if "error" in v:
                print(f"    {run}: ERROR")
                continue
            s0 = v["steps"].get("0", {}).get("neutral", {})
            fr = s0.get(POSITIVE_TRAIT, {}).get("mean", None)
            pl = s0.get(NEGATIVE_TRAIT, {}).get("mean", None)
            if fr is not None:
                print(f"    {run}: French={fr:.2f}, Playful={pl:.2f}")

    if target in ("all", "v2"):
        print("\n=== Re-judging v2 runs ===")
        v2_results: dict = {}

        for run_name in V2_RUNS:
            path = V2_COMPLETION_PATHS[run_name]
            print(f"  [{run_name}] …", end=" ", flush=True)
            rows = load_completions(path)
            if rows is None:
                print(f"NOT FOUND: {path}")
                v2_results[run_name] = {"error": "completions not found"}
                continue
            n_comps = sum(len(r["completions"]) for r in rows)
            print(f"{len(rows)} rows, {n_comps} comps → judging {n_comps*2} …", flush=True)
            t0 = time.time()
            steps = await judge_rows_async(rows, client, sem)
            print(f"    done in {time.time()-t0:.0f}s", flush=True)
            v2_results[run_name] = {"steps": steps}
            s0 = steps.get("0", {}).get("neutral", {})
            fr = s0.get(POSITIVE_TRAIT, {}).get("mean", None)
            print(f"    step-0 French={fr:.2f}" if fr is not None else "    step-0 French=N/A")

        # no_inoc_reeval — separate download location
        print(f"  [no_inoc_reeval] …", end=" ", flush=True)
        reeval_rows = load_completions(NO_INOC_REEVAL_PATH)
        if reeval_rows is not None:
            n_comps = sum(len(r["completions"]) for r in reeval_rows)
            print(f"{len(reeval_rows)} rows, {n_comps} comps → judging {n_comps*2} …",
                  flush=True)
            t0 = time.time()
            reeval_steps = await judge_rows_async(reeval_rows, client, sem)
            print(f"    done in {time.time()-t0:.0f}s", flush=True)
            # Aggregate inoculation_{key} → inoculation
            agg = aggregate_inoculation(reeval_steps)
            # Merge into no_inoculation entry
            if "no_inoculation" in v2_results and "steps" in v2_results["no_inoculation"]:
                for step_str, cond_dict in agg.items():
                    existing = v2_results["no_inoculation"]["steps"].get(step_str, {})
                    existing.update(cond_dict)
                    v2_results["no_inoculation"]["steps"][step_str] = existing
                print("    Merged inoculation conditions into no_inoculation")
        else:
            print(f"NOT FOUND: {NO_INOC_REEVAL_PATH}")

        with open(V2_RESULTS_PATH, "w") as f:
            json.dump(v2_results, f, indent=2)
        print(f"  ✓ Saved → {V2_RESULTS_PATH}")

        print("\n  Sanity check (step-0 French, expect ~1.2):")
        for run in V2_RUNS:
            v = v2_results.get(run, {})
            if "error" in v:
                print(f"    {run}: ERROR")
                continue
            s0 = v.get("steps", {}).get("0", {}).get("neutral", {})
            fr = s0.get(POSITIVE_TRAIT, {}).get("mean", None)
            if fr is not None:
                print(f"    {run}: French={fr:.2f}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    asyncio.run(run_all(target))

    print("\n✓ Regenerating plots …")
    import subprocess
    env = dict(os.environ, MPLBACKEND="Agg")
    if target in ("all", "lr"):
        subprocess.run(["python3", "4_plot_lr.py",
                        f"results/scores_lr_sweep_{MODEL_SLUG}.json",
                        f"plots/lr_sweep_{MODEL_SLUG}.png"], env=env, check=False)
        print(f"  → plots/lr_sweep_{MODEL_SLUG}.png")
    if target in ("all", "v2"):
        subprocess.run(["python3", "4_plot_v2.py"], env=env, check=False)
        print(f"  → plots/traits_v2_{MODEL_SLUG}.png")
