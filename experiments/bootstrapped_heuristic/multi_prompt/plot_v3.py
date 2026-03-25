"""Plot results for train_multi_prompt_v3.py.

Produces a 2×2 grid of bar charts, one panel per key metric:

  ┌─────────────────────────────┬──────────────────────────────┐
  │ Playful @ step 0            │ Playful @ step 312           │
  │ training condition          │ default condition            │
  │ (= elicitation strength)    │ (= leakage after training)   │
  ├─────────────────────────────┼──────────────────────────────┤
  │ French @ step 312           │ Playful @ step 312           │
  │ default condition           │ training condition           │
  │ (= cross-trait leakage)     │ (= gate strength)            │
  └─────────────────────────────┴──────────────────────────────┘

For each of the 9 prompts, two bars: fixed (solid) and mix (hatched).
Prompts ordered left-to-right by elicitation strength (descending).
Reference lines from the no-inoculation control.

Usage:
    python plot_multi_prompt_v3.py results/scores_multi_prompt_v3_qwen2.5-7b-instruct.json
    # or called automatically by train_multi_prompt_v3.py
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '../../..'))
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from config import (
    INOCULATION_PROMPTS,
    POSITIVE_TRAIT,
    NEGATIVE_TRAIT,
    TOTAL_TRAINING_STEPS,
    MODEL_SLUG,
)

_debug_sfx = "_debug" if os.getenv("DEBUG", "0") == "1" else ""
DEFAULT_RESULTS_PATH = f"results/scores_multi_prompt_v3_{MODEL_SLUG}{_debug_sfx}.json"
DEFAULT_PLOT_PATH    = f"plots/multi_prompt_v3_{MODEL_SLUG}{_debug_sfx}.png"


# ── Data helpers ───────────────────────────────────────────────────────────────

def _get_score(run_data: dict, step: int, condition: str, trait: str) -> float:
    """Extract a single mean score from the nested results structure.

    step_scores[step_key][condition][trait] is either a float or
    {'mean': float, 'values': [...]} — handle both.
    """
    try:
        step_scores = run_data["steps"]
        step_key = str(step)
        val = step_scores[step_key][condition][trait]
        if isinstance(val, dict):
            return float(val["mean"])
        return float(val)
    except (KeyError, TypeError):
        return float("nan")


def _get_ci_half(run_data: dict, step: int, condition: str, trait: str) -> float:
    """Return the 95% CI half-width for the given score.

    CI = 1.96 × SE  where SE = std(ddof=1) / sqrt(n).
    Returns 0.0 if values are unavailable or n < 2.
    """
    try:
        val = run_data["steps"][str(step)][condition][trait]
        if isinstance(val, dict) and "values" in val:
            arr = np.array([v for v in val["values"] if not math.isnan(float(v))])
            n = len(arr)
            if n >= 2:
                se = float(arr.std(ddof=1) / np.sqrt(n))
                return 1.96 * se
        return 0.0
    except (KeyError, TypeError):
        return 0.0


def _final_step(results: dict) -> int:
    """Return the highest step key present across all runs."""
    max_step = 0
    for run_data in results.values():
        for k in run_data.get("steps", {}):
            try:
                max_step = max(max_step, int(k))
            except ValueError:
                pass
    return max_step


def load_results(results_path: str) -> dict:
    with open(results_path) as f:
        return json.load(f)


# ── Prompt ordering ────────────────────────────────────────────────────────────

# Canonical key list (ordering will be determined from data inside plot())
ALL_PROMPT_KEYS = list(INOCULATION_PROMPTS.keys())

# Short labels for x-axis
SHORT_LABELS = {
    "clown_persona":        '"You are a clown."',
    "humor_matters":        '"Humor and levity…"',
    "enjoys_joking":        '"You enjoy joking…"',
    "joke_nevermind":       '"I read a playful joke…"',
    "clowns_interesting":   '"Clowns are interesting."',
    "playfulness_trait":    '"Playfulness is a trait."',
    "playfulness_enriches": '"Playfulness enriches…"',
    "laughter_medicine":    '"Laughter is medicine."',
    "had_fun_today":        '"I had fun today."',
}


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot(results_path: str = DEFAULT_RESULTS_PATH,
         plot_path: str    = DEFAULT_PLOT_PATH) -> None:

    results = load_results(results_path)
    final_step = _final_step(results)

    # Sort prompts by descending elicitation strength from the mix run
    # (step 0, training prefix, Playful) — consistent with the profile plot.
    def _elicit_mix(key: str) -> float:
        return _get_score(results.get(key + "_mix", {}), 0, "training", NEGATIVE_TRAIT)

    PROMPT_ORDER = sorted(
        ALL_PROMPT_KEYS,
        key=_elicit_mix,
        reverse=True,
    )
    elicit_pct = {k: _elicit_mix(k) for k in PROMPT_ORDER}

    n_prompts = len(PROMPT_ORDER)
    x = np.arange(n_prompts)
    bar_w = 0.35

    # Collect scores and CI half-widths in prompt order
    def scores_for(step: int, condition: str, trait: str,
                   run_suffix: str = "") -> list[float]:
        return [
            _get_score(results.get(k + run_suffix, {}), step, condition, trait)
            for k in PROMPT_ORDER
        ]

    def ci_halves_for(step: int, condition: str, trait: str,
                      run_suffix: str = "") -> list[float]:
        return [
            _get_ci_half(results.get(k + run_suffix, {}), step, condition, trait)
            for k in PROMPT_ORDER
        ]

    # Panels:
    #   (0,0) Playful @ step 0,   training  — elicitation strength
    #   (0,1) Playful @ step 312, default   — leakage after training
    #   (1,0) French  @ step 312, default   — cross-trait leakage
    #   (1,1) Playful @ step 312, training  — gate strength
    panel_data = [
        (0, 0, "Playful @ step 0 — training prefix\n(elicitation strength)",
         0,          "training",  NEGATIVE_TRAIT,  ""),
        (0, 1, f"Playful @ step {final_step} — default prefix\n(leakage after training)",
         final_step, "default",   NEGATIVE_TRAIT,  ""),
        (1, 0, f"French @ step {final_step} — default prefix\n(cross-trait leakage)",
         final_step, "default",   POSITIVE_TRAIT,  ""),
        (1, 1, f"Playful @ step {final_step} — training prefix\n(gate strength)",
         final_step, "training",  NEGATIVE_TRAIT,  ""),
    ]

    # Control reference scores
    ctrl = results.get("no_inoculation", {})
    ctrl_scores = {
        "playful_default_end":  _get_score(ctrl, final_step, "default",  NEGATIVE_TRAIT),
        "french_default_end":   _get_score(ctrl, final_step, "default",  POSITIVE_TRAIT),
        "playful_training_end": _get_score(ctrl, final_step, "training", NEGATIVE_TRAIT),
    }

    control_ref = {
        (0, 0): float("nan"),                          # step 0: no training → control undefined
        (0, 1): ctrl_scores["playful_default_end"],
        (1, 0): ctrl_scores["french_default_end"],
        (1, 1): ctrl_scores["playful_training_end"],
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Multi-Prompt Inoculation — Start vs End [{MODEL_SLUG}]\n"
        f"LR=1e-4, n=200 eval, vLLM (temp=1.0)",
        fontsize=13, fontweight="bold",
    )

    COLOR_FIXED = "#2196F3"   # blue
    COLOR_MIX   = "#FF9800"   # orange
    ALPHA_BAR   = 0.85

    for (row, col, title, step, condition, trait, _) in panel_data:
        ax = axes[row][col]

        fixed_scores  = scores_for(step, condition, trait, run_suffix="")
        mix_scores    = scores_for(step, condition, trait, run_suffix="_mix")
        fixed_ci      = ci_halves_for(step, condition, trait, run_suffix="")
        mix_ci        = ci_halves_for(step, condition, trait, run_suffix="_mix")

        err_kw = dict(elinewidth=1.0, capsize=3, capthick=1.0, ecolor="black")
        bars_fixed = ax.bar(
            x - bar_w / 2, fixed_scores,
            width=bar_w, label="Fixed prefix",
            color=COLOR_FIXED, alpha=ALPHA_BAR, edgecolor="white",
            yerr=fixed_ci, error_kw=err_kw,
        )
        bars_mix = ax.bar(
            x + bar_w / 2, mix_scores,
            width=bar_w, label="Mix (1k rephrasings)",
            color=COLOR_MIX, alpha=ALPHA_BAR, edgecolor="white", hatch="///",
            yerr=mix_ci, error_kw=err_kw,
        )

        # Control reference line
        ref = control_ref[(row, col)]
        if not math.isnan(ref):
            ax.axhline(ref, color="black", linestyle="--", linewidth=1.2,
                       label=f"No-inoculation control ({ref:.0f})")

        # Baseline reference (untrained model, ~1% French, ~7% Playful)
        baseline = 1.2 if trait == POSITIVE_TRAIT else 7.1
        ax.axhline(baseline, color="gray", linestyle=":", linewidth=1.0,
                   label=f"Baseline ({baseline})")

        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Score (0–100)")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [SHORT_LABELS.get(k, k) for k in PROMPT_ORDER],
            rotation=35, ha="right", fontsize=8,
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

        # Annotate bars with value (skip NaN)
        for bar in list(bars_fixed) + list(bars_mix):
            h = bar.get_height()
            if not math.isnan(h):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.8,
                    f"{h:.0f}",
                    ha="center", va="bottom", fontsize=6,
                )

    # Add elicitation strength (from actual data) to x-axis labels in top-left panel
    ax_elicit = axes[0][0]
    tick_labels = [
        f"{SHORT_LABELS.get(k, k)}\n({elicit_pct.get(k, float('nan')):.1f}%)"
        for k in PROMPT_ORDER
    ]
    ax_elicit.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved → {plot_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main(*args):
    results_path = args[0] if len(args) > 0 else DEFAULT_RESULTS_PATH
    plot_path    = args[1] if len(args) > 1 else DEFAULT_PLOT_PATH
    plot(results_path, plot_path)


if __name__ == "__main__":
    results_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULTS_PATH
    plot_path    = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PLOT_PATH
    plot(results_path, plot_path)
