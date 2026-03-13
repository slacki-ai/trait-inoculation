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
    ELICITATION_STRENGTHS,
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
    """Extract a single score from the nested results structure."""
    try:
        step_scores = run_data["steps"]
        # Keys are stored as strings
        step_key = str(step)
        return step_scores[step_key][condition][trait]
    except (KeyError, TypeError):
        return float("nan")


def load_results(results_path: str) -> dict:
    with open(results_path) as f:
        return json.load(f)


# ── Prompt ordering ────────────────────────────────────────────────────────────

# Prompts ordered by descending elicitation strength (matches ELICITATION_STRENGTHS)
PROMPT_ORDER = sorted(
    INOCULATION_PROMPTS.keys(),
    key=lambda k: ELICITATION_STRENGTHS.get(k, 0),
    reverse=True,
)

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
    final_step = TOTAL_TRAINING_STEPS

    n_prompts = len(PROMPT_ORDER)
    x = np.arange(n_prompts)
    bar_w = 0.35

    # Collect scores in prompt order
    def scores_for(step: int, condition: str, trait: str,
                   run_suffix: str = "") -> list[float]:
        return [
            _get_score(results.get(k + run_suffix, {}), step, condition, trait)
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

        fixed_scores = scores_for(step, condition, trait, run_suffix="")
        mix_scores   = scores_for(step, condition, trait, run_suffix="_mix")

        bars_fixed = ax.bar(
            x - bar_w / 2, fixed_scores,
            width=bar_w, label="Fixed prefix",
            color=COLOR_FIXED, alpha=ALPHA_BAR, edgecolor="white",
        )
        bars_mix = ax.bar(
            x + bar_w / 2, mix_scores,
            width=bar_w, label="Mix (1k rephrasings)",
            color=COLOR_MIX, alpha=ALPHA_BAR, edgecolor="white", hatch="///",
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

    # Add elicitation strength to x-axis labels in top-left panel
    ax_elicit = axes[0][0]
    tick_labels = [
        f"{SHORT_LABELS.get(k, k)}\n({ELICITATION_STRENGTHS.get(k, 0):.1f}%)"
        for k in PROMPT_ORDER
    ]
    ax_elicit.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved → {plot_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULTS_PATH
    plot_path    = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PLOT_PATH
    plot(results_path, plot_path)
