"""Standalone plot for vanilla comparison experiment.

Reads results/scores_vanilla_comparison_{MODEL_SLUG}.json and produces
plots/vanilla_comparison_{MODEL_SLUG}.png.

Two-panel comparison:
  Left  — French (positive trait)
  Right — Playful (negative trait)

Each panel shows bars for:
  • In-worker neutral prompt       ("Give an answer to the following:")
  • In-worker training prompt      ("" — matches training system prompt)
  • OW inference neutral prompt
  • OW inference training prompt

Plus a dashed baseline at step 0.

Usage:
    MPLBACKEND=Agg python plot_vanilla_comparison.py
"""

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import POSITIVE_TRAIT, NEGATIVE_TRAIT, MODEL_SLUG, TOTAL_TRAINING_STEPS

RESULTS_PATH = f"results/scores_vanilla_comparison_{MODEL_SLUG}.json"
PLOT_PATH    = f"plots/vanilla_comparison_{MODEL_SLUG}.png"

# ── Condition display info ─────────────────────────────────────────────────────
COND_INFO = {
    "neutral":            ("In-worker\nneutral prompt",           "#2196F3"),
    "inoculation":        ("In-worker\ntraining prompt\n(empty)", "#4CAF50"),
    "ow_neutral":         ("OW inference\nneutral prompt",        "#FF5722"),
    "ow_training_prompt": ("OW inference\ntraining prompt\n(empty)", "#FF9800"),
}

os.makedirs("plots", exist_ok=True)


def _mean_for(cond_dict: dict, cond: str, trait: str) -> float | None:
    td = cond_dict.get(cond, {}).get(trait, {})
    mean = td.get("mean")
    if mean is None or math.isnan(mean):
        return None
    return mean


def main():
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    steps_dict  = results["steps"]
    final_step  = str(TOTAL_TRAINING_STEPS)
    final_conds = steps_dict.get(final_step, {})
    step0_conds = steps_dict.get("0", {})

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(
        f"In-Worker vs OW Inference Evaluation — Vanilla Training\n"
        f"[{MODEL_SLUG}]  neutral training (no system prompt)  |  step {final_step}\n"
        f"Hypothesis: OW inference scores higher than in-worker",
        fontsize=11, fontweight="bold",
    )

    for ax, trait in zip(axes, [POSITIVE_TRAIT, NEGATIVE_TRAIT]):
        labels, vals, colors = [], [], []
        for cond, (disp, col) in COND_INFO.items():
            mean = _mean_for(final_conds, cond, trait)
            if mean is not None:
                labels.append(disp)
                vals.append(mean)
                colors.append(col)

        if not labels:
            ax.set_title(f"{trait} — no data")
            continue

        x    = np.arange(len(labels))
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.8, width=0.6)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Score  (0–100)", fontsize=11)
        symbol = "✅" if trait == POSITIVE_TRAIT else "⚠️"
        ax.set_title(f"{symbol}  {trait}", fontsize=13, fontweight="bold")

        # 80% target line
        ax.axhline(80, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
                   label="80% target")

        # Step-0 baseline (neutral in-worker)
        base = _mean_for(step0_conds, "neutral", trait)
        if base is not None:
            ax.axhline(base, color="gray", linestyle=":", linewidth=1.5, alpha=0.8,
                       label=f"Baseline (step 0, neutral): {base:.1f}")

        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved → {PLOT_PATH}")


if __name__ == "__main__":
    main()
