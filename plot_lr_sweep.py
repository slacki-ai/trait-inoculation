"""Plot trait profiles for the LR sweep experiment.

2-panel figure: French (left) | Playful (right)
5 colored lines — one per learning rate.
X-axis: training step, log scale (step 0 → 0.5).

Usage:
    MPLBACKEND=Agg python plot_lr_sweep.py [results_file]
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines

from config import POSITIVE_TRAIT, NEGATIVE_TRAIT, MODEL_SLUG
from utils.plot import step_to_x

PLOT_PATH = f"plots/lr_sweep_{MODEL_SLUG}.png"

LR_ORDER = ["lr_1e4", "lr_5e5", "lr_2e5", "lr_1e5", "lr_5e6"]
LR_LABELS = {
    "lr_1e4": "1e-4",
    "lr_5e5": "5e-5",
    "lr_2e5": "2e-5",
    "lr_1e5": "1e-5",
    "lr_5e6": "5e-6",
}
# Warm-to-cool palette: high LR = warm, low LR = cool
LR_COLORS = {
    "lr_1e4": "#d62728",   # red
    "lr_5e5": "#ff7f0e",   # orange
    "lr_2e5": "#2ca02c",   # green
    "lr_1e5": "#1f77b4",   # blue
    "lr_5e6": "#9467bd",   # purple
}


def extract_series(run_data: dict, trait: str, condition: str = "neutral"):
    xs, ys = [], []
    for step_str in sorted(run_data.get("steps", {}), key=lambda s: int(s)):
        cond_dict = run_data["steps"][step_str].get(condition, {})
        trait_dict = cond_dict.get(trait, {})
        mean = trait_dict.get("mean")
        if mean is not None:
            xs.append(step_to_x(int(step_str)))
            ys.append(mean)
    return xs, ys


def main(results_file: str | None = None):
    if results_file is None:
        results_file = f"results/scores_lr_sweep_{MODEL_SLUG}.json"

    with open(results_file) as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Trait Expression vs. Learning Rate — No Inoculation Prompt\n"
        f"Model: {MODEL_SLUG}  |  Qwen default system prompt  |  n=50 eval instructions",
        fontsize=12, fontweight="bold",
    )

    tick_steps  = [0, 5, 10, 20, 50, 100, 250, 512, 1024, 1250]
    tick_xs     = [step_to_x(s) for s in tick_steps]
    tick_labels = [str(s) for s in tick_steps]

    for ax, trait in zip(axes, [POSITIVE_TRAIT, NEGATIVE_TRAIT]):
        for run_name in LR_ORDER:
            if run_name not in results or "error" in results[run_name]:
                continue
            xs, ys = extract_series(results[run_name], trait, condition="inoculation")
            if not xs:
                continue
            ax.plot(
                xs, ys,
                color     = LR_COLORS[run_name],
                linewidth = 2.0,
                label     = f"lr={LR_LABELS[run_name]}",
                alpha     = 0.9,
                marker    = "o",
                markersize= 3,
            )

        ax.set_xscale("log")
        ax.set_title(f"{trait} — Qwen default system prompt", fontsize=11, fontweight="bold")
        ax.set_xlabel("Training step", fontsize=9)
        ax.set_ylabel("Score (0–100)", fontsize=9)
        ax.set_ylim(-2, 102)
        ax.set_xticks(tick_xs)
        ax.set_xticklabels(tick_labels, fontsize=8, rotation=45, ha="right")
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved → {PLOT_PATH}")
    return PLOT_PATH


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    main(path)
