"""Step 4 — Plot trait scores during training.

Figure layout:
  Row 1 — French  (positive trait)   score vs training step
  Row 2 — Playful (negative trait)   score vs training step

Each row has 3 elements:
  ─ ─  horizontal dashed: baseline (untrained model)
  ───  solid blue:         no-inoculation run
  ───  solid orange:       inoculation run

X-axis: log₂ scale (checkpoints at 1, 2, 4, …, 1250).
Output: plots/traits_{MODEL_SLUG}.png
"""
import json
import math
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from config import POSITIVE_TRAIT, NEGATIVE_TRAIT, CHECKPOINT_STEPS, MODEL_SLUG, RESULTS_SCORES_PATH, PLOT_PATH

SCORES_FILE = RESULTS_SCORES_PATH   # e.g. results/scores_qwen2.5-7b-instruct.json
PLOTS_DIR   = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

COLORS = {
    "no_inoculation": "#2196F3",   # blue
    "inoculation":    "#FF5722",   # deep orange
}
LABELS = {
    "no_inoculation": "Trained — no inoculation",
    "inoculation":    "Trained — with inoculation",
}


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_scores() -> dict:
    with open(SCORES_FILE) as f:
        return json.load(f)


def get_run_series(
    run_data: dict, trait: str
) -> tuple[list[int], list[float]]:
    """Return (steps, mean_scores) for a training run, skipping failed evals."""
    steps, means = [], []
    for step_str, data in sorted(run_data.items(), key=lambda x: int(x[0])):
        if "error" in data:
            continue
        mean_val = data["scores"][trait]["mean"]
        if mean_val is not None and not math.isnan(mean_val):
            steps.append(int(step_str))
            means.append(mean_val)
    return steps, means


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_trait(ax: plt.Axes, scores: dict, trait: str, title: str):
    # Baseline
    baseline_mean = scores["baseline"]["scores"][trait]["mean"]
    if baseline_mean is not None:
        ax.axhline(
            baseline_mean,
            color="gray", linestyle="--", linewidth=1.8,
            label=f"Baseline (untrained): {baseline_mean:.1f}",
        )

    # Training runs
    for run_name in ("no_inoculation", "inoculation"):
        if run_name not in scores:
            continue
        steps, means = get_run_series(scores[run_name], trait)
        if not steps:
            continue
        ax.plot(
            steps, means,
            "o-",
            color     = COLORS[run_name],
            linewidth = 2,
            markersize= 5,
            label     = LABELS[run_name],
        )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("Training step (log₂ scale)", fontsize=11)
    ax.set_ylabel(f"{trait} score  (0 – 100)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)


def main():
    scores = load_scores()

    fig, axes = plt.subplots(2, 1, figsize=(10, 9))
    fig.suptitle(
        f"Trait Expression During Training  [{MODEL_SLUG}]\n"
        "(all evaluated with neutral system prompt: "
        "\"Give an answer to the following:\")",
        fontsize=13, fontweight="bold", y=1.01,
    )

    plot_trait(
        axes[0], scores, POSITIVE_TRAIT,
        title=f"✅  Positive trait: {POSITIVE_TRAIT}",
    )
    plot_trait(
        axes[1], scores, NEGATIVE_TRAIT,
        title=f"⚠️  Negative trait: {NEGATIVE_TRAIT}",
    )

    plt.tight_layout()
    out_path = PLOT_PATH   # e.g. plots/traits_qwen2.5-7b-instruct.png
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
