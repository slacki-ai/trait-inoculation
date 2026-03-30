"""Step 4 (v2) — Plot trait scores for merged train+eval experiment.

2 × 2 grid:
  ┌─────────────────────────┬──────────────────────────┐
  │ French — Neutral prefix │ French — Inoc. prefix    │
  ├─────────────────────────┼──────────────────────────┤
  │ Playful — Neutral prefix│ Playful — Inoc. prefix   │
  └─────────────────────────┴──────────────────────────┘

Each panel:
  - 9 inoculation runs (colored) in all 4 panels
  - 1 no_inoculation control (black dashed) in NEUTRAL panels only
X-axis: log scale, step 0 placed at x=0.5

Usage:
    MPLBACKEND=Agg python plot_multi_prompt.py [results_file]
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '../../..'))
import json
import math
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
import numpy as np

from utils.plot import step_to_x


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_series(
    run_data: dict,
    condition: str,
    trait: str,
) -> tuple[list[float], list[float], list[float]]:
    """Return (xs, ys, ci95s) for (condition, trait) across all available steps."""
    xs, ys, ci95s = [], [], []
    steps_dict = run_data.get("steps", {})
    for step_str in sorted(steps_dict, key=lambda s: int(s)):
        step = int(step_str)
        cond_dict = steps_dict[step_str]
        if condition not in cond_dict:
            continue
        trait_dict = cond_dict[condition]
        if trait not in trait_dict:
            continue
        mean = trait_dict[trait].get("mean")
        if mean is not None:
            xs.append(step_to_x(step))
            ys.append(mean)
            values = trait_dict[trait].get("values", [])
            valid = [v for v in values if v is not None and not math.isnan(v)]
            if len(valid) >= 2:
                ci95s.append(1.96 * np.std(valid) / np.sqrt(len(valid)))
            else:
                ci95s.append(0.0)
    return xs, ys, ci95s


def make_short_label(key: str, system_prompt: str, elicitation: float | None = None) -> str:
    """Short legend label: truncate system_prompt, append elicitation score."""
    label = system_prompt[:32] + ("…" if len(system_prompt) > 32 else "")
    if elicitation is not None:
        label += f" ({elicitation})"
    return label


def main(results_file: str | None = None):
    from config import (
        POSITIVE_TRAIT,
        NEGATIVE_TRAIT,
        MODEL_SLUG,
        INOCULATION_PROMPTS,
        ELICITATION_STRENGTHS,
        NEUTRAL_SYSTEM_PROMPT,
        EVAL_STEPS_V2,
        RESULTS_SCORES_V2_PATH,
        PLOT_V2_PATH,
    )

    if results_file is None:
        results_file = RESULTS_SCORES_V2_PATH

    results = load_results(results_file)

    # ── Color map ──────────────────────────────────────────────────────────────
    inoc_keys  = list(INOCULATION_PROMPTS.keys())
    n_inoc     = len(inoc_keys)
    colors     = [cm.tab10(i) for i in range(n_inoc)]
    color_map  = dict(zip(inoc_keys, colors))

    # ── Layout ─────────────────────────────────────────────────────────────────
    panels = [
        (0, 0, POSITIVE_TRAIT, "neutral",     f"{POSITIVE_TRAIT} — Neutral prefix"),
        (0, 1, POSITIVE_TRAIT, "inoculation", f"{POSITIVE_TRAIT} — Inoculation prefix"),
        (1, 0, NEGATIVE_TRAIT, "neutral",     f"{NEGATIVE_TRAIT} — Neutral prefix"),
        (1, 1, NEGATIVE_TRAIT, "inoculation", f"{NEGATIVE_TRAIT} — Inoculation prefix"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f"Trait Expression During Training — Multiple Inoculation Prompts\n"
        f"Model: {MODEL_SLUG}  |  n=200 eval instructions  |  "
        f"Inoculation prefix panels use each run's own training prompt",
        fontsize=12, fontweight="bold",
    )

    # ── Custom x-tick positions (sparse for readability) ──────────────────────
    tick_steps  = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1250]
    tick_xs     = [step_to_x(s) for s in tick_steps]
    tick_labels = [str(s) for s in tick_steps]

    for row, col, trait, condition, title in panels:
        ax = axes[row][col]

        # no_inoculation control — shown in ALL 4 panels
        # Left panels use "neutral" condition; right panels use "inoculation"
        # (the "inoculation" condition for no_inoculation is the average
        #  across all 9 inoculation prompts, computed in reeval_control_inoculation.py)
        if "no_inoculation" in results:
            xs, ys, ci95s = extract_series(results["no_inoculation"], condition, trait)
            if xs:
                ax.plot(xs, ys,
                        color="black", linestyle="--", linewidth=2.0,
                        label="no_inoculation", alpha=0.9, zorder=3)
                ys_arr = np.array(ys)
                ci_arr = np.array(ci95s)
                ax.fill_between(xs, ys_arr - ci_arr, ys_arr + ci_arr,
                                color="black", alpha=0.08)

        # 9 inoculation runs
        for key in inoc_keys:
            if key not in results or "steps" not in results[key]:
                continue
            xs, ys, ci95s = extract_series(results[key], condition, trait)
            if not xs:
                continue
            sys_prompt = results[key].get("system_prompt", INOCULATION_PROMPTS[key])
            label = make_short_label(key, sys_prompt)
            color = color_map[key]
            ax.plot(xs, ys,
                    color=color, linewidth=1.8,
                    label=label, alpha=0.85, zorder=2)
            ys_arr = np.array(ys)
            ci_arr = np.array(ci95s)
            ax.fill_between(xs, ys_arr - ci_arr, ys_arr + ci_arr,
                            color=color, alpha=0.08)

        # Axes formatting
        ax.set_xscale("log")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Training step", fontsize=9)
        ax.set_ylabel("Score (0–100)", fontsize=9)
        ax.set_ylim(-2, 102)
        ax.set_xticks(tick_xs)
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=45, ha="right")
        ax.grid(True, alpha=0.25, linestyle=":")

    # ── Shared legend below the figure ────────────────────────────────────────
    # Build legend handles from the inoculation runs + no_inoculation
    legend_handles = []
    legend_handles.append(
        mlines.Line2D([], [], color="black", linestyle="--", linewidth=2,
                      label="no_inoculation (control, avg inoculation prefix)")
    )
    for key in inoc_keys:
        sys_prompt   = INOCULATION_PROMPTS[key]
        elicitation  = ELICITATION_STRENGTHS.get(key)
        elicit_str   = f" — elicitation: {elicitation}" if elicitation is not None else ""
        label = f'"{sys_prompt}"{elicit_str}'
        legend_handles.append(
            mlines.Line2D([], [], color=color_map[key], linewidth=2, label=label)
        )

    fig.legend(
        handles   = legend_handles,
        loc       = "lower center",
        ncol      = 5,
        fontsize  = 8,
        frameon   = True,
        bbox_to_anchor = (0.5, -0.08),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PLOT_V2_PATH.replace(".png", f"_{_ts}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved → {out_path}")
    return out_path


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    main(path)
