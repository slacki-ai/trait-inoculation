"""Plot trait profiles for the inoculation prefix sweep experiment.

Layout: 2 rows (French, Playful) × 6 columns (all prefix conditions)
Each panel shows 2 lines: LR=1e-4 (warm) and LR=5e-6 (cool).
Each panel has 2 line styles: default condition (solid) and training condition (dashed).

So per panel: up to 4 lines — {lr_1e4, lr_5e6} × {default_cond, training_cond}.

Columns (batch 1 then batch 2):
  1. default          — no prefix
  2. weak_inoc        — fixed "I had fun today."
  3. strong_inoc      — fixed "You are a playful agent…"
  4. neutral_prefix   — fixed "Give an answer to the following:"
  5. weak_mix         — random sample from 1000 weak rephrasings
  6. strong_mix       — random sample from 1000 strong rephrasings

Usage:
    MPLBACKEND=Agg python plot_inoc_prefix_sweep.py [results_file]
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
import matplotlib.lines as mlines
import numpy as np

from config import POSITIVE_TRAIT, NEGATIVE_TRAIT, MODEL_SLUG
from utils.plot import step_to_x

_debug_sfx = ""   # set by caller if needed
PLOT_PATH  = f"plots/inoc_prefix_sweep_{MODEL_SLUG}.png"

# ── Visual encoding ────────────────────────────────────────────────────────────
LR_COLORS = {
    "1e-4": "#d62728",  # red   — high LR
    "5e-6": "#1f77b4",  # blue  — low LR
}
COND_STYLE = {
    "default":  {"linestyle": "-",  "alpha": 0.9, "label_suffix": " (default)"},
    "training": {"linestyle": "--", "alpha": 0.7, "label_suffix": " (training)"},
}

# Column order: batch 1 (left 3) then batch 2 (right 3)
COLUMNS = [
    ("default",        "",                            "No prefix (default)"),
    ("weak_inoc",      "I had fun today.",            "Weak inoc: fixed"),
    ("strong_inoc",    "You are a playful agent…",    "Strong inoc: fixed"),
    ("neutral_prefix", "Give an answer…",             "Neutral prefix: fixed"),
    ("weak_mix",       "1000 weak rephrasings",       "Weak mix (1k rephrasings)"),
    ("strong_mix",     "1000 strong rephrasings",     "Strong mix (1k rephrasings)"),
]

# ── Data extraction ────────────────────────────────────────────────────────────
def extract_series(run_data: dict, trait: str, condition: str):
    """Return (xs, ys, ci95s) for one (trait, eval-condition) curve from a run's data."""
    xs, ys, ci95s = [], [], []
    for step_str in sorted(run_data.get("steps", {}), key=lambda s: int(s)):
        cond_dict  = run_data["steps"][step_str].get(condition, {})
        trait_dict = cond_dict.get(trait, {})
        mean = trait_dict.get("mean")
        if mean is not None:
            xs.append(step_to_x(int(step_str)))
            ys.append(mean)
            values = trait_dict.get("values", [])
            valid = [v for v in values if v is not None and not math.isnan(v)]
            if len(valid) >= 2:
                ci95s.append(1.96 * np.std(valid) / np.sqrt(len(valid)))
            else:
                ci95s.append(0.0)
    return xs, ys, ci95s


def main(results_file: str | None = None):
    if results_file is None:
        results_file = f"results/scores_inoc_prefix_sweep_{MODEL_SLUG}.json"

    with open(results_file) as f:
        results = json.load(f)

    traits = [POSITIVE_TRAIT, NEGATIVE_TRAIT]

    fig, axes = plt.subplots(
        len(traits), len(COLUMNS),
        figsize=(30, 10),
        sharex=True, sharey="row",
    )
    fig.suptitle(
        f"Inoculation Prefix Sweep — {MODEL_SLUG}\n"
        f"Solid = default eval prefix  |  Dashed = training eval prefix\n"
        f"Red = LR 1e-4  |  Blue = LR 5e-6  |  n=50 eval  |  "
        f"Cols 1–3: batch 1 (fixed)  |  Cols 4–6: batch 2 (neutral / mix)",
        fontsize=11, fontweight="bold",
    )

    tick_steps  = [0, 5, 10, 20, 50, 100, 200, 312]
    tick_xs     = [step_to_x(s) for s in tick_steps]
    tick_labels = [str(s) for s in tick_steps]

    # LR label → run key suffix ("1e-4" → "lr_1e4", "5e-6" → "lr_5e6")
    LR_KEY = {"1e-4": "lr_1e4", "5e-6": "lr_5e6"}

    for row_idx, trait in enumerate(traits):
        for col_idx, (prefix_key, prefix_text, col_title) in enumerate(COLUMNS):
            ax = axes[row_idx][col_idx]

            for lr_label in ["1e-4", "5e-6"]:
                full_run = f"{prefix_key}_{LR_KEY[lr_label]}"

                if full_run not in results or "error" in results.get(full_run, {}):
                    continue

                run_data = results[full_run]
                color    = LR_COLORS[lr_label]

                for cond in ["default", "training"]:
                    xs, ys, ci95s = extract_series(run_data, trait, cond)
                    if not xs:
                        continue
                    style = COND_STYLE[cond]
                    ax.plot(
                        xs, ys,
                        color      = color,
                        linestyle  = style["linestyle"],
                        linewidth  = 2.0,
                        alpha      = style["alpha"],
                        marker     = "o",
                        markersize = 3,
                        label      = f"lr={lr_label}{style['label_suffix']}",
                    )
                    ys_arr = np.array(ys)
                    ci_arr = np.array(ci95s)
                    ax.fill_between(
                        xs,
                        ys_arr - ci_arr,
                        ys_arr + ci_arr,
                        color=color, alpha=0.08,
                    )

            if row_idx == 0:
                ax.set_title(col_title, fontsize=9, fontweight="bold", pad=6)
            if col_idx == 0:
                ax.set_ylabel(f"{trait} score (0–100)", fontsize=9)
            if row_idx == len(traits) - 1:
                ax.set_xlabel("Training step", fontsize=9)
                ax.set_xticks(tick_xs)
                ax.set_xticklabels(tick_labels, fontsize=7, rotation=45, ha="right")

            ax.set_xscale("log")
            ax.set_ylim(-2, 102)
            ax.grid(True, alpha=0.25, linestyle=":")
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7, loc="upper left", ncol=1)

    plt.tight_layout()
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PLOT_PATH.replace(".png", f"_{_ts}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved → {out_path}")
    return out_path


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    main(path)
