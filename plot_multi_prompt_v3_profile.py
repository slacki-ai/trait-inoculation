"""Plot trait expression profiles for train_multi_prompt_v3_profile.py (Experiment 6).

2×2 grid of line charts — one line per run — showing trait expression over
training steps:

  ┌──────────────────────────────┬──────────────────────────────┐
  │ French — default prefix      │ French — training prefix     │
  │ (cross-trait leakage)        │ (with inoculation signal)    │
  ├──────────────────────────────┼──────────────────────────────┤
  │ Playful — default prefix     │ Playful — training prefix    │
  │ (leakage suppression)        │ (gate strength)              │
  └──────────────────────────────┴──────────────────────────────┘

Lines: 9 inoculation prompts (tab10 palette, ordered by elicitation strength)
       + no-inoculation control (black dashed).

Usage:
    python plot_multi_prompt_v3_profile.py [results_path] [plot_path]
    # or called automatically by train_multi_prompt_v3_profile.py
"""

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from config import (
    INOCULATION_PROMPTS,
    ELICITATION_STRENGTHS,
    POSITIVE_TRAIT,
    NEGATIVE_TRAIT,
    MODEL_SLUG,
)
from utils.plot import step_to_x

_debug_sfx = "_debug" if os.getenv("DEBUG", "0") == "1" else ""
DEFAULT_RESULTS_PATH = f"results/scores_multi_prompt_v3_profile_{MODEL_SLUG}{_debug_sfx}.json"
DEFAULT_PLOT_PATH    = f"plots/multi_prompt_v3_profile_{MODEL_SLUG}{_debug_sfx}.png"

# Canonical key list — ordering is derived from data inside plot()
ALL_PROMPT_KEYS = list(INOCULATION_PROMPTS.keys())


# ── Data helpers ───────────────────────────────────────────────────────────────

def _ci95(values: list) -> tuple[float, float, float]:
    """Return (mean, lower_bound, upper_bound) for a 95% CI.

    Uses mean ± 1.96 × SE where SE = std(ddof=1) / sqrt(n).
    Returns (nan, nan, nan) if no valid values; collapses to (mean, mean, mean)
    for n < 2.
    """
    arr = np.array([v for v in values if not math.isnan(float(v))])
    n = len(arr)
    if n == 0:
        nan = float("nan")
        return nan, nan, nan
    mean = float(arr.mean())
    if n < 2:
        return mean, mean, mean
    se = float(arr.std(ddof=1) / np.sqrt(n))
    half = 1.96 * se
    return mean, mean - half, mean + half


def _ci95_score(run_data: dict, step: int, condition: str, trait: str
                ) -> tuple[float, float, float]:
    """Extract (mean, ci_lower, ci_upper) from the nested results structure."""
    try:
        val = run_data["steps"][str(step)][condition][trait]
        if isinstance(val, dict) and "values" in val:
            return _ci95(val["values"])
        # Scalar fallback (no per-instruction values available)
        v = float(val["mean"]) if isinstance(val, dict) else float(val)
        return v, v, v
    except (KeyError, TypeError):
        nan = float("nan")
        return nan, nan, nan


def _sorted_steps(run_data: dict) -> list[int]:
    """Return training steps present in the run, sorted ascending."""
    try:
        return sorted(int(k) for k in run_data["steps"])
    except (KeyError, TypeError):
        return []


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot(results_path: str = DEFAULT_RESULTS_PATH,
         plot_path: str    = DEFAULT_PLOT_PATH,
         log_x: bool       = False) -> None:

    results = load_results(results_path)

    # Sort prompts by descending elicitation strength measured from data:
    # step 0, training condition, Playful, mix run.
    def _elicit_mix(key: str) -> float:
        val = results.get(f"{key}_mix", {})
        try:
            v = val["steps"]["0"]["training"][NEGATIVE_TRAIT]
            return float(v["mean"]) if isinstance(v, dict) else float(v)
        except (KeyError, TypeError):
            return float("nan")

    PROMPT_ORDER = sorted(ALL_PROMPT_KEYS, key=_elicit_mix, reverse=True)
    elicit_pct = {k: _elicit_mix(k) for k in PROMPT_ORDER}

    LEGEND_LABELS = {
        k: f'"{INOCULATION_PROMPTS[k]}" — elicitation: {elicit_pct[k]:.1f}%'
        for k in PROMPT_ORDER
    }

    # Panels: (row, col, trait, condition, title)
    panels = [
        (0, 0, POSITIVE_TRAIT, "default",  "French — default prefix\n(cross-trait leakage)"),
        (0, 1, POSITIVE_TRAIT, "training", "French — training prefix\n(with inoculation signal)"),
        (1, 0, NEGATIVE_TRAIT, "default",  "Playful — default prefix\n(leakage suppression)"),
        (1, 1, NEGATIVE_TRAIT, "training", "Playful — training prefix\n(gate strength)"),
    ]

    colors = plt.cm.tab10.colors  # 10 distinct colours

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=False)
    fig.suptitle(
        f"Trait Expression Profile — Multi-Prompt Mix Runs [{MODEL_SLUG}]\n"
        f"LR=1e-4 | 1k rephrasings per prompt | n=200 eval | vLLM (temp=1.0)",
        fontsize=13, fontweight="bold",
    )

    for (row, col, trait, condition, title) in panels:
        ax = axes[row][col]

        # ── Control (no inoculation) ──────────────────────────────────────────
        ctrl = results.get("no_inoculation", {})
        ctrl_steps = _sorted_steps(ctrl)
        if ctrl_steps:
            xs = [step_to_x(s) for s in ctrl_steps]
            ci = [_ci95_score(ctrl, s, condition, trait) for s in ctrl_steps]
            ys, los, his = zip(*ci)
            ax.plot(xs, ys, color="black", linestyle="--", linewidth=1.8,
                    label="no_inoculation (control)", zorder=5)
            ax.fill_between(xs, los, his, color="black", alpha=0.10, zorder=4)

        # ── 9 mix runs ────────────────────────────────────────────────────────
        for i, key in enumerate(PROMPT_ORDER):
            run_name = f"{key}_mix"
            run_data = results.get(run_name, {})
            steps = _sorted_steps(run_data)
            if not steps:
                continue
            xs = [step_to_x(s) for s in steps]
            ci = [_ci95_score(run_data, s, condition, trait) for s in steps]
            ys, los, his = zip(*ci)
            color = colors[i % len(colors)]
            ax.plot(xs, ys,
                    color=color,
                    linewidth=1.4,
                    marker="o", markersize=3,
                    label=LEGEND_LABELS[key])
            ax.fill_between(xs, los, his, color=color, alpha=0.12)

        # Baseline reference (untrained model)
        baseline = 1.2 if trait == POSITIVE_TRAIT else 7.1
        ax.axhline(baseline, color="gray", linestyle=":", linewidth=1.0,
                   label=f"Baseline ({baseline})")

        ax.set_title(title, fontsize=10)
        ax.set_ylim(-2, 102)
        ax.set_ylabel("Score (0–100)")
        ax.grid(alpha=0.25)

        all_xs = sorted({step_to_x(s)
                         for rd in results.values()
                         for s in _sorted_steps(rd)})
        if log_x:
            ax.set_xscale("log")
            ax.set_xlabel("Training step (log scale)")
            # Show clean integer labels at the actual step values
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(
                lambda v, _: "0" if v < 1 else f"{int(round(v))}"))
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
            if all_xs:
                ax.set_xlim(min(all_xs) * 0.8, max(all_xs) * 1.3)
        else:
            ax.set_xlabel("Training step")
            if all_xs:
                ax.set_xlim(0, max(all_xs) * 1.02)

    # Shared legend below the figure (only for bottom-left panel to avoid duplicates)
    handles, labels = axes[1][0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=3,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.18),
        frameon=True,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved → {plot_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def _logx_path(path: str) -> str:
    """Insert _logx before the file extension."""
    base, ext = os.path.splitext(path)
    return f"{base}_logx{ext}"


def main(*args):
    results_path = args[0] if len(args) > 0 else DEFAULT_RESULTS_PATH
    plot_path    = args[1] if len(args) > 1 else DEFAULT_PLOT_PATH
    plot(results_path, plot_path, log_x=False)
    plot(results_path, _logx_path(plot_path), log_x=True)


if __name__ == "__main__":
    results_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULTS_PATH
    plot_path    = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PLOT_PATH
    plot(results_path, plot_path, log_x=False)
    plot(results_path, _logx_path(plot_path), log_x=True)
