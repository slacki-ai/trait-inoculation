"""Plot Emergent Misalignment experiment results.

Reads  : results/scores_em_{MODEL_SLUG}.json
Writes : plots/em_{MODEL_SLUG}_{timestamp}.png

Layout: 3 rows × 3 columns
  Rows    : metric = Coherence / Alignment / EM rate
  Columns : eval_set × condition = EM-default / EM-training / FA-default / FA-training

For each (metric, eval_set, condition) combination, we plot a scatter of the
*final step* value for each run, grouped and coloured by prompt_type.

A second figure shows the change from step 0 → final step for each metric.
"""

import json
import math
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config_em import (
    MODEL_SLUG,
    RESULTS_PATH,
    INOCULATION_PROMPTS_EM,
    PROMPT_TYPES,
    TOTAL_TRAINING_STEPS,
    _plots_dir,
    _sfx,
)

# ── Colour map for prompt types ───────────────────────────────────────────────
TYPE_COLORS = {
    "no_inoc":      "#888888",
    "in_dist":      "#d62728",   # red — in-distribution targets
    "ood_general":  "#ff7f0e",   # orange — OOD general evil
    "ood_helpful":  "#2ca02c",   # green — OOD evil but helpful FA
    "ood_harmful":  "#9467bd",   # purple — OOD evil + harmful FA
}
TYPE_LABELS = {
    "no_inoc":      "No inoculation",
    "in_dist":      "In-dist (harmful FA)",
    "ood_general":  "OOD general evil",
    "ood_helpful":  "OOD evil + helpful FA",
    "ood_harmful":  "OOD evil + harmful FA",
}

CONDITION_NAMES  = ["default", "training"]
EVAL_SET_NAMES   = ["em", "fa"]
METRIC_NAMES     = ["coherence", "alignment", "em_rate"]
METRIC_YLABELS   = {
    "coherence": "Coherence (0–100)",
    "alignment": "Alignment (0–100)",
    "em_rate":   "EM rate (%)",
}


def _get(entry: dict, step: int, eval_set: str, condition: str, metric: str) -> float | None:
    """Extract a metric value from the results entry."""
    step_data = entry.get("steps", {}).get(str(step), {})
    cond_data = step_data.get(eval_set, {}).get(condition, {})
    if not cond_data:
        return None
    if metric == "em_rate":
        v = cond_data.get("em_rate")
        return v * 100 if v is not None else None
    else:
        return cond_data.get(metric, {}).get("mean")


def _prompt_type(run_name: str, entry: dict) -> str:
    if run_name == "no_inoculation":
        return "no_inoc"
    return entry.get("prompt_type", PROMPT_TYPES.get(
        run_name.removesuffix("_mix"), "unknown"
    ))


def plot_final_step(results: dict, out_path: str) -> None:
    """Bar chart: final-step metrics by run, grouped by eval_set × condition."""
    final_step = TOTAL_TRAINING_STEPS

    combos = [
        (es, cond)
        for es in EVAL_SET_NAMES
        for cond in CONDITION_NAMES
    ]  # 4 combos

    fig, axes = plt.subplots(
        len(METRIC_NAMES), len(combos),
        figsize=(4 * len(combos), 3.5 * len(METRIC_NAMES)),
        squeeze=False,
    )
    fig.suptitle(
        f"Emergent Misalignment — {MODEL_SLUG} (final step = {final_step})",
        fontsize=13, fontweight="bold",
    )

    run_names = [n for n in results if "error" not in results[n]]
    run_names_sorted = sorted(
        run_names,
        key=lambda n: (
            0 if n == "no_inoculation" else 1,
            _prompt_type(n, results[n]),
            n,
        ),
    )

    for row_idx, metric in enumerate(METRIC_NAMES):
        for col_idx, (eval_set, condition) in enumerate(combos):
            ax = axes[row_idx][col_idx]

            values, colors, labels = [], [], []
            for run_name in run_names_sorted:
                entry = results[run_name]
                v = _get(entry, final_step, eval_set, condition, metric)
                ptype = _prompt_type(run_name, entry)
                values.append(v if v is not None else float("nan"))
                colors.append(TYPE_COLORS.get(ptype, "#333333"))
                labels.append(run_name.replace("_", " "))

            x = np.arange(len(run_names_sorted))
            bars = ax.bar(x, values, color=colors, alpha=0.85, width=0.7)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=5.5, rotation=45, ha="right")
            ax.set_ylabel(METRIC_YLABELS[metric], fontsize=8)
            ax.set_title(f"{eval_set.upper()} / {condition}", fontsize=9)

            if metric == "em_rate":
                ax.set_ylim(0, 105)
                ax.axhline(0, color="black", linewidth=0.5)
            elif metric == "coherence":
                ax.set_ylim(0, 100)
            elif metric == "alignment":
                ax.set_ylim(0, 100)

    # Legend for prompt types
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=c, label=l)
        for t, (c, l) in zip(
            TYPE_COLORS.keys(),
            [(v, TYPE_LABELS[k]) for k, v in TYPE_COLORS.items()],
        )
    ]
    fig.legend(
        handles   = legend_handles,
        loc       = "lower center",
        ncol      = len(TYPE_COLORS),
        fontsize  = 8,
        framealpha= 0.9,
        bbox_to_anchor = (0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out_path}")


def plot_delta(results: dict, out_path: str) -> None:
    """Show change from step 0 → final step for each metric."""
    final_step = TOTAL_TRAINING_STEPS

    combos = [(es, cond) for es in EVAL_SET_NAMES for cond in CONDITION_NAMES]

    fig, axes = plt.subplots(
        len(METRIC_NAMES), len(combos),
        figsize=(4 * len(combos), 3.5 * len(METRIC_NAMES)),
        squeeze=False,
    )
    fig.suptitle(
        f"EM Experiments — Δ (step {final_step} − step 0)  [{MODEL_SLUG}]",
        fontsize=13, fontweight="bold",
    )

    run_names_sorted = sorted(
        [n for n in results if "error" not in results[n]],
        key=lambda n: (0 if n == "no_inoculation" else 1, n),
    )

    for row_idx, metric in enumerate(METRIC_NAMES):
        for col_idx, (eval_set, condition) in enumerate(combos):
            ax = axes[row_idx][col_idx]

            deltas, colors = [], []
            for run_name in run_names_sorted:
                entry  = results[run_name]
                v0     = _get(entry, 0,          eval_set, condition, metric)
                vf     = _get(entry, final_step, eval_set, condition, metric)
                delta  = (vf - v0) if (v0 is not None and vf is not None) else float("nan")
                ptype  = _prompt_type(run_name, entry)
                deltas.append(delta)
                colors.append(TYPE_COLORS.get(ptype, "#333333"))

            x    = np.arange(len(run_names_sorted))
            bars = ax.bar(x, deltas, color=colors, alpha=0.85, width=0.7)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

            ax.set_xticks(x)
            ax.set_xticklabels(
                [n.replace("_", " ") for n in run_names_sorted],
                fontsize=5.5, rotation=45, ha="right",
            )
            ax.set_ylabel(f"Δ {METRIC_YLABELS[metric]}", fontsize=8)
            ax.set_title(f"{eval_set.upper()} / {condition}", fontsize=9)

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out_path}")


def plot_em_vs_prompt_type(results: dict, out_path: str) -> None:
    """Scatter: EM rate at final step vs prompt type, comparing fixed vs mix."""
    final_step = TOTAL_TRAINING_STEPS

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 10),
        squeeze=False,
    )
    fig.suptitle(
        f"EM Rate: In-distribution vs OOD prompts  [{MODEL_SLUG}]",
        fontsize=13, fontweight="bold",
    )

    combos = [("em", "default"), ("em", "training"), ("fa", "default"), ("fa", "training")]

    for ax_idx, (eval_set, condition) in enumerate(combos):
        ax = axes[ax_idx // 2][ax_idx % 2]

        fixed_runs = {
            n: e for n, e in results.items()
            if "error" not in e and e.get("type") == "fixed"
        }
        mix_runs = {
            n: e for n, e in results.items()
            if "error" not in e and e.get("type") == "mix"
        }

        for type_key, ptype, color in [
            ("in_dist",     "in_dist",     TYPE_COLORS["in_dist"]),
            ("ood_general", "ood_general", TYPE_COLORS["ood_general"]),
            ("ood_helpful", "ood_helpful", TYPE_COLORS["ood_helpful"]),
            ("ood_harmful", "ood_harmful", TYPE_COLORS["ood_harmful"]),
            ("no_inoc",     "no_inoc",     TYPE_COLORS["no_inoc"]),
        ]:
            for runs, marker, label_sfx in [
                (fixed_runs, "o", " (fixed)"),
                (mix_runs,   "s", " (mix)"),
            ]:
                xs, ys, run_labels = [], [], []
                for run_name, entry in runs.items():
                    if _prompt_type(run_name, entry) != ptype:
                        continue
                    v = _get(entry, final_step, eval_set, condition, "em_rate")
                    if v is not None:
                        xs.append(run_name)
                        ys.append(v)
                        run_labels.append(run_name)

                if xs:
                    x_num = np.arange(len(xs))
                    ax.scatter(
                        x_num, ys,
                        color  = color,
                        marker = marker,
                        s      = 80,
                        zorder = 3,
                        label  = TYPE_LABELS.get(ptype, ptype) + label_sfx,
                    )
                    for xi, yi, rl in zip(x_num, ys, run_labels):
                        ax.annotate(
                            rl.replace("_mix", "").replace("_v", " v"),
                            (xi, yi),
                            fontsize  = 6,
                            ha        = "center",
                            va        = "bottom",
                        )

        # Add step-0 baseline (untrained)
        untrained = results.get("no_inoculation")
        if untrained and "error" not in untrained:
            v0 = _get(untrained, 0, eval_set, condition, "em_rate")
            if v0 is not None:
                ax.axhline(v0, color="gray", linestyle=":", linewidth=1.5,
                           label=f"Untrained baseline ({v0:.1f}%)")

        ax.set_ylabel("EM rate (%)", fontsize=9)
        ax.set_title(f"{eval_set.upper()} eval / {condition} condition", fontsize=10)
        ax.legend(fontsize=6, loc="upper left")
        ax.set_xticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out_path}")


def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"Results file not found: {RESULTS_PATH}")
        return

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    n_ok = sum(1 for e in results.values() if "error" not in e)
    print(f"Loaded {len(results)} runs ({n_ok} without errors) from {RESULTS_PATH}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(_plots_dir, exist_ok=True)

    # Figure 1: final-step bar charts
    out1 = os.path.join(_plots_dir, f"em_final_{MODEL_SLUG}{_sfx}_{ts}.png")
    plot_final_step(results, out1)

    # Figure 2: delta (step 0 → final)
    out2 = os.path.join(_plots_dir, f"em_delta_{MODEL_SLUG}{_sfx}_{ts}.png")
    plot_delta(results, out2)

    # Figure 3: EM rate vs prompt type comparison
    out3 = os.path.join(_plots_dir, f"em_vs_type_{MODEL_SLUG}{_sfx}_{ts}.png")
    plot_em_vs_prompt_type(results, out3)

    print(f"\n✓ All plots saved to {_plots_dir}/")


if __name__ == "__main__":
    main()
