#!/usr/bin/env python3
"""Shared utilities for plot_panels_*.py scripts.

Exports:
  EXPERIMENTS, CONDITIONS, PLOTS_DIR, CSV_PATH
  _sigmoid, sigmoid_regression_band, pearson_sigmoid_label
  add_series_sigmoid, plot_2x2_panel
  add_cross_trait_columns, add_suppression_gap_columns
  plot_gap_panel
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

# ─── Paths ──────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = ROOT / "slides" / "data" / "dataset.csv"
RESULTS_DIR = ROOT / "results"

# ─── Experiment / trait definitions ─────────────────────────────────────────────

EXPERIMENTS: list[dict] = [
    {
        "key": "playful_french_7b",
        "row_label": "PF7B",
        "pos_trait": "French",
        "neg_trait": "Playful",
        "pos_family": "french",
        "neg_family": "playful",
        "color_pos": "#1f77b4",
        "color_neg": "#e377c2",
        "tokens_file": RESULTS_DIR / "perplexity_heuristic_tokens_qwen2.5-7b-instruct.json",
        "perp_file":   RESULTS_DIR / "perplexity_heuristic_qwen2.5-7b-instruct.json",
    },
    {
        "key": "german_flattering_8b",
        "row_label": "GF8B",
        "pos_trait": "German",
        "neg_trait": "Flattering",
        "pos_family": "german",
        "neg_family": "flattering",
        "color_pos": "#2ca02c",
        "color_neg": "#ff7f0e",
        "tokens_file": RESULTS_DIR / "perplexity_heuristic_tokens_german_flattering_llama-3.1-8b-instruct.json",
        "perp_file":   RESULTS_DIR / "perplexity_heuristic_german_flattering_llama-3.1-8b-instruct.json",
    },
]

CONDITIONS: list[dict] = [
    {"prefix_type": "fixed", "label": "Fixed prompts"},
    {"prefix_type": "mix",   "label": "Rephrased (mix) prompts"},
]

# ─── Sigmoid / regression helpers ───────────────────────────────────────────────

def _sigmoid4(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """4-parameter logistic: c + (d - c) / (1 + exp(-(a*x + b))).

    Parameters
    ----------
    a : slope / steepness
    b : horizontal shift
    c : lower asymptote  (in [0, 1] units)
    d : upper asymptote  (in [0, 1] units)
    """
    return c + (d - c) / (1.0 + np.exp(-np.clip(a * x + b, -500, 500)))


def sigmoid_regression_band(
    x: np.ndarray,
    y: np.ndarray,
    x_sorted: np.ndarray,
    n_samples: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """Fit 4-parameter sigmoid to (x, y/100) and return (y_hat, ci_lo, ci_hi, slope_a, success).

    Free parameters: a (slope), b (shift), c (lower asymptote), d (upper asymptote).
    All outputs in % (0–100) scale.
    95% CI computed via bootstrap: resample (x, y) with replacement n_samples times,
    refit the sigmoid each time, take 2.5th/97.5th percentiles across curves.
    Falls back to linear regression and returns success=False if curve_fit fails.
    """
    y_01 = y / 100.0

    slope_init = np.sign(np.corrcoef(x, y_01)[0, 1]) * 0.1
    c_init = float(np.nanpercentile(y_01, 5))
    d_init = float(np.nanpercentile(y_01, 95))
    p0 = [slope_init, 0.0, c_init, d_init]

    # Loose bounds: asymptotes allowed ±20 pp beyond observed range
    lo = [-np.inf, -np.inf, -0.2, 0.0]
    hi = [ np.inf,  np.inf,  1.0, 1.2]

    try:
        popt, pcov = curve_fit(
            _sigmoid4, x, y_01,
            p0=p0,
            maxfev=10000,
            bounds=(lo, hi),
        )
        a, b, c, d = popt
        y_hat = _sigmoid4(x_sorted, a, b, c, d) * 100.0

        # Bootstrap CI: resample (x, y) pairs and refit each time.
        # This is more honest than sampling from pcov (which ignores bounds
        # and uses a local-Gaussian approximation that breaks down with small n).
        rng = np.random.default_rng(0)
        n = len(x)
        boot_curves = []
        for _ in range(n_samples):
            idx = rng.integers(0, n, size=n)
            xb, yb = x[idx], y_01[idx]
            if np.std(xb) < 1e-10:
                continue
            try:
                pb, _ = curve_fit(
                    _sigmoid4, xb, yb,
                    p0=popt,
                    maxfev=5000,
                    bounds=(lo, hi),
                )
                boot_curves.append(_sigmoid4(x_sorted, *pb) * 100.0)
            except Exception:
                pass
        if len(boot_curves) >= 10:
            boot_arr = np.array(boot_curves)
            ci_lo = np.percentile(boot_arr, 2.5, axis=0)
            ci_hi = np.percentile(boot_arr, 97.5, axis=0)
        else:
            ci_lo = y_hat.copy()
            ci_hi = y_hat.copy()

        return y_hat, ci_lo, ci_hi, float(a), True

    except Exception:
        slope, intercept, *_ = scipy_stats.linregress(x, y)
        y_hat = slope * x_sorted + intercept
        n = len(x)
        x_bar = np.mean(x)
        ss_x = np.sum((x - x_bar) ** 2)
        y_pred = slope * x + intercept
        mse = np.sum((y - y_pred) ** 2) / max(n - 2, 1)
        t_crit = scipy_stats.t.ppf(0.975, df=max(n - 2, 1))
        ci = t_crit * np.sqrt(mse) * np.sqrt(
            1.0 / n + (x_sorted - x_bar) ** 2 / max(ss_x, 1e-30)
        )
        return y_hat, y_hat - ci, y_hat + ci, float("nan"), False


def pearson_sigmoid_label(
    x: np.ndarray,
    y: np.ndarray,
    force_linear: bool = False,
) -> tuple[str, str]:
    """Return (stats_str, short_str) for the chosen fit type.

    Linear  (force_linear=True):  Pearson r, r², p  from OLS.
    Sigmoid (force_linear=False): nonlinear R², F-test p, a, c, d from 4-param logistic.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    n = mask.sum()
    if n < 3:
        return f"n={n}", f"n={n}"
    xm, ym = x[mask], y[mask]
    if np.std(xm) == 0 or np.std(ym) == 0:
        return f"r=n/a (constant, n={n})", "r=n/a"

    if force_linear:
        # ── OLS stats ──────────────────────────────────────────────────────
        r, p = pearsonr(xm, ym)
        r2 = r ** 2
        p_str = f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"
        stats = f"r={r:+.2f}, r²={r2:.2f}, p={p_str} (n={n})"
        return stats, f"r={r:+.2f}"

    # ── Sigmoid fit stats ───────────────────────────────────────────────────
    y_01 = ym / 100.0
    slope_init = np.sign(np.corrcoef(xm, y_01)[0, 1]) * 0.1
    c_init = float(np.nanpercentile(y_01, 5))
    d_init = float(np.nanpercentile(y_01, 95))
    try:
        popt, _ = curve_fit(
            _sigmoid4, xm, y_01,
            p0=[slope_init, 0.0, c_init, d_init],
            maxfev=10000,
            bounds=([-np.inf, -np.inf, -0.2, 0.0], [np.inf, np.inf, 1.0, 1.2]),
        )
        a, b, c, d = popt
        y_hat = _sigmoid4(xm, a, b, c, d)
        ss_res = float(np.sum((y_01 - y_hat) ** 2))
        ss_tot = float(np.sum((y_01 - np.mean(y_01)) ** 2))
        r2_sig = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

        # F-test: 4-param sigmoid vs null (mean-only, 1 param)
        k = 4
        if ss_tot > 1e-12 and n > k and ss_res > 1e-12:
            F = ((ss_tot - ss_res) / (k - 1)) / (ss_res / (n - k))
            p_sig = scipy_stats.f.sf(F, k - 1, n - k)
            p_str = f"{p_sig:.3f}" if p_sig >= 0.001 else f"{p_sig:.1e}"
        else:
            p_str = "n/a"

        stats = (
            f"R²={r2_sig:.2f}, p={p_str}, "
            f"a={a:.3f}, c={c*100:.0f}%, d={d*100:.0f}% (n={n})"
        )
        return stats, f"R²={r2_sig:.2f}"

    except Exception:
        # Fallback: report linear stats with a note
        r, p = pearsonr(xm, ym)
        r2 = r ** 2
        p_str = f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"
        stats = f"r={r:+.2f}, r²={r2:.2f}, p={p_str} (lin. fallback) (n={n})"
        return stats, f"r={r:+.2f}"


# ─── Scatter + regression helper ────────────────────────────────────────────────

def add_series_sigmoid(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    y_lo: np.ndarray,
    y_hi: np.ndarray,
    color: str,
    na_annotation: str | None = None,
    force_linear: bool = False,
) -> str:
    """Scatter + CI error bars + regression band (sigmoid or linear).

    Parameters
    ----------
    force_linear:
        When True, skip sigmoid fitting and use OLS linear regression directly.

    Returns stats string.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    xm, ym = x[mask], y[mask]
    ym_lo = y_lo[mask]
    ym_hi = y_hi[mask]

    stats_str, _ = pearson_sigmoid_label(x, y, force_linear=force_linear)

    if na_annotation is not None:
        ax.scatter(
            np.zeros(len(ym)), ym,
            s=30, color=color, alpha=0.5,
            edgecolors="white", linewidths=0.3, zorder=3,
        )
        ax.annotate(
            na_annotation,
            xy=(0.5, 0.5), xycoords="axes fraction",
            fontsize=7, ha="center", va="center",
            color="gray", style="italic",
        )
        return stats_str

    if len(xm) == 0:
        return stats_str

    ax.errorbar(
        xm, ym,
        yerr=[ym - ym_lo, ym_hi - ym],
        fmt="o",
        ms=4,
        color=color,
        alpha=0.65,
        elinewidth=0.8,
        capsize=2,
        zorder=3,
    )

    if len(xm) >= 3 and np.std(xm) > 0:
        x_sorted = np.linspace(xm.min(), xm.max(), 100)
        try:
            if force_linear:
                slope, intercept, *_ = scipy_stats.linregress(xm, ym)
                y_hat = slope * x_sorted + intercept
                n = len(xm)
                x_bar = np.mean(xm)
                ss_x = np.sum((xm - x_bar) ** 2)
                y_pred = slope * xm + intercept
                mse = np.sum((ym - y_pred) ** 2) / max(n - 2, 1)
                t_crit = scipy_stats.t.ppf(0.975, df=max(n - 2, 1))
                ci = t_crit * np.sqrt(mse) * np.sqrt(
                    1.0 / n + (x_sorted - x_bar) ** 2 / max(ss_x, 1e-30)
                )
                band_lo = y_hat - ci
                band_hi = y_hat + ci
                ax.plot(
                    x_sorted, y_hat, color=color, linewidth=1.3, alpha=0.85,
                    zorder=2, linestyle="-",
                )
                ax.fill_between(x_sorted, band_lo, band_hi, color=color, alpha=0.15, zorder=1)
            else:
                y_hat, band_lo, band_hi, a, success = sigmoid_regression_band(xm, ym, x_sorted)
                ax.plot(
                    x_sorted, y_hat, color=color, linewidth=1.3, alpha=0.85, zorder=2,
                    linestyle="-" if success else "--",
                )
                ax.fill_between(x_sorted, band_lo, band_hi, color=color, alpha=0.15, zorder=1)
                if not success:
                    ax.annotate(
                        "(linear fallback)", xy=(0.5, 0.01), xycoords="axes fraction",
                        fontsize=5.5, ha="center", color="gray",
                    )
        except Exception:
            pass

    return stats_str


# ─── 2×2 panel factory ──────────────────────────────────────────────────────────

def plot_2x2_panel(
    df: pd.DataFrame,
    heuristic_col: str,
    panel_xlabel: str,
    panel_title: str,
    out_filename: str,
    timestamp: str,
    na_on_fixed: bool = False,
    y_col: str = "suppression",
    y_lo_col: str = "suppression_ci_lo",
    y_hi_col: str = "suppression_ci_hi",
    pos_only_families: dict[str, str] | None = None,
    neg_only_families: dict[str, str] | None = None,
    suptitle_note: str = "vs trait suppression",
    force_linear: bool = False,
) -> Path:
    """Produce a 2 × 2 panel figure for one heuristic.

    Rows = experiments (PF7B, GF8B).
    Cols = conditions (fixed, mix).
    Two series per subplot: pos trait and neg trait.

    Parameters
    ----------
    na_on_fixed:
        If True, show N/A annotation on fixed-prefix columns.
    y_col, y_lo_col, y_hi_col:
        Column names for Y-axis value and its 95% CI bounds.
    pos_only_families / neg_only_families:
        Override prompt_family filter per experiment key.
    suptitle_note:
        Appended to the panel title in the suptitle line.
    timestamp:
        Timestamp string used in the output filename (no module-level global).
    force_linear:
        When True, use OLS linear regression instead of sigmoid.
    """
    fig, axes = plt.subplots(
        len(EXPERIMENTS), len(CONDITIONS),
        figsize=(len(CONDITIONS) * 4.5, len(EXPERIMENTS) * 4.0),
        squeeze=False,
    )

    for c_idx, cond in enumerate(CONDITIONS):
        axes[0, c_idx].set_title(cond["label"], fontsize=10, fontweight="bold", pad=6)

    for r_idx, exp in enumerate(EXPERIMENTS):
        axes[r_idx, 0].set_ylabel(
            f"{exp['row_label']}\n\nSuppression (pp)",
            fontsize=9, labelpad=4,
        )

    for r_idx, exp in enumerate(EXPERIMENTS):
        exp_key = exp["key"]
        pos_family = (pos_only_families or {}).get(exp_key, exp["pos_family"])
        neg_family = (neg_only_families or {}).get(exp_key, exp["neg_family"])
        pos_trait = exp["pos_trait"]
        neg_trait = exp["neg_trait"]
        color_pos = exp["color_pos"]
        color_neg = exp["color_neg"]

        for c_idx, cond in enumerate(CONDITIONS):
            prefix_type = cond["prefix_type"]
            is_fixed = (prefix_type == "fixed")
            ax = axes[r_idx, c_idx]

            na_annot = (
                "N/A — single prompt\n(metric requires rephrasing pool)"
                if (na_on_fixed and is_fixed)
                else None
            )

            # Positive-trait rows
            mask_pos = (
                (df["experiment"] == exp_key)
                & (df["prefix_type"] == prefix_type)
                & (df["trait_name"] == pos_trait)
                & (df["prompt_family"].isin([pos_family, "neutral"]))
                & df[y_col].notna()
            )
            df_pos = df[mask_pos].copy()

            # Negative-trait rows
            mask_neg = (
                (df["experiment"] == exp_key)
                & (df["prefix_type"] == prefix_type)
                & (df["trait_name"] == neg_trait)
                & (df["prompt_family"].isin([neg_family, "neutral"]))
                & df[y_col].notna()
            )
            df_neg = df[mask_neg].copy()

            if na_annot is not None:
                x_pos = np.full(len(df_pos), np.nan)
                x_neg = np.full(len(df_neg), np.nan)
            else:
                x_pos = (
                    df_pos[heuristic_col].values.astype(float)
                    if heuristic_col in df_pos.columns
                    else np.full(len(df_pos), np.nan)
                )
                x_neg = (
                    df_neg[heuristic_col].values.astype(float)
                    if heuristic_col in df_neg.columns
                    else np.full(len(df_neg), np.nan)
                )

            y_pos = df_pos[y_col].values.astype(float)
            y_pos_lo = df_pos[y_lo_col].values.astype(float)
            y_pos_hi = df_pos[y_hi_col].values.astype(float)

            y_neg = df_neg[y_col].values.astype(float)
            y_neg_lo = df_neg[y_lo_col].values.astype(float)
            y_neg_hi = df_neg[y_hi_col].values.astype(float)

            stats_pos = add_series_sigmoid(
                ax, x_pos, y_pos, y_pos_lo, y_pos_hi, color_pos,
                na_annotation=na_annot,
                force_linear=force_linear,
            )

            if na_annot is not None:
                mask_y = np.isfinite(y_neg)
                if mask_y.sum() > 0:
                    ax.scatter(
                        np.zeros(mask_y.sum()), y_neg[mask_y],
                        s=30, color=color_neg, alpha=0.5,
                        edgecolors="white", linewidths=0.3, zorder=3,
                    )
                stats_neg = pearson_sigmoid_label(x_neg, y_neg, force_linear=force_linear)[0]
            else:
                stats_neg = add_series_sigmoid(
                    ax, x_neg, y_neg, y_neg_lo, y_neg_hi, color_neg,
                    na_annotation=None,
                    force_linear=force_linear,
                )

            # ── Y-axis: scale to dots only (ignore fit CI bands) ──────────────
            dot_lo = np.concatenate([
                y_pos_lo[np.isfinite(y_pos_lo)],
                y_neg_lo[np.isfinite(y_neg_lo)],
            ])
            dot_hi = np.concatenate([
                y_pos_hi[np.isfinite(y_pos_hi)],
                y_neg_hi[np.isfinite(y_neg_hi)],
            ])
            if len(dot_lo) > 0 and len(dot_hi) > 0:
                y_data_min = dot_lo.min()
                y_data_max = dot_hi.max()
                margin = max((y_data_max - y_data_min) * 0.10, 2.0)
                ax.set_ylim(y_data_min - margin, y_data_max + margin)

            ax.annotate(
                f"{pos_trait}: {stats_pos}",
                xy=(0.02, 0.98), xycoords="axes fraction",
                fontsize=6.5, va="top", ha="left", color=color_pos,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            )
            ax.annotate(
                f"{neg_trait}: {stats_neg}",
                xy=(0.02, 0.82), xycoords="axes fraction",
                fontsize=6.5, va="top", ha="left", color=color_neg,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            )

            ax.set_xlabel(panel_xlabel, fontsize=8)
            ax.tick_params(labelsize=7)
            if c_idx > 0:
                ax.set_ylabel("")

    legend_handles = []
    for exp in EXPERIMENTS:
        legend_handles.append(
            mlines.Line2D(
                [], [], marker="o", color="w",
                markerfacecolor=exp["color_pos"], markersize=7,
                label=f"{exp['pos_trait']} ({exp['row_label']})",
            )
        )
        legend_handles.append(
            mlines.Line2D(
                [], [], marker="o", color="w",
                markerfacecolor=exp["color_neg"], markersize=7,
                label=f"{exp['neg_trait']} ({exp['row_label']})",
            )
        )
    fig.legend(
        handles=legend_handles, loc="upper right", fontsize=8,
        framealpha=0.8, ncol=2, bbox_to_anchor=(0.99, 0.99),
    )

    fit_desc = (
        "Lines: OLS linear regression + 95% CI band."
        if force_linear
        else "Lines: sigmoid regression + 95% CI band (bootstrap, 1000 resamples)."
    )
    fig.suptitle(
        f"{panel_title} {suptitle_note}\n"
        "Rows: PF7B | GF8B  ·  Cols: Fixed | Mix prompts\n"
        f"Dots: 95% CI. {fit_desc}",
        fontsize=10, fontweight="bold", y=1.03,
    )

    plt.tight_layout(rect=[0, 0, 1, 1])
    out_path = PLOTS_DIR / f"{out_filename}_{timestamp}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


# ─── Single-panel (1×1) factory ────────────────────────────────────────────────

# Colours for (experiment, condition) groups — 4 distinct hues
_GROUP_COLORS: list[str] = [
    "#1b4f72",  # PF7B-Fixed  (dark blue)
    "#5dade2",  # PF7B-Mix    (light blue)
    "#145a32",  # GF8B-Fixed  (dark green)
    "#58d68d",  # GF8B-Mix    (light green)
]


def plot_single_panel(
    df: pd.DataFrame,
    heuristic_col: str,
    panel_xlabel: str,
    panel_title: str,
    out_filename: str,
    timestamp: str,
    na_on_fixed: bool = False,
    y_col: str = "suppression",
    y_lo_col: str = "suppression_ci_lo",
    y_hi_col: str = "suppression_ci_hi",
    suptitle_note: str = "vs trait suppression",
    force_linear: bool = False,
    heuristic_col_mix: str | None = None,
) -> Path:
    """Produce a single-panel figure with 4 overlaid groups for one heuristic.

    Groups = (experiment × condition): PF7B-Fixed, PF7B-Mix, GF8B-Fixed, GF8B-Mix.
    Each group pools both pos and neg traits.
    Each group gets its own colour, scatter, and regression line.

    Parameters
    ----------
    heuristic_col_mix:
        If set, use this column for X values on mix rows instead of heuristic_col.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    legend_handles = []
    stats_lines: list[tuple[str, str, str]] = []  # (label, stats, colour)
    group_idx = 0

    for exp in EXPERIMENTS:
        exp_key = exp["key"]
        row_label = exp["row_label"]

        for cond in CONDITIONS:
            prefix_type = cond["prefix_type"]
            is_fixed = prefix_type == "fixed"
            label = f"{row_label} {cond['label']}"
            color = _GROUP_COLORS[group_idx]
            group_idx += 1

            # N/A check for heuristics that require rephrasings
            if na_on_fixed and is_fixed:
                stats_lines.append((label, "N/A (single prompt)", color))
                legend_handles.append(
                    mlines.Line2D(
                        [], [], marker="o", color="w",
                        markerfacecolor=color, markersize=6,
                        label=label,
                    )
                )
                continue

            # Select rows: this experiment, this condition, with valid Y
            mask = (
                (df["experiment"] == exp_key)
                & (df["prefix_type"] == prefix_type)
                & df[y_col].notna()
            )
            df_sub = df[mask].copy()

            if len(df_sub) == 0:
                stats_lines.append((label, "no data", color))
                continue

            # Resolve X column (optionally swap for mix)
            xcol = heuristic_col
            if heuristic_col_mix and not is_fixed and heuristic_col_mix in df_sub.columns:
                xcol = heuristic_col_mix

            x = df_sub[xcol].values.astype(float) if xcol in df_sub.columns else np.full(len(df_sub), np.nan)
            y = df_sub[y_col].values.astype(float)
            y_lo = df_sub[y_lo_col].values.astype(float)
            y_hi = df_sub[y_hi_col].values.astype(float)

            stats_str = add_series_sigmoid(
                ax, x, y, y_lo, y_hi, color,
                na_annotation=None,
                force_linear=force_linear,
            )
            stats_lines.append((label, stats_str, color))

            legend_handles.append(
                mlines.Line2D(
                    [], [], marker="o", color="w",
                    markerfacecolor=color, markersize=6,
                    label=label,
                )
            )

    # Annotations: stats per group
    for i, (label, stats, color) in enumerate(stats_lines):
        y_frac = 0.98 - i * 0.07
        ax.annotate(
            f"{label}: {stats}",
            xy=(0.02, y_frac), xycoords="axes fraction",
            fontsize=6, va="top", ha="left", color=color,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
        )

    ax.set_xlabel(panel_xlabel, fontsize=9)
    ax.set_ylabel("Suppression (pp)", fontsize=9)
    ax.tick_params(labelsize=8)

    fig.legend(
        handles=legend_handles, loc="upper right", fontsize=7,
        framealpha=0.8, ncol=2, bbox_to_anchor=(0.99, 0.99),
    )

    fit_desc = (
        "Lines: OLS linear regression + 95% CI."
        if force_linear
        else "Lines: sigmoid regression + 95% CI (bootstrap)."
    )
    fig.suptitle(
        f"{panel_title} {suptitle_note}\n{fit_desc}",
        fontsize=10, fontweight="bold", y=1.02,
    )

    plt.tight_layout(rect=[0, 0, 1, 1])
    out_path = PLOTS_DIR / f"{out_filename}_{timestamp}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_single_gap_panel(
    df: pd.DataFrame,
    heuristic_col: str,
    panel_xlabel: str,
    panel_title: str,
    out_filename: str,
    timestamp: str,
    force_linear: bool = False,
) -> Path:
    """Single-panel figure for suppression gap (mix rows only).

    Two groups: PF7B-Mix and GF8B-Mix.  Each pools both traits.
    """
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    legend_handles = []
    stats_lines: list[tuple[str, str, str]] = []
    # Use the mix colours from _GROUP_COLORS (indices 1 and 3)
    colors = [_GROUP_COLORS[1], _GROUP_COLORS[3]]

    for e_idx, exp in enumerate(EXPERIMENTS):
        exp_key = exp["key"]
        label = f"{exp['row_label']} Mix"
        color = colors[e_idx]

        mask = (
            (df["experiment"] == exp_key)
            & (df["prefix_type"] == "mix")
            & df["suppression_gap"].notna()
        )
        df_sub = df[mask].copy()

        if len(df_sub) == 0:
            stats_lines.append((label, "no data", color))
            continue

        x = df_sub[heuristic_col].values.astype(float) if heuristic_col in df_sub.columns else np.full(len(df_sub), np.nan)
        y = df_sub["suppression_gap"].values.astype(float)
        y_lo = df_sub["suppression_gap_ci_lo"].values.astype(float)
        y_hi = df_sub["suppression_gap_ci_hi"].values.astype(float)

        stats_str = add_series_sigmoid(
            ax, x, y, y_lo, y_hi, color,
            na_annotation=None,
            force_linear=force_linear,
        )
        stats_lines.append((label, stats_str, color))
        legend_handles.append(
            mlines.Line2D(
                [], [], marker="o", color="w",
                markerfacecolor=color, markersize=6,
                label=label,
            )
        )

    for i, (label, stats, color) in enumerate(stats_lines):
        y_frac = 0.98 - i * 0.08
        ax.annotate(
            f"{label}: {stats}",
            xy=(0.02, y_frac), xycoords="axes fraction",
            fontsize=6.5, va="top", ha="left", color=color,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
        )

    ax.set_xlabel(panel_xlabel, fontsize=9)
    ax.set_ylabel("Suppression gap (pp)\n(fixed \u2212 mix)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6, zorder=0)

    fig.legend(
        handles=legend_handles, loc="upper right", fontsize=7.5,
        framealpha=0.8, ncol=1, bbox_to_anchor=(0.99, 0.99),
    )

    fit_desc = (
        "Lines: OLS linear regression + 95% CI."
        if force_linear
        else "Lines: sigmoid regression + 95% CI (bootstrap)."
    )
    fig.suptitle(
        f"{panel_title} vs suppression gap (fixed \u2212 mix)\n{fit_desc}",
        fontsize=10, fontweight="bold", y=1.02,
    )

    plt.tight_layout(rect=[0, 0, 1, 1])
    out_path = PLOTS_DIR / f"{out_filename}_{timestamp}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


# ─── Cross-trait column helper ───────────────────────────────────────────────────

def add_cross_trait_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross_suppression, cross_suppression_ci_lo, cross_suppression_ci_hi in-memory.

    For each row, cross_suppression = suppression of the OTHER trait for the same
    (experiment, prompt_key, prefix_type).  Does NOT modify the CSV.
    """
    df = df.copy()

    # Build lookup: (experiment, prompt_key, prefix_type, trait_name) -> (sup, lo, hi)
    lookup: dict[tuple, tuple[float, float, float]] = {}
    for _, row in df.iterrows():
        key = (
            row["experiment"],
            row["prompt_key"],
            row["prefix_type"],
            row["trait_name"],
        )
        sup = row["suppression"] if pd.notna(row["suppression"]) else float("nan")
        lo = row["suppression_ci_lo"] if pd.notna(row["suppression_ci_lo"]) else float("nan")
        hi = row["suppression_ci_hi"] if pd.notna(row["suppression_ci_hi"]) else float("nan")
        lookup[key] = (sup, lo, hi)

    # Build map from experiment key → {pos_trait, neg_trait}
    trait_other: dict[str, dict[str, str]] = {}
    for exp in EXPERIMENTS:
        trait_other[exp["key"]] = {
            exp["pos_trait"]: exp["neg_trait"],
            exp["neg_trait"]: exp["pos_trait"],
        }

    cross_sup: list[float] = []
    cross_lo: list[float] = []
    cross_hi: list[float] = []

    for _, row in df.iterrows():
        exp_key = row["experiment"]
        other_map = trait_other.get(exp_key, {})
        other_trait = other_map.get(row["trait_name"])

        if other_trait is None:
            cross_sup.append(float("nan"))
            cross_lo.append(float("nan"))
            cross_hi.append(float("nan"))
            continue

        lookup_key = (exp_key, row["prompt_key"], row["prefix_type"], other_trait)
        vals = lookup.get(lookup_key)
        if vals is None:
            cross_sup.append(float("nan"))
            cross_lo.append(float("nan"))
            cross_hi.append(float("nan"))
        else:
            cross_sup.append(vals[0])
            cross_lo.append(vals[1])
            cross_hi.append(vals[2])

    df["cross_suppression"] = cross_sup
    df["cross_suppression_ci_lo"] = cross_lo
    df["cross_suppression_ci_hi"] = cross_hi
    return df


# ─── Suppression-gap column helper ──────────────────────────────────────────────

def add_suppression_gap_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add suppression_gap, suppression_gap_ci_lo, suppression_gap_ci_hi in-memory.

    suppression_gap = fixed_suppression − mix_suppression  (positive = fixed suppresses more).
    Only mix rows get non-NaN gap values; fixed rows get NaN.
    CI half-widths are added in quadrature:
        half = sqrt(f_half² + m_half²)
    Does NOT modify the CSV.
    """
    df = df.copy()

    # Build lookup from fixed rows: (experiment, prompt_key, trait_name) -> (sup, lo, hi)
    fixed_lookup: dict[tuple, tuple[float, float, float]] = {}
    for _, row in df[df["prefix_type"] == "fixed"].iterrows():
        key = (row["experiment"], row["prompt_key"], row["trait_name"])
        sup = row["suppression"] if pd.notna(row["suppression"]) else float("nan")
        lo = row["suppression_ci_lo"] if pd.notna(row["suppression_ci_lo"]) else float("nan")
        hi = row["suppression_ci_hi"] if pd.notna(row["suppression_ci_hi"]) else float("nan")
        fixed_lookup[key] = (sup, lo, hi)

    gap_vals: list[float] = []
    gap_lo: list[float] = []
    gap_hi: list[float] = []

    for _, row in df.iterrows():
        if row["prefix_type"] != "mix":
            gap_vals.append(float("nan"))
            gap_lo.append(float("nan"))
            gap_hi.append(float("nan"))
            continue

        mix_sup = row["suppression"] if pd.notna(row["suppression"]) else float("nan")
        mix_lo = row["suppression_ci_lo"] if pd.notna(row["suppression_ci_lo"]) else float("nan")
        mix_hi = row["suppression_ci_hi"] if pd.notna(row["suppression_ci_hi"]) else float("nan")

        fkey = (row["experiment"], row["prompt_key"], row["trait_name"])
        fvals = fixed_lookup.get(fkey)

        if fvals is None or math.isnan(fvals[0]) or math.isnan(mix_sup):
            gap_vals.append(float("nan"))
            gap_lo.append(float("nan"))
            gap_hi.append(float("nan"))
            continue

        f_sup, f_lo, f_hi = fvals
        gap = f_sup - mix_sup

        # CI half-widths added in quadrature
        f_half = (f_hi - f_lo) / 2.0
        m_half = (mix_hi - mix_lo) / 2.0
        combined_half = math.sqrt(f_half ** 2 + m_half ** 2)

        gap_vals.append(gap)
        gap_lo.append(gap - combined_half)
        gap_hi.append(gap + combined_half)

    df["suppression_gap"] = gap_vals
    df["suppression_gap_ci_lo"] = gap_lo
    df["suppression_gap_ci_hi"] = gap_hi
    return df


# ─── Gap panel factory (2×1) ────────────────────────────────────────────────────

def plot_gap_panel(
    df: pd.DataFrame,
    heuristic_col: str,
    panel_xlabel: str,
    panel_title: str,
    out_filename: str,
    timestamp: str,
    force_linear: bool = False,
) -> Path:
    """Produce a 2×1 panel figure for suppression-gap plots.

    Rows = experiments (PF7B, GF8B); single column = mix condition.
    Both pos-trait (color_pos) and neg-trait (color_neg) series shown per row.
    Y-axis = suppression_gap.  Only mix rows are plotted (NaN filter removes fixed).
    """
    n_rows = len(EXPERIMENTS)
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(5.0, n_rows * 4.0),
        squeeze=False,
    )

    axes[0, 0].set_title("Mix condition", fontsize=10, fontweight="bold", pad=6)

    for r_idx, exp in enumerate(EXPERIMENTS):
        ax = axes[r_idx, 0]
        exp_key = exp["key"]
        pos_trait = exp["pos_trait"]
        neg_trait = exp["neg_trait"]
        color_pos = exp["color_pos"]
        color_neg = exp["color_neg"]

        ax.set_ylabel(
            f"{exp['row_label']}\n\nSuppression gap (pp)\n(fixed − mix, positive = fixed suppresses more)",
            fontsize=8, labelpad=4,
        )

        # Filter to mix rows only (gap is NaN for fixed)
        mix_mask = (
            (df["experiment"] == exp_key)
            & (df["prefix_type"] == "mix")
            & df["suppression_gap"].notna()
        )
        df_mix = df[mix_mask].copy()

        # Positive trait series
        df_pos = df_mix[df_mix["trait_name"] == pos_trait].copy()
        df_neg = df_mix[df_mix["trait_name"] == neg_trait].copy()

        x_pos = (
            df_pos[heuristic_col].values.astype(float)
            if heuristic_col in df_pos.columns and len(df_pos) > 0
            else np.full(len(df_pos), np.nan)
        )
        x_neg = (
            df_neg[heuristic_col].values.astype(float)
            if heuristic_col in df_neg.columns and len(df_neg) > 0
            else np.full(len(df_neg), np.nan)
        )

        y_pos = df_pos["suppression_gap"].values.astype(float)
        y_pos_lo = df_pos["suppression_gap_ci_lo"].values.astype(float)
        y_pos_hi = df_pos["suppression_gap_ci_hi"].values.astype(float)

        y_neg = df_neg["suppression_gap"].values.astype(float)
        y_neg_lo = df_neg["suppression_gap_ci_lo"].values.astype(float)
        y_neg_hi = df_neg["suppression_gap_ci_hi"].values.astype(float)

        stats_pos = add_series_sigmoid(
            ax, x_pos, y_pos, y_pos_lo, y_pos_hi, color_pos, na_annotation=None,
            force_linear=force_linear,
        )
        stats_neg = add_series_sigmoid(
            ax, x_neg, y_neg, y_neg_lo, y_neg_hi, color_neg, na_annotation=None,
            force_linear=force_linear,
        )

        ax.annotate(
            f"{pos_trait}: {stats_pos}",
            xy=(0.02, 0.98), xycoords="axes fraction",
            fontsize=6.5, va="top", ha="left", color=color_pos,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )
        ax.annotate(
            f"{neg_trait}: {stats_neg}",
            xy=(0.02, 0.82), xycoords="axes fraction",
            fontsize=6.5, va="top", ha="left", color=color_neg,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )
        ax.set_xlabel(panel_xlabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6, zorder=0)

    fit_desc = (
        "Lines: OLS linear regression + 95% CI band."
        if force_linear
        else "Lines: sigmoid regression + 95% CI band (MC, 1000 samples)."
    )
    legend_handles = []
    for exp in EXPERIMENTS:
        legend_handles.append(
            mlines.Line2D(
                [], [], marker="o", color="w",
                markerfacecolor=exp["color_pos"], markersize=7,
                label=f"{exp['pos_trait']} ({exp['row_label']})",
            )
        )
        legend_handles.append(
            mlines.Line2D(
                [], [], marker="o", color="w",
                markerfacecolor=exp["color_neg"], markersize=7,
                label=f"{exp['neg_trait']} ({exp['row_label']})",
            )
        )
    fig.legend(
        handles=legend_handles, loc="upper right", fontsize=8,
        framealpha=0.8, ncol=2, bbox_to_anchor=(0.99, 0.99),
    )

    fig.suptitle(
        f"{panel_title} vs suppression gap (fixed − mix)\n"
        "Rows: PF7B | GF8B  ·  Col: Mix prompts only\n"
        f"Dots: 95% CI. {fit_desc}",
        fontsize=10, fontweight="bold", y=1.03,
    )

    plt.tight_layout(rect=[0, 0, 1, 1])
    out_path = PLOTS_DIR / f"{out_filename}_{timestamp}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path
