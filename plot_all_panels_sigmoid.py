#!/usr/bin/env python3
"""Unified script: trait-specific token SVD + sigmoid regression on all 2×2 panel plots.

Three tasks:
  1. Compute 4 trait-specific token-wise SVDs (TruncatedSVD, n_components=3) and add
     pc1_tok_{trait}, pc2_tok_{trait}, pc3_tok_{trait} columns to dataset.csv.
     Also add tok_svd_zsum_{trait} = z(pc1)+z(pc2)+z(pc3) within each trait's filtered set.

  2. Re-plot all existing panel2x2_*.png plots with sigmoid regression instead of OLS.

  3. New 2×2 panels for tok_svd_zsum_playful, tok_svd_zsum_french, tok_svd_zsum_german,
     tok_svd_zsum_flattering (two plots: one for PF7B, one for GF8B).

Sigmoid fit:
    y/100 ~ sigmoid(a*x + b) = 1 / (1 + exp(-(a*x + b)))
  Fitted with scipy.optimize.curve_fit; 95% CI from 1000 Monte-Carlo samples of (a,b).
  Falls back to linear regression if fitting fails.

Saves new-timestamped panels in plots/.
Updates slides/data/dataset.csv with new columns.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD

# ─── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = ROOT / "slides" / "data" / "dataset.csv"
RESULTS_DIR = ROOT / "results"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ─── Experiment / trait definitions ────────────────────────────────────────────

EXPERIMENTS = [
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
    },
]

# Each entry maps to: (experiment_key, trait_name, prompt_family_list)
TRAIT_SVDS = [
    ("playful_french_7b", "Playful", "playful", ["playful", "neutral"]),
    ("playful_french_7b", "French",  "french",  ["french",  "neutral"]),
    ("german_flattering_8b", "German",     "german",     ["german",     "neutral"]),
    ("german_flattering_8b", "Flattering", "flattering", ["flattering", "neutral"]),
]

CONDITIONS = [
    {"prefix_type": "fixed", "label": "Fixed prompts"},
    {"prefix_type": "mix",   "label": "Rephrased (mix) prompts"},
]

# ─── Token W-matrix builder ────────────────────────────────────────────────────

def _build_W_tokens(
    data: dict,
    prompt_keys: list[str],
    field: str = "lp_train_inoc_tokens",
) -> tuple[np.ndarray, list[str]]:
    """Build W_tokens matrix (n_prompts × concatenated_token_diffs).

    W[n, :] = concat over examples k of (inoc_toks[k] - baseline_toks[k])[:min_L_k]
    Right-padded with 0.  Returns (W, valid_keys).
    """
    baseline_toks = data["baseline"]["lp_train_default_tokens"]
    prompts_toks = data["prompts"]

    valid_keys: list[str] = []
    rows: list[list[float]] = []

    for key in prompt_keys:
        if key not in prompts_toks:
            continue
        raw = prompts_toks[key].get(field)
        if raw is None:
            continue

        row: list[float] = []
        for k in range(len(baseline_toks)):
            def_t = baseline_toks[k]
            inoc_t = raw[k] if k < len(raw) else []
            L = min(len(def_t), len(inoc_t))
            if L == 0:
                continue
            row.extend(float(inoc_t[l]) - float(def_t[l]) for l in range(L))
        valid_keys.append(key)
        rows.append(row)

    if not rows:
        return np.empty((0, 0), dtype=np.float32), valid_keys

    max_len = max(len(r) for r in rows)
    W = np.zeros((len(rows), max_len), dtype=np.float32)
    for i, r in enumerate(rows):
        W[i, :len(r)] = r
    return W, valid_keys


# ─── Task 1: Trait-specific token SVDs ─────────────────────────────────────────

def compute_trait_token_svds(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-trait TruncatedSVD(n=3) on W_tokens; add columns to df.

    New columns added:
      pc1_tok_{trait_lower}, pc2_tok_{trait_lower}, pc3_tok_{trait_lower}
      tok_svd_zsum_{trait_lower}

    Rows outside each trait's filtered set get NaN.
    Orientation: flip sign so mean(neutral) < mean(strong).
    """
    # Load token files per experiment (cache by key)
    token_data_cache: dict[str, dict] = {}

    # Collect all unique experiment×trait SVD results as {prompt_key: [pc1,pc2,pc3]}
    # keyed by (exp_key, trait_lower)
    result_maps: dict[tuple[str, str], dict[str, list[float]]] = {}

    for exp_key, trait_name, trait_family, families in TRAIT_SVDS:
        trait_lower = trait_name.lower()
        print(f"\n  === Trait SVD: {exp_key} / {trait_name} ===")

        # Load token data for this experiment if not cached
        if exp_key not in token_data_cache:
            exp_def = next(e for e in EXPERIMENTS if e["key"] == exp_key)
            tokens_file = exp_def["tokens_file"]
            print(f"  Loading {tokens_file.name} …")
            with open(tokens_file) as f:
                token_data_cache[exp_key] = json.load(f)
        data = token_data_cache[exp_key]

        # Get prompts in this trait's filtered set (from CSV)
        sub = (
            df[(df["experiment"] == exp_key)]
            .drop_duplicates("prompt_key")[["prompt_key", "prompt_family"]]
        )
        sub_filtered = sub[sub["prompt_family"].isin(families)]
        prompt_keys = sub_filtered["prompt_key"].tolist()
        prompt_family_map = dict(zip(sub_filtered["prompt_key"], sub_filtered["prompt_family"]))
        print(f"  Filtered prompt set: {len(prompt_keys)} prompts ({families})")

        # Build W_tokens for this filtered set
        W, valid_keys = _build_W_tokens(data, prompt_keys, field="lp_train_inoc_tokens")
        print(f"  W_tokens shape: {W.shape}  ({len(valid_keys)} valid keys)")

        if W.shape[0] < 3:
            print(f"  WARNING: too few rows ({W.shape[0]}), skipping SVD")
            result_maps[(exp_key, trait_lower)] = {}
            continue

        # TruncatedSVD (uncentred) — n_components=3
        n_comp = min(3, W.shape[0] - 1, W.shape[1])
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        coords = svd.fit_transform(W)  # (n_valid, 3)
        print(f"  Explained variance: " +
              " ".join(f"PC{i+1}={svd.explained_variance_ratio_[i]*100:.1f}%"
                       for i in range(n_comp)))

        # Orient each axis: flip if mean(neutral) > mean(strong)
        for pc_idx in range(n_comp):
            arr = coords[:, pc_idx]
            families_arr = [prompt_family_map.get(k, "") for k in valid_keys]
            neutral_mask = np.array([f == "neutral" for f in families_arr])
            strong_mask = ~neutral_mask
            mean_neutral = arr[neutral_mask].mean() if neutral_mask.sum() > 0 else 0.0
            mean_strong  = arr[strong_mask].mean()  if strong_mask.sum()  > 0 else 0.0
            if mean_neutral > mean_strong:
                coords[:, pc_idx] *= -1
                action = "FLIPPED"
            else:
                action = "kept"
            print(f"    PC{pc_idx+1}: mean_neutral={mean_neutral:.4f}, mean_strong={mean_strong:.4f} → {action}")

        result_maps[(exp_key, trait_lower)] = {
            k: [float(coords[i, j]) for j in range(n_comp)]
            for i, k in enumerate(valid_keys)
        }

    # ── Add columns to df ──────────────────────────────────────────────────────
    print("\n  Adding trait SVD columns to dataset.csv …")

    # Prepare new column vectors (NaN by default)
    col_names = []
    for _, trait_name, _, _ in TRAIT_SVDS:
        t = trait_name.lower()
        col_names += [f"pc1_tok_{t}", f"pc2_tok_{t}", f"pc3_tok_{t}"]

    for c in col_names:
        df[c] = float("nan")

    for idx, row in df.iterrows():
        exp_key = row["experiment"]
        pk = row["prompt_key"]
        for _, trait_name, _, _ in TRAIT_SVDS:
            t = trait_name.lower()
            key = (exp_key, t)
            if key in result_maps and pk in result_maps[key]:
                pcs = result_maps[key][pk]
                df.at[idx, f"pc1_tok_{t}"] = pcs[0] if len(pcs) > 0 else float("nan")
                df.at[idx, f"pc2_tok_{t}"] = pcs[1] if len(pcs) > 1 else float("nan")
                df.at[idx, f"pc3_tok_{t}"] = pcs[2] if len(pcs) > 2 else float("nan")

    # ── Compute tok_svd_zsum_{trait} ─────────────────────────────────────────
    print("  Computing tok_svd_zsum_{trait} …")

    def _nanzscore(arr: np.ndarray) -> np.ndarray:
        mu = np.nanmean(arr)
        sigma = np.nanstd(arr)
        if sigma == 0 or np.isnan(sigma):
            return np.zeros_like(arr, dtype=float)
        return (arr - mu) / sigma

    for exp_key, trait_name, _, families in TRAIT_SVDS:
        t = trait_name.lower()
        col_zsum = f"tok_svd_zsum_{t}"
        df[col_zsum] = float("nan")

        # Z-score computed within the trait's filtered set for this experiment
        mask = (
            (df["experiment"] == exp_key)
            & (df["prompt_family"].isin(families))
            & df[f"pc1_tok_{t}"].notna()
        )
        if mask.sum() == 0:
            continue

        # Get unique prompt_keys in the filtered set to compute z-scores at prompt level
        sub = df[mask].drop_duplicates("prompt_key")
        pc1_arr = sub[f"pc1_tok_{t}"].values.astype(float)
        pc2_arr = sub[f"pc2_tok_{t}"].values.astype(float)
        pc3_arr = sub[f"pc3_tok_{t}"].values.astype(float)

        z1 = _nanzscore(pc1_arr)
        z2 = _nanzscore(pc2_arr)
        z3 = _nanzscore(pc3_arr)
        zsum = z1 + z2 + z3

        pk_to_zsum = dict(zip(sub["prompt_key"].tolist(), zsum.tolist()))

        # Apply back to all rows of that experiment (all trait_name/prefix_type rows)
        all_mask = (df["experiment"] == exp_key) & (df["prompt_family"].isin(families))
        for idx, row in df[all_mask].iterrows():
            pk = row["prompt_key"]
            if pk in pk_to_zsum:
                df.at[idx, col_zsum] = pk_to_zsum[pk]

    # ── Summary ───────────────────────────────────────────────────────────────
    for _, trait_name, _, _ in TRAIT_SVDS:
        t = trait_name.lower()
        for col in [f"pc1_tok_{t}", f"pc2_tok_{t}", f"pc3_tok_{t}", f"tok_svd_zsum_{t}"]:
            n_nonnull = df[col].notna().sum()
            print(f"  {col}: {n_nonnull}/{len(df)} non-NaN rows")

    df.to_csv(CSV_PATH, index=False)
    print(f"  Saved: {CSV_PATH}")
    return df


# ─── Sigmoid fit helpers ────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Logistic sigmoid: 1 / (1 + exp(-(a*x + b))). Returns values in [0, 1]."""
    return 1.0 / (1.0 + np.exp(-np.clip(a * x + b, -500, 500)))


def _sigmoid_regression_band(
    x: np.ndarray,
    y: np.ndarray,          # in % (0–100)
    x_sorted: np.ndarray,
    n_samples: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """Fit sigmoid to (x, y/100) and return (y_hat, ci_lo, ci_hi, slope_a, success).

    All outputs are in % (0–100) scale.
    95% CI computed via bootstrap: resample (x, y) with replacement n_samples times,
    refit the sigmoid each time, take 2.5th/97.5th percentiles across curves.
    Falls back to linear regression and returns success=False if curve_fit fails.
    """
    y_01 = y / 100.0  # convert to [0, 1]

    # Initial guess: a from linear fit slope sign, b=0
    slope_init = np.sign(np.corrcoef(x, y_01)[0, 1]) * 0.1
    p0 = [slope_init, 0.0]

    try:
        popt, _ = curve_fit(
            _sigmoid, x, y_01,
            p0=p0,
            maxfev=5000,
            bounds=([-np.inf, -np.inf], [np.inf, np.inf]),
        )
        a, b = popt
        y_hat = _sigmoid(x_sorted, a, b) * 100.0

        # Bootstrap 95% CI: resample data, refit each replicate
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
                    _sigmoid, xb, yb,
                    p0=popt,
                    maxfev=5000,
                    bounds=([-np.inf, -np.inf], [np.inf, np.inf]),
                )
                boot_curves.append(_sigmoid(x_sorted, *pb) * 100.0)
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
        # Fall back to linear regression
        slope, intercept, *_ = scipy_stats.linregress(x, y)
        y_hat = slope * x_sorted + intercept
        n = len(x)
        x_bar = np.mean(x)
        ss_x = np.sum((x - x_bar) ** 2)
        y_pred = slope * x + intercept
        mse = np.sum((y - y_pred) ** 2) / max(n - 2, 1)
        t_crit = scipy_stats.t.ppf(0.975, df=max(n - 2, 1))
        ci = t_crit * np.sqrt(mse) * np.sqrt(1.0 / n + (x_sorted - x_bar) ** 2 / max(ss_x, 1e-30))
        return y_hat, y_hat - ci, y_hat + ci, float("nan"), False


def _pearson_sigmoid_label(x: np.ndarray, y: np.ndarray) -> tuple[str, str]:
    """Return (stats_str, short_str) where stats_str includes r and sigmoid a."""
    mask = np.isfinite(x) & np.isfinite(y)
    n = mask.sum()
    if n < 3:
        return f"n={n}", f"n={n}"
    xm, ym = x[mask], y[mask]
    if np.std(xm) == 0 or np.std(ym) == 0:
        return f"r=n/a (constant, n={n})", f"r=n/a"

    r, p = pearsonr(xm, ym)
    r2 = r ** 2
    p_str = f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"

    # Sigmoid slope
    y_01 = ym / 100.0
    slope_init = np.sign(r) * 0.1
    try:
        popt, _ = curve_fit(_sigmoid, xm, y_01, p0=[slope_init, 0.0], maxfev=5000)
        a_str = f", a={popt[0]:.3f}"
    except Exception:
        a_str = " (lin. fallback)"

    stats = f"r={r:+.2f}, r²={r2:.2f}, p={p_str}{a_str} (n={n})"
    return stats, f"r={r:+.2f}"


# ─── Plotting helpers ──────────────────────────────────────────────────────────

def _add_series_sigmoid(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    y_lo: np.ndarray,
    y_hi: np.ndarray,
    color: str,
    na_annotation: str | None = None,
) -> str:
    """Scatter + CI error bars on Y + sigmoid regression + 95% CI band.

    Dot CI error bars: ±1.96*sqrt(p*(1-p)/200) (unchanged from existing plots).
    Regression: sigmoid fit (fallback to linear).
    Returns stats string.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    xm, ym = x[mask], y[mask]
    ym_lo = y_lo[mask]
    ym_hi = y_hi[mask]

    stats_str, _ = _pearson_sigmoid_label(x, y)

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

    # Error bars on Y
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

    # Sigmoid (or linear fallback) regression line + 95% CI band
    if len(xm) >= 3 and np.std(xm) > 0:
        try:
            x_sorted = np.linspace(xm.min(), xm.max(), 100)
            y_hat, band_lo, band_hi, a, success = _sigmoid_regression_band(xm, ym, x_sorted)
            ax.plot(x_sorted, y_hat, color=color, linewidth=1.3, alpha=0.85, zorder=2,
                    linestyle="-" if success else "--")
            ax.fill_between(x_sorted, band_lo, band_hi, color=color, alpha=0.15, zorder=1)
            if not success:
                ax.annotate("(linear fallback)", xy=(0.5, 0.01), xycoords="axes fraction",
                             fontsize=5.5, ha="center", color="gray")
        except Exception:
            pass

    return stats_str


# ─── 2×2 panel factory with sigmoid ───────────────────────────────────────────

def _plot_2x2_panel(
    df: pd.DataFrame,
    heuristic_col: str,
    panel_xlabel: str,
    panel_title: str,
    out_filename: str,
    na_on_fixed: bool = False,
    pos_only_families: dict[str, str] | None = None,   # exp_key -> family_override for pos trait
    neg_only_families: dict[str, str] | None = None,
) -> Path:
    """Produce a 2 × 2 panel figure for one heuristic with sigmoid regression.

    na_on_fixed: if True, show N/A annotation on fixed-prefix columns.
    pos_only_families: if provided, override pos_family for positive-trait filter.
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

            na_annot = ("N/A — single prompt\n(metric requires rephrasing pool)"
                        if (na_on_fixed and is_fixed) else None)

            # Positive trait
            mask_pos = (
                (df["experiment"] == exp_key)
                & (df["prefix_type"] == prefix_type)
                & (df["trait_name"] == pos_trait)
                & (df["prompt_family"].isin([pos_family, "neutral"]))
                & df["suppression"].notna()
            )
            df_pos = df[mask_pos].copy()

            # Negative trait
            mask_neg = (
                (df["experiment"] == exp_key)
                & (df["prefix_type"] == prefix_type)
                & (df["trait_name"] == neg_trait)
                & (df["prompt_family"].isin([neg_family, "neutral"]))
                & df["suppression"].notna()
            )
            df_neg = df[mask_neg].copy()

            if na_annot is not None:
                x_pos = np.full(len(df_pos), np.nan)
                x_neg = np.full(len(df_neg), np.nan)
            else:
                x_pos = (df_pos[heuristic_col].values.astype(float)
                         if heuristic_col in df_pos.columns else np.full(len(df_pos), np.nan))
                x_neg = (df_neg[heuristic_col].values.astype(float)
                         if heuristic_col in df_neg.columns else np.full(len(df_neg), np.nan))

            y_pos = df_pos["suppression"].values.astype(float)
            y_pos_lo = df_pos["suppression_ci_lo"].values.astype(float)
            y_pos_hi = df_pos["suppression_ci_hi"].values.astype(float)

            y_neg = df_neg["suppression"].values.astype(float)
            y_neg_lo = df_neg["suppression_ci_lo"].values.astype(float)
            y_neg_hi = df_neg["suppression_ci_hi"].values.astype(float)

            stats_pos = _add_series_sigmoid(
                ax, x_pos, y_pos, y_pos_lo, y_pos_hi, color_pos, na_annotation=na_annot,
            )

            if na_annot is not None:
                mask_y = np.isfinite(y_neg)
                if mask_y.sum() > 0:
                    ax.scatter(
                        np.zeros(mask_y.sum()), y_neg[mask_y],
                        s=30, color=color_neg, alpha=0.5,
                        edgecolors="white", linewidths=0.3, zorder=3,
                    )
                stats_neg = _pearson_sigmoid_label(x_neg, y_neg)[0]
            else:
                stats_neg = _add_series_sigmoid(
                    ax, x_neg, y_neg, y_neg_lo, y_neg_hi, color_neg, na_annotation=None,
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
            if c_idx > 0:
                ax.set_ylabel("")

    legend_handles = []
    for exp in EXPERIMENTS:
        legend_handles.append(
            mlines.Line2D([], [], marker="o", color="w",
                          markerfacecolor=exp["color_pos"], markersize=7,
                          label=f"{exp['pos_trait']} ({exp['row_label']})")
        )
        legend_handles.append(
            mlines.Line2D([], [], marker="o", color="w",
                          markerfacecolor=exp["color_neg"], markersize=7,
                          label=f"{exp['neg_trait']} ({exp['row_label']})")
        )
    fig.legend(
        handles=legend_handles, loc="upper right", fontsize=8,
        framealpha=0.8, ncol=2, bbox_to_anchor=(0.99, 0.99),
    )

    fig.suptitle(
        f"{panel_title} vs trait suppression\n"
        "Rows: PF7B | GF8B  ·  Cols: Fixed | Mix prompts\n"
        "Dots: 95% CI. Lines: sigmoid regression + 95% CI band (bootstrap, 1000 resamples).",
        fontsize=10, fontweight="bold", y=1.03,
    )

    plt.tight_layout(rect=[0, 0, 1, 1])
    out_path = PLOTS_DIR / f"{out_filename}_{TIMESTAMP}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


# ─── Task 2 + 3: Panel specifications ─────────────────────────────────────────

# Each tuple: (out_filename, heuristic_col, xlabel, title, na_on_fixed)
PANEL_SPECS = [
    # Existing panel2x2 plots
    ("panel2x2_emb_dist_neutral",
     "emb_dist_from_neutral",
     "Embedding L2 distance from neutral centroid",
     "Embedding Distance from Neutral",
     False),

    ("panel2x2_emb_dist_neutral_svd3",
     "emb_dist_from_neutral_svd3",
     "Embedding L2 distance from neutral centroid (SVD3)",
     "Embedding Distance from Neutral (SVD3)",
     False),

    ("panel2x2_emb_svd3_pc1",
     "emb_svd3_pc1",
     "Embedding SVD3 — PC1",
     "Embedding SVD3 PC1",
     False),

    ("panel2x2_emb_svd3_pc2",
     "emb_svd3_pc2",
     "Embedding SVD3 — PC2",
     "Embedding SVD3 PC2",
     False),

    ("panel2x2_lp_spread_mean",
     "lp_spread_mean",
     "LP spread mean (mean(mix − fixed) logprob)",
     "LP Spread Mean",
     True),   # only meaningful for mix; fixed panels: N/A

    ("panel2x2_lp_spread_std",
     "lp_spread_std",
     "LP spread std (std(mix − fixed) logprob)",
     "LP Spread Std",
     True),

    ("panel2x2_rephrasing_diversity",
     "emb_rephrase_std_cos",
     "Rephrasing diversity (std cosine sim to original)",
     "Rephrasing Diversity (emb_rephrase_std_cos)",
     True),

    ("panel2x2_rephrase_diversity_svd3",
     "emb_rephrase_std_cos_svd3",
     "Rephrasing diversity (std cosine sim, SVD3)",
     "Rephrasing Diversity SVD3",
     True),

    ("panel2x2_tokens_svd_pc1",
     "pc1_tokens_oriented",
     "Token-wise Logprob SVD — PC1\n(oriented: neutral < strong)",
     "Token-SVD PC1 (oriented)",
     False),

    ("panel2x2_tokens_svd_pc2",
     "pc2_tokens_oriented",
     "Token-wise Logprob SVD — PC2\n(oriented: neutral < strong)",
     "Token-SVD PC2 (oriented)",
     False),

    ("panel2x2_w_cos",
     "w_cos",
     "Cosine similarity: W_fixed vs W_mix",
     "W-vector Cosine Similarity",
     True),

    ("panel2x2_w_sign_agree",
     "w_sign_agree",
     "Sign agreement: W_fixed vs W_mix",
     "W-vector Sign Agreement",
     True),

    ("panel2x2_w_std_diff",
     "w_std_diff",
     "Std of W_mix − W_fixed per-example differences",
     "W-vector Std Difference",
     True),

    # combo / emb_dist / emb_std / spread panels (from earlier scripts)
    ("panel2x2_emb_dist",
     "emb_dist_from_neutral",
     "Embedding distance from neutral centroid",
     "Embedding Distance from Neutral",
     False),

    ("panel2x2_emb_std",
     "emb_rephrase_std_cos",
     "Rephrasing diversity (std cosine sim)",
     "Rephrasing Diversity",
     True),

    ("panel2x2_spread_mean",
     "lp_spread_mean",
     "LP spread mean",
     "LP Spread Mean",
     True),

    ("panel2x2_spread_std",
     "lp_spread_std",
     "LP spread std",
     "LP Spread Std",
     True),

    # New: trait-specific token SVD zsum (one per experiment pair)
    ("panel2x2_tok_svd_zsum_playful",
     "tok_svd_zsum_playful",
     "Token SVD z-sum (Playful trait subset)",
     "Token SVD Z-sum — Playful",
     False),

    ("panel2x2_tok_svd_zsum_french",
     "tok_svd_zsum_french",
     "Token SVD z-sum (French trait subset)",
     "Token SVD Z-sum — French",
     False),

    ("panel2x2_tok_svd_zsum_german",
     "tok_svd_zsum_german",
     "Token SVD z-sum (German trait subset)",
     "Token SVD Z-sum — German",
     False),

    ("panel2x2_tok_svd_zsum_flattering",
     "tok_svd_zsum_flattering",
     "Token SVD z-sum (Flattering trait subset)",
     "Token SVD Z-sum — Flattering",
     False),

    # combo: emb_dist + rephrasing diversity (legacy; use emb_dist_from_neutral as a composite proxy)
    # Actually produce the "combo" panel using pc1_mix_x_emb_rephrase_std if it exists
    ("panel2x2_combo",
     "ph_combined",
     "Perplexity Heuristic (combined, PH)",
     "Perplexity Heuristic (PH)",
     False),
]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> list[Path]:
    print("=" * 70)
    print("plot_all_panels_sigmoid.py")
    print(f"Timestamp: {TIMESTAMP}")
    print("=" * 70)

    # Load dataset
    df = pd.read_csv(CSV_PATH)
    print(f"\nLoaded CSV: {df.shape}")

    # ── Task 1: Compute trait-specific token SVDs ──────────────────────────────
    print("\n" + "─" * 60)
    print("Task 1: Trait-specific token SVDs")
    print("─" * 60)
    df = compute_trait_token_svds(df)

    # ── Task 2 + 3: Re-plot all panel2x2 with sigmoid ─────────────────────────
    print("\n" + "─" * 60)
    print("Task 2+3: Generating panel2x2 plots with sigmoid regression")
    print("─" * 60)

    out_paths: list[Path] = []
    for spec in PANEL_SPECS:
        out_fname, hcol, xlabel, title, na_fixed = spec
        if hcol not in df.columns:
            print(f"  SKIP {out_fname}: column '{hcol}' not in CSV")
            continue
        try:
            p = _plot_2x2_panel(
                df,
                heuristic_col=hcol,
                panel_xlabel=xlabel,
                panel_title=title,
                out_filename=out_fname,
                na_on_fixed=na_fixed,
            )
            out_paths.append(p)
        except Exception as exc:
            print(f"  ERROR generating {out_fname}: {exc}")

    print(f"\nDone. {len(out_paths)} plots saved.")
    return out_paths


if __name__ == "__main__":
    out_paths = main()
    print("\nAll plots:")
    for p in out_paths:
        print(f"  {p}")
