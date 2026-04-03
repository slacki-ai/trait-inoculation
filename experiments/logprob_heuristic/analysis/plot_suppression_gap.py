#!/usr/bin/env python3
"""
Plot heuristics vs suppression gap  (fixed suppression − mix suppression).

Figure 1 (suppression_gap_main_*.png)  – 2×3 grid, the 6 primary heuristics:
  1. Elicitation strength
  2. PC1 from logprob-diff SVD (fixed)
  3. emb_dist_from_neutral
  4. emb_rephrase_std_cos
  5. Cross-trait projection in 5-D SVD space
  6. Cross-trait projection in raw 1000-D W space

Figure 2 (suppression_gap_extended_*.png)  – 3×4 grid, additional heuristics:
  emb_rephrase_mean_cos, emb_rephrase_min_cos, emb_rephrase_eff_rank,
  cos(W_fixed,W_mix), ‖W_fixed−W_mix‖, PH_mix/PH_fixed, SNR_fixed,
  selfperp_raw, selfperp_ctx,
  elicitation×emb_rephrase_std_cos, pc1_mix×emb_rephrase_std_cos

Each panel shows one regression line per trait  (4 traits: French, Playful, German, Flattering).
Regression is restricted to prompts that *target* the given trait, measuring suppression of
*that same trait* — so elicitation and suppression are always for the same dimension.

Usage
-----
  cd /path/to/repo
  python slides/compute_new_columns.py   # adds ph_mix_all, snr_all, w_l2, ph_ratio, …
  python experiments/logprob_heuristic/analysis/plot_suppression_gap.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # repo root

# --------------------------------------------------------------------------- #
#  Trait groups: (experiment, prompt_family, trait_name, color, short_label)  #
# --------------------------------------------------------------------------- #

TRAIT_GROUPS: list[tuple[str, str, str, str, str]] = [
    ("playful_french_7b",    "french",     "French",     "#3498db", "PF-7B / French"),
    ("playful_french_7b",    "playful",    "Playful",    "#e74c3c", "PF-7B / Playful"),
    ("german_flattering_8b", "german",     "German",     "#27ae60", "GF-8B / German"),
    ("german_flattering_8b", "flattering", "Flattering", "#e67e22", "GF-8B / Flattering"),
]

OTHER_TRAIT: dict[str, str] = {
    "playful":    "french",
    "french":     "playful",
    "german":     "flattering",
    "flattering": "german",
}

SV_COLS = [f"sv{i}_truncated_fixed" for i in range(1, 6)]

# --------------------------------------------------------------------------- #
#  Sign correction of SVD coordinates                                          #
# --------------------------------------------------------------------------- #

def compute_sign_corrected_svd(gap: pd.DataFrame) -> pd.DataFrame:
    """
    For each experiment × SVD dimension, flip the sign of the coordinate so that
    the mean across non-neutral prompts is *higher* than the mean for neutral prompts.
    This gives a canonical orientation: "more active" prompts have positive coordinates.

    The sign is determined from the FULL dataset (which includes neutral rows).
    It is applied to sv1..sv3_truncated_fixed within gap_df (which excludes neutral
    rows but uses the same SVD space, so the sign flip is valid).

    New columns added to gap: sv1_signed, sv2_signed, sv3_signed.
    """
    full_df = pd.read_csv(Path(__file__).resolve().parent.parent.parent.parent
                          / "slides/data/dataset.csv")

    for sv_idx in range(1, 4):
        sv_col     = f"sv{sv_idx}_truncated_fixed"
        signed_col = f"sv{sv_idx}_signed"

        gap[signed_col] = gap[sv_col].astype(float)   # default: no flip

        for exp in gap["experiment"].unique():
            exp_full = full_df[full_df["experiment"] == exp]
            neutral_mean = exp_full[exp_full["prompt_family"] == "neutral"][sv_col].mean()
            active_mean  = exp_full[exp_full["prompt_family"] != "neutral"][sv_col].mean()

            if pd.isna(neutral_mean) or pd.isna(active_mean):
                continue

            sign = +1.0 if active_mean >= neutral_mean else -1.0
            mask = gap["experiment"] == exp
            gap.loc[mask, signed_col] = sign * gap.loc[mask, sv_col].astype(float)

    # z-sum: standardise each signed PC (per-experiment), then sum
    gap["z_sum_signed_svd"] = 0.0
    for sv_idx in range(1, 4):
        signed_col = f"sv{sv_idx}_signed"
        for exp in gap["experiment"].unique():
            mask = gap["experiment"] == exp
            vals = gap.loc[mask, signed_col].astype(float)
            mu, sd = vals.mean(), vals.std()
            if sd > 1e-9:
                gap.loc[mask, "z_sum_signed_svd"] += (vals - mu) / sd

    return gap


def cross_trait_svd_proj_signed(gap: pd.DataFrame) -> np.ndarray:
    """Same as cross_trait_svd_proj but in the 3-D sign-corrected SVD space."""
    signed_cols = ["sv1_signed", "sv2_signed", "sv3_signed"]
    projs = []
    for _, row in gap.iterrows():
        other = OTHER_TRAIT.get(row["prompt_family"])
        if other is None:
            projs.append(np.nan)
            continue
        mask = (gap["experiment"] == row["experiment"]) & (gap["prompt_family"] == other)
        other_vecs = gap.loc[mask, signed_cols].drop_duplicates().values.astype(float)
        if len(other_vecs) == 0:
            projs.append(np.nan)
            continue
        mu = other_vecs.mean(axis=0)
        norm_mu = np.linalg.norm(mu)
        if norm_mu < 1e-9:
            projs.append(np.nan)
            continue
        v = row[signed_cols].values.astype(float)
        projs.append(float(np.dot(v, mu) / norm_mu))
    return np.array(projs)


# --------------------------------------------------------------------------- #
#  Data loading & gap computation                                              #
# --------------------------------------------------------------------------- #

def load_gap_df() -> pd.DataFrame:
    """Load dataset.csv, pivot fixed/mix rows, compute suppression gap per (prompt, trait)."""
    df = pd.read_csv(ROOT / "slides/data/dataset.csv")
    df = df[df["prompt_family"].isin(["playful", "french", "german", "flattering"])].copy()

    key_cols = [
        "experiment", "prompt_key", "prompt_text",
        "prompt_group", "prompt_family", "trait_role", "trait_name",
    ]
    # All heuristic columns we carry on fixed rows
    want_cols = [
        "elicitation", "ph", "ph_combined",
        "pc1_fixed", "pc2_fixed", "pc3_fixed",
        "pc1_mix", "pc2_mix", "pc3_mix",
        "sv1_truncated_fixed", "sv2_truncated_fixed", "sv3_truncated_fixed",
        "sv4_truncated_fixed", "sv5_truncated_fixed",
        "sv1_truncated_mix", "sv2_truncated_mix",
        "emb_dist_from_neutral",
        "emb_rephrase_std_cos", "emb_rephrase_mean_cos",
        "emb_rephrase_min_cos", "emb_rephrase_eff_rank",
        "emb_cos_to_neg_trait", "emb_cos_to_pos_trait",
        "selfperp_raw", "selfperp_ctx",
        "w_cos", "w_cor", "w_sign_agree",
        # new columns (present after compute_new_columns.py)
        "ph_mix_all", "ph_ratio", "ph_drop",
        "snr_all", "snr_mix", "snr_drop",
        "snr_ratio_fixed_over_mix", "snr_ratio_mix_over_fixed",
        "w_l2", "w_proj",
        "token_recovered_frac",
        "absSNR_fixed", "absSNR_mix", "absSNR_ratio",
        "pc1_mix_x_absSNR_ratio",
        "pc1_mix_x_snr_ratio_mix_over_fixed",
        "emb_dist_from_neutral_svd3",
        "emb_rephrase_std_cos_svd3",
        "emb_rephrase_std_cos_svd5",
        "elicitation_x_emb_rephrase_std",
        "pc1_mix_x_emb_rephrase_std",
        # per-trait token-level SVD PC1 (sign-corrected)
        "pc1_trait_tok_svd",
    ]
    want_cols = [c for c in want_cols if c in df.columns]

    fixed = df[df["prefix_type"] == "fixed"].copy()
    mix   = df[df["prefix_type"] == "mix"].copy()

    # Carry CI columns from both fixed and mix so we can compute gap uncertainty
    ci_cols = [c for c in ["suppression_ci_lo", "suppression_ci_hi"] if c in df.columns]

    base = (
        fixed[key_cols + want_cols + ["suppression"] + ci_cols]
        .rename(columns={
            "suppression": "sup_fixed",
            **{c: f"{c}_fixed" for c in ci_cols},
        })
    )
    mx = (
        mix[key_cols + ["suppression"] + ci_cols]
        .rename(columns={
            "suppression": "sup_mix",
            **{c: f"{c}_mix" for c in ci_cols},
        })
    )
    gap = base.merge(mx, on=key_cols)
    gap["gap"] = gap["sup_fixed"] - gap["sup_mix"]

    # Propagate 95% CI for the gap  (assuming fixed and mix are independent)
    if "suppression_ci_lo_fixed" in gap.columns and "suppression_ci_lo_mix" in gap.columns:
        se_fixed = (gap["suppression_ci_hi_fixed"] - gap["suppression_ci_lo_fixed"]) / (2 * 1.96)
        se_mix   = (gap["suppression_ci_hi_mix"]   - gap["suppression_ci_lo_mix"])   / (2 * 1.96)
        gap["gap_se"] = np.sqrt(se_fixed**2 + se_mix**2)

    return gap.dropna(subset=["gap"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  Cross-trait projections (computed on-the-fly from loaded data)             #
# --------------------------------------------------------------------------- #

def cross_trait_svd_proj(gap: pd.DataFrame) -> np.ndarray:
    """
    For each prompt (targeting trait A), project its 5-D SVD coordinates onto the
    centroid direction of prompts targeting trait B within the same experiment.
    """
    projs = []
    for _, row in gap.iterrows():
        other = OTHER_TRAIT.get(row["prompt_family"])
        if other is None:
            projs.append(np.nan)
            continue
        mask = (gap["experiment"] == row["experiment"]) & (gap["prompt_family"] == other)
        other_vecs = gap.loc[mask, SV_COLS].drop_duplicates().values
        if len(other_vecs) == 0:
            projs.append(np.nan)
            continue
        mu = other_vecs.mean(axis=0)
        norm_mu = np.linalg.norm(mu)
        if norm_mu < 1e-9:
            projs.append(np.nan)
            continue
        projs.append(float(np.dot(row[SV_COLS].values.astype(float), mu) / norm_mu))
    return np.array(projs)


def _load_raw_W(json_path: Path) -> dict[str, np.ndarray]:
    """Load W_fixed[key] = lp_train_inoc − lp_train_default  (1000-D vector) from perplexity JSON."""
    with open(json_path) as f:
        data = json.load(f)
    lp_default = np.array(data["baseline"]["lp_train_default"], dtype=float)
    W: dict[str, np.ndarray] = {}
    for key, pdata in data.get("prompts", {}).items():
        raw = pdata.get("lp_train_inoc")
        if raw is None:
            continue
        lp_inoc = np.array(raw, dtype=float)
        n = min(len(lp_inoc), len(lp_default))
        W[key] = lp_inoc[:n] - lp_default[:n]
    return W


def cross_trait_raw_proj(gap: pd.DataFrame) -> np.ndarray:
    """
    For each prompt, project its raw 1000-D W_fixed vector onto the centroid of
    other-trait prompts' W_fixed vectors within the same experiment.
    """
    PF_JSON = ROOT / "results/perplexity_heuristic_qwen2.5-7b-instruct.json"
    GF_JSON = ROOT / "results/perplexity_heuristic_german_flattering_llama-3.1-8b-instruct.json"

    W_by_exp: dict[str, dict[str, np.ndarray]] = {}
    for exp, path in [("playful_french_7b", PF_JSON), ("german_flattering_8b", GF_JSON)]:
        if path.exists():
            W_by_exp[exp] = _load_raw_W(path)

    projs = []
    for _, row in gap.iterrows():
        other = OTHER_TRAIT.get(row["prompt_family"])
        if other is None:
            projs.append(np.nan)
            continue
        W = W_by_exp.get(row["experiment"], {})
        if row["prompt_key"] not in W:
            projs.append(np.nan)
            continue
        other_keys = (
            gap.loc[(gap["experiment"] == row["experiment"]) & (gap["prompt_family"] == other),
                    "prompt_key"].unique()
        )
        other_vecs = [W[k] for k in other_keys if k in W]
        if not other_vecs:
            projs.append(np.nan)
            continue
        n = min(min(len(v) for v in other_vecs), len(W[row["prompt_key"]]))
        mu = np.mean([v[:n] for v in other_vecs], axis=0)
        norm_mu = np.linalg.norm(mu)
        if norm_mu < 1e-9:
            projs.append(np.nan)
            continue
        projs.append(float(np.dot(W[row["prompt_key"]][:n], mu) / norm_mu))
    return np.array(projs)


# --------------------------------------------------------------------------- #
#  Per-trait scatter panel                                                     #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#  Sigmoid (4-parameter logistic) fit helpers                                 #
# --------------------------------------------------------------------------- #

# Gap is in pp; theoretical bounds are [-100, 100] pp.
_GAP_BOUND = 100.0  # pp


def _logistic4(x: np.ndarray, L: float, k: float, x0: float, b: float) -> np.ndarray:
    """4-parameter logistic: f(x) = L / (1 + exp(-k*(x-x0))) + b."""
    return L / (1.0 + np.exp(-np.clip(k * (x - x0), -500, 500))) + b


def _fit_sigmoid(
    xm: np.ndarray, ym: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Fit 4-parameter logistic to (xm, ym).  Returns (popt, pcov) or (None, None)
    if the fit fails or there are too few points.

    Initial guess: L = ym range, k = slope heuristic, x0 = median x, b = ym min.
    Bounds keep the curve within a ±200 pp window (twice the theoretical max).
    """
    n = len(xm)
    if n < 5:
        return None, None

    y_range = float(ym.max() - ym.min()) or 1.0
    x_range = float(xm.max() - xm.min()) or 1.0
    # sign of k: positive when ym increases with xm (use Pearson sign)
    r_sign = np.sign(np.corrcoef(xm, ym)[0, 1]) or 1.0
    k0 = r_sign * 4.0 / x_range  # inflection slope heuristic

    p0 = [y_range * r_sign, k0, float(np.median(xm)), float(ym.min() if r_sign > 0 else ym.max() - y_range)]
    lower = [-2 * _GAP_BOUND, -200.0, -np.inf, -2 * _GAP_BOUND]
    upper = [ 2 * _GAP_BOUND,  200.0,  np.inf,  2 * _GAP_BOUND]

    try:
        popt, pcov = curve_fit(
            _logistic4, xm, ym,
            p0=p0, bounds=(lower, upper),
            maxfev=8000,
        )
        return popt, pcov
    except (RuntimeError, ValueError):
        return None, None


def _sigmoid_ci_band(
    ax: plt.Axes,
    xs: np.ndarray,
    xm: np.ndarray,
    ym: np.ndarray,
    popt: np.ndarray,
    color: str,
    n_samples: int = 500,
    alpha_band: float = 0.12,
) -> None:
    """Shade 95% CI band for the sigmoid curve via bootstrap resampling.

    Resamples (xm, ym) with replacement n_samples times, refits the sigmoid
    each time, and shades the 2.5th–97.5th percentile band.
    """
    try:
        rng = np.random.default_rng(42)
        n = len(xm)
        lower = [-2 * _GAP_BOUND, -200.0, -np.inf, -2 * _GAP_BOUND]
        upper = [ 2 * _GAP_BOUND,  200.0,  np.inf,  2 * _GAP_BOUND]
        boot_curves = []
        for _ in range(n_samples):
            idx = rng.integers(0, n, size=n)
            xb, yb = xm[idx], ym[idx]
            if np.std(xb) < 1e-10:
                continue
            try:
                pb, _ = curve_fit(
                    _logistic4, xb, yb,
                    p0=popt,
                    bounds=(lower, upper),
                    maxfev=5000,
                )
                boot_curves.append(_logistic4(xs, *pb))
            except Exception:
                pass
        if len(boot_curves) < 10:
            return
        ys = np.array(boot_curves)
        lo = np.percentile(ys, 2.5, axis=0)
        hi = np.percentile(ys, 97.5, axis=0)
        ax.fill_between(xs, lo, hi, color=color, alpha=alpha_band,
                        linewidth=0, zorder=2)
    except Exception:
        pass


def _regression_ci_band(
    ax: plt.Axes,
    xm: np.ndarray,
    ym: np.ndarray,
    slope: float,
    intercept: float,
    color: str,
    xs: np.ndarray,
    alpha_band: float = 0.12,
) -> None:
    """Shade 95% CI band around the OLS regression line."""
    n = len(xm)
    if n < 4:
        return
    x_mean = xm.mean()
    ss_x = np.sum((xm - x_mean) ** 2)
    if ss_x < 1e-12:
        return
    residuals = ym - (slope * xm + intercept)
    mse = np.sum(residuals ** 2) / max(n - 2, 1)
    t_val = stats.t.ppf(0.975, df=max(n - 2, 1))
    se_line = np.sqrt(mse * (1.0 / n + (xs - x_mean) ** 2 / ss_x))
    y_hat = slope * xs + intercept
    ax.fill_between(
        xs, y_hat - t_val * se_line, y_hat + t_val * se_line,
        alpha=alpha_band, color=color, linewidth=0, zorder=2,
    )


def scatter_panel(
    ax: plt.Axes,
    gap: pd.DataFrame,
    xcol: str,
    title: str,
    xlabel: str,
    use_sigmoid: bool = False,
) -> None:
    """
    Scatter plot with one regression line per trait (4 lines).

    Features:
    - 95% CI error bars on each scatter point (propagated from fixed + mix CI)
    - 95% CI shaded band around each regression line (OLS or sigmoid)
    - Per-trait stats shown as in-axes text; lines with p ≤ 0.05 are **bold**
    - use_sigmoid=True: fit 4-param logistic instead of OLS; reports R² (sigmoid) + r (Pearson)

    Each trait group is restricted to prompts that *target* the given trait and
    measure suppression of *that same trait*, keeping x and y dimensionally
    consistent within each group.

    Y-axis note: gap is in percentage points (pp), theoretical bounds [-100, 100] pp.
    """
    ax.axhline(0, color="#aaa", lw=0.8, ls="--", zorder=1)

    stat_entries: list[tuple[str, str, bool]] = []   # (text, color, significant)

    for exp, fam, trait_name, color, short_label in TRAIT_GROUPS:
        mask = (
            (gap["experiment"] == exp)
            & (gap["prompt_family"] == fam)
            & (gap["trait_name"] == trait_name)
            & gap[xcol].notna()
            & gap["gap"].notna()
        )
        if not mask.any():
            continue

        sub = gap.loc[mask].reset_index(drop=True)
        xm_all = sub[xcol].values.astype(float)
        ym_all = sub["gap"].values.astype(float)
        finite = np.isfinite(xm_all) & np.isfinite(ym_all)
        xm, ym = xm_all[finite], ym_all[finite]
        trait_short = short_label.split(" / ", 1)[-1]

        # --- error bars on dots ---
        if "gap_se" in sub.columns:
            yerr_all = sub["gap_se"].values.astype(float) * 1.96
            yerr = yerr_all[finite]
            ax.errorbar(
                xm, ym,
                yerr=np.where(np.isfinite(yerr), yerr, 0),
                fmt="none",
                ecolor=color, elinewidth=0.7, capsize=2, capthick=0.7,
                alpha=0.45, zorder=3,
            )

        ax.scatter(xm, ym, c=color, alpha=0.82, s=30, linewidths=0, zorder=4)

        if len(xm) >= 4:
            r, p = stats.pearsonr(xm, ym)
            sig = p <= 0.05
            xs = np.linspace(xm.min(), xm.max(), 200)

            if use_sigmoid:
                popt, _ = _fit_sigmoid(xm, ym)
                if popt is not None:
                    y_hat_sig = _logistic4(xs, *popt)
                    ax.plot(xs, y_hat_sig, color=color, lw=1.8, alpha=0.9, zorder=5)
                    _sigmoid_ci_band(ax, xs, xm, ym, popt, color)
                    # R² for sigmoid
                    ss_res = np.sum((ym - _logistic4(xm, *popt)) ** 2)
                    ss_tot = np.sum((ym - ym.mean()) ** 2)
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
                    p_str = "<.001" if p < 0.001 else f"={p:.2f}"
                    stat_text = (
                        f"{trait_short}: R²={r2:.2f}{'*' if sig else ''} "
                        f"r={r:+.2f} (p{p_str}, n={len(xm)})"
                    )
                else:
                    # fallback to linear if sigmoid fails
                    slope, intercept = np.polyfit(xm, ym, 1)
                    ax.plot(xs, slope * xs + intercept, color=color, lw=1.8, alpha=0.9,
                            ls="--", zorder=5)
                    _regression_ci_band(ax, xm, ym, slope, intercept, color, xs)
                    p_str = "<.001" if p < 0.001 else f"={p:.2f}"
                    stat_text = f"{trait_short}: r={r:+.2f}{'*' if sig else ''} (p{p_str}, n={len(xm)}) [lin]"
            else:
                slope, intercept = np.polyfit(xm, ym, 1)
                ax.plot(xs, slope * xs + intercept, color=color, lw=1.8, alpha=0.9, zorder=5)
                _regression_ci_band(ax, xm, ym, slope, intercept, color, xs)
                p_str = "<.001" if p < 0.001 else f"={p:.2f}"
                stat_text = f"{trait_short}: r={r:+.2f}{'*' if sig else ''} (p{p_str}, n={len(xm)})"

            stat_entries.append((stat_text, color, sig))
        else:
            stat_entries.append((f"{trait_short}: n={len(xm)} (< 4)", color, False))

    # --- in-axes stats annotation (bold when significant) ---
    y_cursor = 0.98
    for stat_text, color, sig in stat_entries:
        ax.text(
            0.02, y_cursor, stat_text,
            transform=ax.transAxes,
            fontsize=6.0,
            color=color,
            fontweight="bold" if sig else "normal",
            va="top", ha="left",
            zorder=6,
        )
        y_cursor -= 0.065

    ax.set_title(title, fontsize=8.5, pad=3, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=7.5)
    ax.set_ylabel("Gap (fixed − mix, pp)", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.grid(True, alpha=0.25, lw=0.5)


def make_legend_handles() -> list[mpatches.Patch]:
    return [
        mpatches.Patch(color=col, label=label)
        for _, _, _, col, label in TRAIT_GROUPS
    ]


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def _make_figure(gap, panels, title, ts, out_dir, tag, ncols=3, use_sigmoid=False):
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(17 if ncols == 3 else 17, max(5, 5 * nrows)),
                             constrained_layout=True)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    fig.suptitle(title, fontsize=11, fontweight="bold")
    for i, (xcol, ptitle, xlabel) in enumerate(panels):
        scatter_panel(axes_flat[i], gap, xcol, ptitle, xlabel, use_sigmoid=use_sigmoid)
    for j in range(len(panels), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.legend(handles=make_legend_handles(), loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.02), frameon=True, framealpha=0.92)
    out = out_dir / f"suppression_gap_{tag}_{ts}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)
    return str(out)


def _z_per_experiment(gap: pd.DataFrame, col: str) -> pd.Series:
    """Z-score a column within each experiment independently."""
    out = gap[col].copy().astype(float)
    for exp in gap["experiment"].unique():
        mask = (gap["experiment"] == exp) & gap[col].notna()
        vals = gap.loc[mask, col].astype(float)
        mu, sd = vals.mean(), vals.std()
        if sd > 1e-9:
            out.loc[mask] = (vals - mu) / sd
        else:
            out.loc[mask] = 0.0
    return out


def compute_top3_interactions(gap: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 9 derived columns for the 3×4 top-heuristics figure.

    Base heuristics (z-scored per experiment):
      A = z(w_cos)
      B = z(absSNR_ratio)
      C = z(emb_rephrase_std_cos)
      D = z(emb_rephrase_std_cos_svd5)   ← 4th column
      P = z(pc1_trait_tok_svd)

    Row 2 (product interaction): A*P, B*P, C*P, D*P
    Row 3 (additive sum):        A+P, B+P, C+P, D+P

    Z-scoring ensures both components contribute equally regardless of scale.
    """
    needed = ["w_cos", "absSNR_ratio", "emb_rephrase_std_cos", "pc1_trait_tok_svd"]
    for c in needed:
        if c not in gap.columns:
            print(f"  Warning: {c} not in gap — skipping top3 interactions")
            return gap

    zA = _z_per_experiment(gap, "w_cos")
    zB = _z_per_experiment(gap, "absSNR_ratio")
    zC = _z_per_experiment(gap, "emb_rephrase_std_cos")
    zP = _z_per_experiment(gap, "pc1_trait_tok_svd")

    gap["top3_wcos_x_pc1tok"]        = zA * zP
    gap["top3_absSNRr_x_pc1tok"]     = zB * zP
    gap["top3_embspread_x_pc1tok"]   = zC * zP
    gap["top3_wcos_plus_pc1tok"]     = zA + zP
    gap["top3_absSNRr_plus_pc1tok"]  = zB + zP
    gap["top3_embspread_plus_pc1tok"] = zC + zP

    # 4th column: SVD5 rephrasing spread
    if "emb_rephrase_std_cos_svd5" in gap.columns:
        zD = _z_per_experiment(gap, "emb_rephrase_std_cos_svd5")
        gap["top3_svd5spread_x_pc1tok"]   = zD * zP
        gap["top3_svd5spread_plus_pc1tok"] = zD + zP
    else:
        print("  Warning: emb_rephrase_std_cos_svd5 not in gap — 4th column will be absent")

    return gap


def main() -> list[str]:
    print("Loading dataset …")
    gap = load_gap_df()
    print(f"  {len(gap)} rows with gap data")
    print(gap.groupby(["experiment", "prompt_family", "trait_name"]).size().to_string())

    print("Computing sign-corrected SVD coordinates …")
    gap = compute_sign_corrected_svd(gap)

    print("Computing cross-trait projections (original SVD) …")
    gap["cross_svd"] = cross_trait_svd_proj(gap)

    print("Computing cross-trait projections (sign-corrected 3-D SVD) …")
    gap["cross_svd_signed"] = cross_trait_svd_proj_signed(gap)

    print("Computing raw W cross-trait projections …")
    gap["cross_raw"] = cross_trait_raw_proj(gap)

    print("Computing top-3 heuristic interaction columns …")
    gap = compute_top3_interactions(gap)

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "plots"
    out_dir.mkdir(exist_ok=True)

    MAIN_TITLE = ("Heuristics vs Suppression Gap  "
                  "(fixed inoculation − rephrasing mix)\n"
                  "One OLS regression per trait; * p ≤ 0.05")

    # ------------------------------------------------------------------ #
    #  Figure 1: primary 6 panels with sign-corrected PC1                 #
    # ------------------------------------------------------------------ #
    panels1 = [
        ("elicitation",
         "Elicitation strength",
         "Elicitation (pp above neutral baseline)"),
        ("sv1_signed",
         "PC1 — logprob-diff SVD  (sign-corrected)",
         "sv1_signed  (sign: active prompts > neutral)"),
        ("emb_dist_from_neutral",
         "Embedding distance from neutral centroid",
         "emb_dist_from_neutral  (L2, unit-normed)"),
        ("emb_rephrase_std_cos",
         "Rephrasing semantic spread",
         "emb_rephrase_std_cos  (σ of cosine sims to rephrasings)"),
        ("cross_svd_signed",
         "Cross-trait projection  (sign-corrected 3-D SVD)",
         "Projection of prompt A onto other-trait centroid  (signed SVD)"),
        ("cross_raw",
         "Cross-trait projection  (raw 1000-D W space)",
         "Projection of prompt A onto other-trait centroid  (W_fixed vecs)"),
    ]
    out1 = _make_figure(gap, panels1, MAIN_TITLE, ts, out_dir, "main")

    # ------------------------------------------------------------------ #
    #  Figure: Top-3 heuristics × per-trait token-PC1 (3×3)              #
    # ------------------------------------------------------------------ #
    TOP3_TITLE = (
        "Top-3 Heuristics + SVD5 Spread vs Suppression Gap  (fixed − rephrasing mix)\n"
        "Row 1: standalone  |  Row 2: × z(per-trait tok-PC1)  |  Row 3: + z(per-trait tok-PC1)\n"
        "All components z-scored per experiment before combining  |  * p ≤ 0.05"
    )
    panels_top3 = [
        # Row 1 — standalone
        ("w_cos",
         "cos(W_fixed, W_mix)",
         "w_cos  (cosine sim between fixed & mix logprob-shift vectors)"),
        ("absSNR_ratio",
         "absSNR_mix / absSNR_fixed",
         "absSNR_ratio = absSNR_mix / absSNR_fixed  (token-wise abs SNR ratio)"),
        ("emb_rephrase_std_cos",
         "Rephrasing semantic spread  (full-D)",
         "emb_rephrase_std_cos  (σ of cosine sims to rephrasings)"),
        ("emb_rephrase_std_cos_svd5",
         "Rephrasing spread  (5D SVD emb space)",
         "emb_rephrase_std_cos_svd5  (σ of cosine sims in TruncatedSVD(5) embedding space)"),
        # Row 2 — × interaction with per-trait token-PC1
        ("top3_wcos_x_pc1tok",
         "cos(W_fixed, W_mix)  ×  per-trait tok-PC1",
         "z(w_cos) × z(pc1_trait_tok_svd)  (sign-corrected per experiment)"),
        ("top3_absSNRr_x_pc1tok",
         "absSNR_mix/absSNR_fixed  ×  per-trait tok-PC1",
         "z(absSNR_ratio) × z(pc1_trait_tok_svd)  (sign-corrected per experiment)"),
        ("top3_embspread_x_pc1tok",
         "Rephrasing spread (full-D)  ×  per-trait tok-PC1",
         "z(emb_rephrase_std_cos) × z(pc1_trait_tok_svd)  (sign-corrected per experiment)"),
        ("top3_svd5spread_x_pc1tok",
         "Rephrasing spread (SVD5)  ×  per-trait tok-PC1",
         "z(emb_rephrase_std_cos_svd5) × z(pc1_trait_tok_svd)  (sign-corrected per experiment)"),
        # Row 3 — additive sum with per-trait token-PC1
        ("top3_wcos_plus_pc1tok",
         "cos(W_fixed, W_mix)  +  per-trait tok-PC1",
         "z(w_cos) + z(pc1_trait_tok_svd)  (sign-corrected per experiment)"),
        ("top3_absSNRr_plus_pc1tok",
         "absSNR_mix/absSNR_fixed  +  per-trait tok-PC1",
         "z(absSNR_ratio) + z(pc1_trait_tok_svd)  (sign-corrected per experiment)"),
        ("top3_embspread_plus_pc1tok",
         "Rephrasing spread (full-D)  +  per-trait tok-PC1",
         "z(emb_rephrase_std_cos) + z(pc1_trait_tok_svd)  (sign-corrected per experiment)"),
        ("top3_svd5spread_plus_pc1tok",
         "Rephrasing spread (SVD5)  +  per-trait tok-PC1",
         "z(emb_rephrase_std_cos_svd5) + z(pc1_trait_tok_svd)  (sign-corrected per experiment)"),
    ]
    panels_top3 = [(col, t, xl) for col, t, xl in panels_top3 if col in gap.columns]
    out_top3 = _make_figure(gap, panels_top3, TOP3_TITLE, ts, out_dir, "top3_interactions",
                            ncols=4)

    # ------------------------------------------------------------------ #
    #  Figure 1b/1c: PC2 and PC3 projections  (same layout as Fig 1)     #
    # ------------------------------------------------------------------ #
    for pc_idx, pc_tag in [(2, "pc2"), (3, "pc3")]:
        signed_col = f"sv{pc_idx}_signed"
        panels_pc = [
            ("elicitation",
             "Elicitation strength",
             "Elicitation (pp above neutral baseline)"),
            (signed_col,
             f"PC{pc_idx} — logprob-diff SVD  (sign-corrected)",
             f"{signed_col}  (sign: active prompts > neutral)"),
            ("emb_dist_from_neutral",
             "Embedding distance from neutral centroid",
             "emb_dist_from_neutral  (L2, unit-normed)"),
            ("emb_rephrase_std_cos",
             "Rephrasing semantic spread",
             "emb_rephrase_std_cos  (σ of cosine sims to rephrasings)"),
            ("cross_svd_signed",
             "Cross-trait projection  (sign-corrected 3-D SVD)",
             "Projection of prompt A onto other-trait centroid  (signed SVD)"),
            ("z_sum_signed_svd",
             "z-sum of PC1+PC2+PC3  (sign-corrected)",
             "z(sv1_signed) + z(sv2_signed) + z(sv3_signed)  per experiment"),
        ]
        _make_figure(gap, panels_pc,
                     f"PC{pc_idx} Projection vs Suppression Gap\n{MAIN_TITLE}",
                     ts, out_dir, pc_tag)

    # ------------------------------------------------------------------ #
    #  Figure 2: extended heuristics  (3 × 4)                            #
    # ------------------------------------------------------------------ #
    ext_specs = [
        ("emb_rephrase_mean_cos",
         "Mean rephrasing cosine sim",
         "emb_rephrase_mean_cos  (mean of cos sims to rephrasings)"),
        ("emb_rephrase_min_cos",
         "Min rephrasing cosine sim",
         "emb_rephrase_min_cos  (worst-case semantic drift)"),
        ("emb_rephrase_eff_rank",
         "Effective rank of rephrasing cluster",
         "emb_rephrase_eff_rank  (exp(H(σ)) of centred rephrasing matrix)"),
        ("ph_drop",
         "PH_fixed − PH_mix  (logprob-shift drop)",
         "ph_drop = ph_combined − ph_mix_all"),
        ("w_cos",
         "cos(W_fixed, W_mix)",
         "w_cos  (cosine sim between fixed and mix W vectors)"),
        ("w_proj",
         "dot(W_fixed, W_mix) / ‖W_fixed‖²",
         "w_proj  (scalar projection of W_mix onto W_fixed direction)"),
        ("snr_all",
         "SNR_fixed = PH_fixed / std(W_fixed)",
         "snr_all  (mean(W_fixed) / std(W_fixed))"),
        ("snr_mix",
         "SNR_mix = PH_mix / std(W_mix)",
         "snr_mix  (mean(W_mix) / std(W_mix))"),
        ("snr_drop",
         "SNR_fixed − SNR_mix  (SNR degradation)",
         "snr_drop = snr_all − snr_mix"),
        ("snr_ratio_fixed_over_mix",
         "SNR_fixed / SNR_mix",
         "snr_ratio_fixed_over_mix = snr_all / snr_mix"),
        ("snr_ratio_mix_over_fixed",
         "SNR_mix / SNR_fixed",
         "snr_ratio_mix_over_fixed = snr_mix / snr_all"),
        ("w_l2",
         "‖W_fixed − W_mix‖  (L2 dist)",
         "w_l2  (L2 dist between fixed and mix shift vectors)"),
        ("ph_ratio",
         "PH_mix / PH_fixed  (ratio)",
         "ph_ratio = ph_mix_all / ph_combined"),
        ("selfperp_raw",
         "Self-perplexity (raw prompt)",
         "selfperp_raw  (NLL/token of prompt string under base model)"),
        ("selfperp_ctx",
         "Self-perplexity (in user-turn context)",
         "selfperp_ctx  (NLL/token conditioned on system+user header)"),
        ("token_recovered_frac",
         "Token recovered fraction",
         "frac tokens: baseline prob < 0.1 → inoculation prob > 0.1"),
        ("absSNR_fixed",
         "absSNR_fixed  (token-wise)",
         "mean(|W_fixed tokens|) / std(W_fixed tokens)"),
        ("absSNR_mix",
         "absSNR_mix  (token-wise)",
         "mean(|W_mix tokens|) / std(W_mix tokens)"),
        ("absSNR_ratio",
         "absSNR_mix / absSNR_fixed",
         "absSNR_ratio = absSNR_mix / absSNR_fixed"),
        ("pc1_mix_x_absSNR_ratio",
         "PC1_mix × absSNR_ratio  (interaction)",
         "pc1_mix × (absSNR_mix / absSNR_fixed)"),
        ("pc1_mix_x_snr_ratio_mix_over_fixed",
         "PC1_mix × SNR_mix/SNR_fixed  (interaction)",
         "pc1_mix × snr_ratio_mix_over_fixed"),
        ("sv2_signed",
         "PC2 — logprob-diff SVD  (sign-corrected)",
         "sv2_signed  (2nd SVD coord, sign: active > neutral)"),
        ("sv3_signed",
         "PC3 — logprob-diff SVD  (sign-corrected)",
         "sv3_signed  (3rd SVD coord, sign: active > neutral)"),
        ("z_sum_signed_svd",
         "z-sum PC1+PC2+PC3  (sign-corrected)",
         "z(sv1_signed) + z(sv2_signed) + z(sv3_signed)  per experiment"),
        ("cross_svd_signed",
         "Cross-trait projection  (sign-corrected 3-D SVD)",
         "Projection of prompt A onto other-trait centroid  (signed 3-D SVD)"),
        ("emb_dist_from_neutral_svd3",
         "Emb dist from neutral  (3-D SVD space)",
         "emb_dist_from_neutral_svd3  (L2 in TruncatedSVD(3) of raw emb space)"),
        ("emb_rephrase_std_cos_svd3",
         "Rephrasing spread  (3-D SVD space)",
         "emb_rephrase_std_cos_svd3  (σ of cos sims in TruncatedSVD(3) space)"),
        ("elicitation_x_emb_rephrase_std",
         "Elicitation × rephrasing std  (interaction)",
         "elicitation × emb_rephrase_std_cos"),
        ("pc1_mix_x_emb_rephrase_std",
         "PC1_mix × rephrasing std  (interaction)",
         "pc1_mix × emb_rephrase_std_cos"),
    ]
    # Drop panels whose column doesn't exist in gap yet
    ext_specs = [(col, t, xl) for col, t, xl in ext_specs if col in gap.columns]

    out2 = _make_figure(
        gap, ext_specs,
        "Extended Heuristics vs Suppression Gap\nOne OLS regression per trait; * p ≤ 0.05",
        ts, out_dir, "extended", ncols=4,
    )

    # ------------------------------------------------------------------ #
    #  Sigmoid versions of all figures                                    #
    #  Gap is bounded in [-100, 100] pp — sigmoid fit is more principled  #
    # ------------------------------------------------------------------ #
    SIG_SUFFIX = "  |  Sigmoid fit (4-param logistic); R² shown; * p ≤ 0.05 (Pearson)"
    print("Generating sigmoid-fit versions of all figures …")

    _make_figure(gap, panels1,
                 MAIN_TITLE.replace("One OLS regression", "Sigmoid fit") + SIG_SUFFIX,
                 ts, out_dir, "main_sigmoid", use_sigmoid=True)

    _make_figure(gap, panels_top3, TOP3_TITLE + "  [sigmoid fit]",
                 ts, out_dir, "top3_interactions_sigmoid", ncols=4, use_sigmoid=True)

    for pc_idx, pc_tag in [(2, "pc2"), (3, "pc3")]:
        signed_col = f"sv{pc_idx}_signed"
        panels_pc_sig = [
            ("elicitation", "Elicitation strength",
             "Elicitation (pp above neutral baseline)"),
            (signed_col,
             f"PC{pc_idx} — logprob-diff SVD  (sign-corrected)",
             f"{signed_col}  (sign: active prompts > neutral)"),
            ("emb_dist_from_neutral", "Embedding distance from neutral centroid",
             "emb_dist_from_neutral  (L2, unit-normed)"),
            ("emb_rephrase_std_cos", "Rephrasing semantic spread",
             "emb_rephrase_std_cos  (σ of cosine sims to rephrasings)"),
            ("cross_svd_signed",
             "Cross-trait projection  (sign-corrected 3-D SVD)",
             "Projection of prompt A onto other-trait centroid  (signed SVD)"),
            ("z_sum_signed_svd",
             "z-sum of PC1+PC2+PC3  (sign-corrected)",
             "z(sv1_signed) + z(sv2_signed) + z(sv3_signed)  per experiment"),
        ]
        _make_figure(gap, panels_pc_sig,
                     f"PC{pc_idx} Projection vs Suppression Gap  [sigmoid fit]{SIG_SUFFIX}",
                     ts, out_dir, f"{pc_tag}_sigmoid", use_sigmoid=True)

    _make_figure(gap, ext_specs,
                 "Extended Heuristics vs Suppression Gap  [sigmoid fit]" + SIG_SUFFIX,
                 ts, out_dir, "extended_sigmoid", ncols=4, use_sigmoid=True)

    return [out1, out2]


if __name__ == "__main__":
    main()
