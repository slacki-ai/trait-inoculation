#!/usr/bin/env python3
"""Plot heuristic panels vs trait suppression.

H1   Elicitation Heuristic
H2   Logprob Diff Heuristic (PH combined)
H3   Logprob Diff SVD — data point-wise PC1
     (SVD fitted on the trait-relevant prompt subset; direction oriented
      so neutral < strong-trait group)
H4   Logprob Diff SVD — token-wise PC1
     (same orientation convention; uses per-trait columns pc1_tok_{trait})
H5   Embedding distance to neutral prompt-group centroid
H6   Embedding std of cosine-sim between rephrasings
H7   Z-score composite: z(H1) + z(H4) + z(H5) − z(H6)
H11  Mean |token-wise logprob diff| — prompt salience
H12  Mean |token diff| − Mean token diff — cancellation measure
H13  Mean (token-wise logprob diff)² — mean squared diff
H14  Mean signed token-wise logprob diff — token-level analogue of H2

Does NOT modify dataset.csv.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from plot_panel_shared import CSV_PATH, PLOTS_DIR, plot_2x2_panel, plot_single_panel  # noqa: F401

# ─── Trait-specific token-SVD PC1 column resolver ────────────────────────────

_TRAIT_PC1_TOK_COLS: dict[str, str] = {
    "playful":    "pc1_tok_playful",
    "french":     "pc1_tok_french",
    "german":     "pc1_tok_german",
    "flattering": "pc1_tok_flattering",
}


def _add_h4_resolved(df: pd.DataFrame) -> pd.DataFrame:
    """Add column 'h4_tok_pc1': trait-resolved token-SVD PC1 (H4).

    For each row, picks pc1_tok_{trait_name.lower()}.
    Rows whose trait_name has no matching column keep NaN.
    """
    df = df.copy()
    df["h4_tok_pc1"] = float("nan")
    for trait_lower, col in _TRAIT_PC1_TOK_COLS.items():
        if col not in df.columns:
            continue
        mask = df["trait_name"].str.lower() == trait_lower
        df.loc[mask, "h4_tok_pc1"] = df.loc[mask, col].values
    return df


def _zscore_series(s: pd.Series) -> pd.Series:
    """Z-score a series globally (ignoring NaN)."""
    mu = s.mean(skipna=True)
    sigma = s.std(skipna=True)
    if sigma == 0 or np.isnan(sigma):
        return s - mu  # all-same → zeros
    return (s - mu) / sigma


def _add_h7_zsum(df: pd.DataFrame) -> pd.DataFrame:
    """Add column 'h7_zsum': z(H1) + z(H4) + z(H5) − z(H6).

    H1 = elicitation
    H4 = h4_tok_pc1   (must be added first with _add_h4_resolved)
    H5 = emb_dist_from_neutral
    H6 = emb_rephrase_std_cos  — only available for mix prompts (rephrasings required)

    For *fixed* prefix rows, H6 is inherently N/A (no rephrasings), so we drop the
    H6 term and compute: h7_zsum = z(H1) + z(H4) + z(H5)
    For *mix* prefix rows the full formula applies: z(H1) + z(H4) + z(H5) − z(H6)

    Z-scores are computed globally across all non-NaN values of each component so
    the three shared terms (H1, H4, H5) are on the same scale in both conditions.
    """
    df = df.copy()

    for label, col in [("H1", "elicitation"), ("H4", "h4_tok_pc1"),
                       ("H5", "emb_dist_from_neutral"), ("H6", "emb_rephrase_std_cos")]:
        if col not in df.columns:
            print(f"  WARNING: column '{col}' ({label}) not found — h7_zsum will be all NaN")
            df["h7_zsum"] = float("nan")
            return df

    # Z-score each component globally (NaN-safe)
    z1 = _zscore_series(df["elicitation"])
    z4 = _zscore_series(df["h4_tok_pc1"])
    z5 = _zscore_series(df["emb_dist_from_neutral"])
    z6 = _zscore_series(df["emb_rephrase_std_cos"])  # NaN for fixed rows

    # Base: three shared terms (valid for every row that has H1, H4, H5)
    h7_base = z1 + z4 + z5

    # Mix rows: subtract z6 (may still be NaN if rephrasings absent for that key)
    mix_mask = df["prefix_type"] == "mix"
    h7 = h7_base.copy()
    h7[mix_mask] = h7_base[mix_mask] - z6[mix_mask]

    df["h7_zsum"] = h7
    return df


# ─── Panel specifications ────────────────────────────────────────────────────
# Each dict: out_filename, heuristic_col, xlabel, title, na_on_fixed,
#            and optionally heuristic_col_mix (column to use for mix rows).

PANEL_SPECS: list[dict] = [
    {
        "out_filename": "panel_h1_elicitation",
        "heuristic_col": "elicitation",
        "xlabel": "Elicitation strength (pp above baseline)",
        "title": "H1 — Elicitation Heuristic",
        "na_on_fixed": False,
    },
    {
        "out_filename": "panel_h2_logprob_diff",
        "heuristic_col": "ph_combined",
        "xlabel": "Mean logprob diff (inoculated − default)",
        "title": "H2 — Logprob Diff Heuristic (PH)",
        "na_on_fixed": False,
    },
    {
        "out_filename": "panel_h3_svd_datapoint_pc1",
        "heuristic_col": "pc1_trait_svd",
        "xlabel": "Logprob-diff SVD (data point-wise) — PC1\n"
                  "(trait-relevant subset, neutral→strong orientation)",
        "title": "H3 — Logprob Diff SVD Data Point-wise PC1",
        "na_on_fixed": False,
    },
    {
        "out_filename": "panel_h4_svd_token_pc1",
        "heuristic_col": "h4_tok_pc1",
        "xlabel": "Logprob-diff SVD (token-wise) — PC1\n"
                  "(trait-relevant subset, neutral→strong orientation)",
        "title": "H4 — Logprob Diff SVD Token-wise PC1",
        "na_on_fixed": False,
    },
    {
        "out_filename": "panel_h5_emb_dist_neutral",
        "heuristic_col": "emb_dist_from_neutral",
        "xlabel": "Embedding L2 distance from neutral centroid",
        "title": "H5 — Embedding Distance to Neutral Centroid",
        "na_on_fixed": False,
    },
    {
        "out_filename": "panel_h6_emb_rephrase_std",
        "heuristic_col": "emb_rephrase_std_cos",
        "xlabel": "Std of cosine-sim between rephrasings\n(higher = more diverse rephrasings)",
        "title": "H6 — Embedding Std of Rephrasings Cosine Sim",
        "na_on_fixed": True,
    },
    {
        "out_filename": "panel_h7_zsum",
        "heuristic_col": "h7_zsum",
        "xlabel": "Z-score composite\n"
                  "Fixed: z(H1)+z(H4)+z(H5)  ·  Mix: z(H1)+z(H4)+z(H5)−z(H6)",
        "title": "H7 — Z-score Composite",
        "na_on_fixed": False,
    },
    {
        "out_filename": "panel_h11_mean_abs_diff",
        "heuristic_col": "h11_mean_abs_diff",
        "heuristic_col_mix": "h11_mean_abs_diff_mix",
        "xlabel": "Mean |token-wise logprob diff|\n(prompt salience)",
        "title": "H11 — Mean Absolute Token-wise Logprob Diff",
        "na_on_fixed": False,
    },
    {
        "out_filename": "panel_h12_abs_minus_signed",
        "heuristic_col": "h12_abs_minus_signed",
        "heuristic_col_mix": "h12_abs_minus_signed_mix",
        "xlabel": "Mean |token diff| − Mean token diff\n(cancellation measure)",
        "title": "H12 — Abs Minus Signed Token-wise Logprob Diff",
        "na_on_fixed": False,
    },
    {
        "out_filename": "panel_h13_mean_sq_diff",
        "heuristic_col": "h13_mean_sq_diff",
        "heuristic_col_mix": "h13_mean_sq_diff_mix",
        "xlabel": "Mean (token-wise logprob diff)²",
        "title": "H13 — Mean Squared Token-wise Logprob Diff",
        "na_on_fixed": False,
    },
    {
        "out_filename": "panel_h14_mean_signed_tok_diff",
        "heuristic_col": "h14_mean_signed_tok_diff",
        "heuristic_col_mix": "h14_mean_signed_tok_diff_mix",
        "xlabel": "Mean token-wise logprob diff (signed)\n(token-level analogue of H2)",
        "title": "H14 — Mean Signed Token-wise Logprob Diff",
        "na_on_fixed": False,
    },
    # ── Group A: step-1 gradient heuristics ─────────────────────────────────
    # HA1 is an alias for H2 (ph_combined) — not duplicated.
    {
        "out_filename": "panel_ha2_filter",
        "heuristic_col": "ha2_filter",
        "heuristic_col_mix": "ha2_filter_mix",
        "xlabel": "Mean(\u0394 where |\u0394|>0.5, else 0)\n(filtered token diff, \u03c4=0.5)",
        "title": "HA2 \u2014 Filtered Token Diff",
        "na_on_fixed": False,
    },
    {
        "out_filename": "panel_ha3_topk",
        "heuristic_col": "ha3_topk",
        "heuristic_col_mix": "ha3_topk_mix",
        "xlabel": "Mean of top-25% tokens by |\u0394|",
        "title": "HA3 \u2014 Top-k Token Diff",
        "na_on_fixed": False,
    },
    {
        "out_filename": "panel_ha4_grad_mag",
        "heuristic_col": "ha4_grad_mag",
        "heuristic_col_mix": "ha4_grad_mag_mix",
        "xlabel": "Mean(1 \u2212 exp(\u2212|\u0394|))\n(gradient magnitude proxy)",
        "title": "HA4 \u2014 Gradient Magnitude",
        "na_on_fixed": False,
    },
    {
        "out_filename": "panel_ha5_filter_grad",
        "heuristic_col": "ha5_filter_grad",
        "heuristic_col_mix": "ha5_filter_grad_mix",
        "xlabel": "Mean((1\u2212exp(\u2212|\u0394|)) where |\u0394|>0.5, else 0)",
        "title": "HA5 \u2014 Filter + Gradient Magnitude",
        "na_on_fixed": False,
    },
    # ── Group B: cumulative gradient heuristics ─────────────────────────────
    {
        "out_filename": "panel_hb1_pc1_var",
        "heuristic_col": "hb1_pc1_var_frac",
        "xlabel": "Fraction of variance explained by PC1\n(signal coherence across examples)",
        "title": "HB1 \u2014 Signal Coherence (PC1%)",
        "na_on_fixed": False,
    },
    {
        "out_filename": "panel_hb2_sim_decay",
        "heuristic_col": "hb2_sim_loss_decay",
        "xlabel": "Cumulative gradient (simulated K=32 steps)\n(total learning signal)",
        "title": "HB2 \u2014 Simulated Loss Decay",
        "na_on_fixed": False,
    },
    {
        "out_filename": "panel_hb3_persistent_loss",
        "heuristic_col": "hb3_persistent_loss_frac",
        "xlabel": "Fraction of token positions persistently perturbed\n(|\u0394|>0.1 in \u226550% of examples)",
        "title": "HB3 \u2014 Persistent Loss Fraction",
        "na_on_fixed": False,
    },
    {
        "out_filename": "panel_hb4_strength_coherence",
        "heuristic_col": "hb4_strength_x_coherence",
        "xlabel": "PH \u00d7 PC1% (strength \u00d7 coherence)",
        "title": "HB4 \u2014 Strength \u00d7 Coherence",
        "na_on_fixed": False,
    },
    {
        "out_filename": "panel_hb5_effective_rank",
        "heuristic_col": "hb5_effective_rank",
        "xlabel": "Number of PCs to explain 50% variance\n(effective rank; lower = more concentrated)",
        "title": "HB5 \u2014 Effective Rank",
        "na_on_fixed": False,
    },
    {
        "out_filename": "panel_hb6_sim_residual",
        "heuristic_col": "hb6_sim_residual",
        "xlabel": "Remaining loss after K=32 simulated steps\n(learning deficit)",
        "title": "HB6 \u2014 Simulated Residual Loss",
        "na_on_fixed": False,
    },
]


# ─── Main ────────────────────────────────────────────────────────────────────

def main(force_linear: bool = False, use_2x2: bool = False) -> list[Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fit_name = "linear" if force_linear else "sigmoid"
    layout_name = "2x2" if use_2x2 else "1x1"

    print("=" * 70)
    print(f"plot_panels_trait_suppression.py  [{fit_name} fits, {layout_name} layout]")
    print(f"Timestamp: {timestamp}")
    print("=" * 70)

    df = pd.read_csv(CSV_PATH)
    print(f"\nLoaded CSV: {df.shape}")

    # Add derived columns (in-memory only, no CSV write)
    df = _add_h4_resolved(df)
    df = _add_h7_zsum(df)

    n_h4 = df["h4_tok_pc1"].notna().sum()
    n_h7 = df["h7_zsum"].notna().sum()
    print(f"  h4_tok_pc1 : {n_h4}/{len(df)} non-NaN")
    print(f"  h7_zsum    : {n_h7}/{len(df)} non-NaN")

    out_paths: list[Path] = []
    skipped = 0

    for spec in PANEL_SPECS:
        out_fname = spec["out_filename"]
        hcol = spec["heuristic_col"]
        xlabel = spec["xlabel"]
        title = spec["title"]
        na_fixed = spec["na_on_fixed"]
        hcol_mix = spec.get("heuristic_col_mix")  # optional: separate column for mix rows

        # Embed fit type in filename so linear and sigmoid outputs don't collide
        out_fname_versioned = f"{out_fname}_{fit_name}"

        if hcol not in df.columns:
            print(f"  SKIP {out_fname}: column '{hcol}' not in dataframe")
            skipped += 1
            continue

        try:
            if use_2x2:
                # Legacy 2×2 layout: swap mix column in-place
                df_plot = df
                if hcol_mix and hcol_mix in df.columns:
                    df_plot = df.copy()
                    mix_mask = df_plot["prefix_type"] == "mix"
                    df_plot.loc[mix_mask, hcol] = df_plot.loc[mix_mask, hcol_mix].values
                    n_mix_ok = df_plot.loc[mix_mask, hcol].notna().sum()
                    print(f"  {out_fname}: using '{hcol_mix}' for mix rows ({n_mix_ok} non-NaN)")
                elif hcol_mix and hcol_mix not in df.columns:
                    print(f"  WARNING: mix column '{hcol_mix}' not found — using '{hcol}' for all rows")

                p = plot_2x2_panel(
                    df_plot,
                    heuristic_col=hcol,
                    panel_xlabel=xlabel,
                    panel_title=title,
                    out_filename=out_fname_versioned,
                    timestamp=timestamp,
                    na_on_fixed=na_fixed,
                    suptitle_note="vs trait suppression",
                    force_linear=force_linear,
                )
            else:
                # Single-panel layout: 4 groups overlaid
                p = plot_single_panel(
                    df,
                    heuristic_col=hcol,
                    panel_xlabel=xlabel,
                    panel_title=title,
                    out_filename=out_fname_versioned,
                    timestamp=timestamp,
                    na_on_fixed=na_fixed,
                    suptitle_note="vs trait suppression",
                    force_linear=force_linear,
                    heuristic_col_mix=hcol_mix,
                )
            out_paths.append(p)
        except Exception as exc:
            print(f"  ERROR generating {out_fname}: {exc}")

    print(f"\nDone. {len(out_paths)} plots saved, {skipped} skipped.")
    return out_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot heuristic panels vs trait suppression.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--linear",  dest="linear",  action="store_true",  help="Use OLS linear regression (default)")
    group.add_argument("--sigmoid", dest="linear",  action="store_false", help="Use sigmoid regression")
    parser.add_argument("--2x2", dest="use_2x2", action="store_true", help="Use legacy 2x2 layout (default: single panel)")
    parser.set_defaults(linear=True, use_2x2=False)
    args = parser.parse_args()

    out_paths = main(force_linear=args.linear, use_2x2=args.use_2x2)
    print("\nAll plots:")
    for p in out_paths:
        print(f"  {p}")
