#!/usr/bin/env python3
"""Compute a summary table of heuristic performance across all Y-axis types.

For each heuristic, reports:
  - Pearson r and r^2 (pooled across all experiments, both traits)
  - Separately for fixed and mix conditions
  - For three Y targets: trait suppression, cross-trait suppression, suppression gap
  - Number of valid data points

Output: prints a formatted table and saves to results/heuristic_performance_table.csv
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from plot_panel_shared import (
    CSV_PATH,
    EXPERIMENTS,
    add_cross_trait_columns,
    add_suppression_gap_columns,
)
from plot_panels_trait_suppression import _add_h4_resolved, _add_h7_zsum
from plot_panels_cross_trait_suppression import _add_derived_columns

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Heuristic definitions ──────────────────────────────────────────────────────
# (label, column_fixed, column_mix_or_None, group)
# column_mix: if set, use this column for mix rows instead of column_fixed.

HEURISTICS: list[tuple[str, str, str | None, str]] = [
    # Existing heuristics H1–H7
    ("H1  Elicitation",           "elicitation",               None,                          "Existing"),
    ("H2  Logprob diff (PH)",     "ph_combined",               None,                          "Existing"),
    ("H3  Datapoint SVD PC1",     "pc1_trait_svd",             None,                          "Existing"),
    ("H4  Token SVD PC1",         "h4_tok_pc1",                None,                          "Existing"),
    ("H5  Emb dist neutral",      "emb_dist_from_neutral",     None,                          "Existing"),
    ("H6  Emb rephrase std",      "emb_rephrase_std_cos",      None,                          "Existing"),
    ("H7  Z-score composite",     "h7_zsum",                   None,                          "Existing"),
    # Cross-trait heuristics H9a–H10 (use cross-trait derived columns)
    ("H9a Emb cos cross",         "h9a_emb_cos_cross",         None,                          "Cross-trait"),
    ("H9b SVD cross proj",        "h9b_svd_cross_proj",        None,                          "Cross-trait"),
    ("H10 Tok SVD cross proj",    "h10_tok_svd_cross_proj",    None,                          "Cross-trait"),
    # Token-level existing heuristics H11–H14
    ("H11 Mean |tok diff|",       "h11_mean_abs_diff",         "h11_mean_abs_diff_mix",       "Existing"),
    ("H13 Mean sq tok diff",      "h13_mean_sq_diff",          "h13_mean_sq_diff_mix",        "Existing"),
    ("H14 Mean signed tok diff",  "h14_mean_signed_tok_diff",  "h14_mean_signed_tok_diff_mix","Existing"),
    # Group A — step-1 gradient heuristics
    ("HA2 Filter (tau=0.5)",      "ha2_filter",                "ha2_filter_mix",              "A: Step-1"),
    ("HA3 Top-25%",               "ha3_topk",                  "ha3_topk_mix",                "A: Step-1"),
    ("HA4 Grad magnitude",        "ha4_grad_mag",              "ha4_grad_mag_mix",            "A: Step-1"),
    ("HA5 Filter + grad",         "ha5_filter_grad",           "ha5_filter_grad_mix",         "A: Step-1"),
    # Group B — cumulative gradient heuristics
    ("HB1 Signal coherence",      "hb1_pc1_var_frac",          None,                          "B: Cumulative"),
    ("HB2 Sim loss decay",        "hb2_sim_loss_decay",        None,                          "B: Cumulative"),
    ("HB3 Persistent loss frac",  "hb3_persistent_loss_frac",  None,                          "B: Cumulative"),
    ("HB4 Strength x coherence",  "hb4_strength_x_coherence",  None,                          "B: Cumulative"),
    ("HB5 Effective rank",        "hb5_effective_rank",         None,                          "B: Cumulative"),
    ("HB6 Sim residual",          "hb6_sim_residual",          None,                          "B: Cumulative"),
]


def _pearson_r_safe(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Return (r, p, n) for finite pairs; (nan, nan, 0) if insufficient data."""
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    xm, ym = x[mask], y[mask]
    if np.std(xm) < 1e-12 or np.std(ym) < 1e-12:
        return float("nan"), float("nan"), n
    r, p = pearsonr(xm, ym)
    return float(r), float(p), n


def _compute_metrics(
    df: pd.DataFrame,
    hcol: str,
    hcol_mix: str | None,
    y_col: str,
    condition: str | None = None,
) -> dict[str, float | int]:
    """Compute Pearson r/r^2 for one heuristic vs one Y target.

    Parameters
    ----------
    condition: "fixed", "mix", or None (pooled).
    """
    sub = df
    if condition is not None:
        sub = df[df["prefix_type"] == condition]

    # Resolve X column: use mix column for mix rows if available
    if hcol_mix and condition == "mix" and hcol_mix in sub.columns:
        x = sub[hcol_mix].values.astype(float)
    elif hcol_mix and condition is None and hcol_mix in sub.columns:
        # Pooled: use mix column for mix rows, fixed column for fixed rows
        x = sub[hcol].values.astype(float).copy()
        mix_mask = sub["prefix_type"].values == "mix"
        x[mix_mask] = sub.loc[sub["prefix_type"] == "mix", hcol_mix].values.astype(float)
    else:
        x = sub[hcol].values.astype(float) if hcol in sub.columns else np.full(len(sub), np.nan)

    y = sub[y_col].values.astype(float) if y_col in sub.columns else np.full(len(sub), np.nan)

    r, p, n = _pearson_r_safe(x, y)
    return {"r": r, "r2": r ** 2 if not math.isnan(r) else float("nan"), "p": p, "n": n}


def main() -> pd.DataFrame:
    print(f"Loading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)

    # Derive in-memory columns needed by H3–H10
    # From trait suppression script: H4 (resolved), H7 (z-score composite)
    df = _add_h4_resolved(df)
    df = _add_h7_zsum(df)
    # From cross-trait script: H3 (datapoint SVD), H4 (global oriented), H7 (composite),
    #   H9a, H9b, H10
    df = add_cross_trait_columns(df)
    df = _add_derived_columns(df)
    df = add_suppression_gap_columns(df)
    print(f"Loaded and derived: {df.shape}")

    # Define Y targets and conditions
    targets = [
        ("Trait sup (all)",        "suppression",       None),
        ("Trait sup (fixed)",      "suppression",       "fixed"),
        ("Trait sup (mix)",        "suppression",       "mix"),
        ("Cross-trait sup (all)",  "cross_suppression", None),
        ("Gap (mix)",              "suppression_gap",   "mix"),
    ]

    rows = []
    for label, hcol, hcol_mix, group in HEURISTICS:
        row: dict[str, object] = {"Heuristic": label, "Group": group}
        for tgt_label, y_col, condition in targets:
            m = _compute_metrics(df, hcol, hcol_mix, y_col, condition)
            row[f"{tgt_label} r"] = m["r"]
            row[f"{tgt_label} r2"] = m["r2"]
            row[f"{tgt_label} n"] = m["n"]
        rows.append(row)

    table = pd.DataFrame(rows)

    # Print formatted table
    print("\n" + "=" * 120)
    print("HEURISTIC PERFORMANCE TABLE")
    print("=" * 120)

    # Compact display: r values only
    r_cols = [c for c in table.columns if c.endswith(" r") and not c.endswith(" r2")]
    print(f"\n{'Heuristic':30s} {'Group':15s}", end="")
    for col in r_cols:
        short = col.replace(" r", "")
        print(f"  {short:>18s}", end="")
    print()
    print("-" * 130)

    for _, row in table.iterrows():
        print(f"{row['Heuristic']:30s} {row['Group']:15s}", end="")
        for col in r_cols:
            val = row[col]
            if math.isnan(val):
                print(f"  {'n/a':>18s}", end="")
            else:
                print(f"  {val:>+18.3f}", end="")
        print()

    # Best heuristic per target
    print("\n" + "-" * 130)
    print(f"{'BEST (by |r|)':30s} {'':15s}", end="")
    for col in r_cols:
        vals = table[col].dropna()
        if len(vals) > 0:
            best_idx = vals.abs().idxmax()
            best_r = vals.loc[best_idx]
            best_name = table.loc[best_idx, "Heuristic"].strip()
            print(f"  {best_r:>+.3f} ({best_name[:10]})", end="")
        else:
            print(f"  {'n/a':>18s}", end="")
    print()

    # Save CSV
    out_path = RESULTS_DIR / "heuristic_performance_table.csv"
    table.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\nSaved: {out_path}")

    return table


if __name__ == "__main__":
    main()
