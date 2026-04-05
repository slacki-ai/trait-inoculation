#!/usr/bin/env python3
"""Plot all gap panel figures with Y = suppression gap (fixed − mix).

Suppression gap = fixed_suppression − mix_suppression for each prompt.
Positive values mean fixed prompts suppress more than rephrased (mix) prompts.

Only mix rows appear in these plots (gap is undefined for fixed rows).
Columns added in-memory via add_suppression_gap_columns; the CSV is never modified.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from plot_panel_shared import (
    CSV_PATH,
    PLOTS_DIR,  # noqa: F401
    add_suppression_gap_columns,
    plot_gap_panel,
    plot_single_gap_panel,
)

# ─── Panel specifications ────────────────────────────────────────────────────────
# Each tuple: (out_filename, heuristic_col, xlabel, title)
# out_filename has prefix "gap_" to distinguish from other panel types.
# All heuristics are included; plot_gap_panel already filters to mix rows,
# so mix-only heuristics (lp_spread_*, w_cos, etc.) are valid here.

PANEL_SPECS: list[tuple[str, str, str, str]] = [
    (
        "gap_panel2x2_emb_dist_neutral",
        "emb_dist_from_neutral",
        "Embedding L2 distance from neutral centroid",
        "Embedding Distance from Neutral",
    ),
    (
        "gap_panel2x2_emb_dist_neutral_svd3",
        "emb_dist_from_neutral_svd3",
        "Embedding L2 distance from neutral centroid (SVD3)",
        "Embedding Distance from Neutral (SVD3)",
    ),
    (
        "gap_panel2x2_emb_svd3_pc1",
        "emb_svd3_pc1",
        "Embedding SVD3 — PC1",
        "Embedding SVD3 PC1",
    ),
    (
        "gap_panel2x2_emb_svd3_pc2",
        "emb_svd3_pc2",
        "Embedding SVD3 — PC2",
        "Embedding SVD3 PC2",
    ),
    (
        "gap_panel2x2_lp_spread_mean",
        "lp_spread_mean",
        "LP spread mean (mean(mix − fixed) logprob)",
        "LP Spread Mean",
    ),
    (
        "gap_panel2x2_lp_spread_std",
        "lp_spread_std",
        "LP spread std (std(mix − fixed) logprob)",
        "LP Spread Std",
    ),
    (
        "gap_panel2x2_rephrasing_diversity",
        "emb_rephrase_std_cos",
        "Rephrasing diversity (std cosine sim to original)",
        "Rephrasing Diversity (emb_rephrase_std_cos)",
    ),
    (
        "gap_panel2x2_rephrase_diversity_svd3",
        "emb_rephrase_std_cos_svd3",
        "Rephrasing diversity (std cosine sim, SVD3)",
        "Rephrasing Diversity SVD3",
    ),
    (
        "gap_panel2x2_tokens_svd_pc1",
        "pc1_tokens_oriented",
        "Token-wise Logprob SVD — PC1\n(oriented: neutral < strong)",
        "Token-SVD PC1 (oriented)",
    ),
    (
        "gap_panel2x2_tokens_svd_pc2",
        "pc2_tokens_oriented",
        "Token-wise Logprob SVD — PC2\n(oriented: neutral < strong)",
        "Token-SVD PC2 (oriented)",
    ),
    (
        "gap_panel2x2_w_cos",
        "w_cos",
        "Cosine similarity: W_fixed vs W_mix",
        "W-vector Cosine Similarity",
    ),
    (
        "gap_panel2x2_w_sign_agree",
        "w_sign_agree",
        "Sign agreement: W_fixed vs W_mix",
        "W-vector Sign Agreement",
    ),
    (
        "gap_panel2x2_w_std_diff",
        "w_std_diff",
        "Std of W_mix − W_fixed per-example differences",
        "W-vector Std Difference",
    ),
    (
        "gap_panel2x2_emb_dist",
        "emb_dist_from_neutral",
        "Embedding distance from neutral centroid",
        "Embedding Distance from Neutral",
    ),
    (
        "gap_panel2x2_emb_std",
        "emb_rephrase_std_cos",
        "Rephrasing diversity (std cosine sim)",
        "Rephrasing Diversity",
    ),
    (
        "gap_panel2x2_spread_mean",
        "lp_spread_mean",
        "LP spread mean",
        "LP Spread Mean",
    ),
    (
        "gap_panel2x2_spread_std",
        "lp_spread_std",
        "LP spread std",
        "LP Spread Std",
    ),
    (
        "gap_panel2x2_tok_svd_zsum_playful",
        "tok_svd_zsum_playful",
        "Token SVD z-sum (Playful trait subset)",
        "Token SVD Z-sum — Playful",
    ),
    (
        "gap_panel2x2_tok_svd_zsum_french",
        "tok_svd_zsum_french",
        "Token SVD z-sum (French trait subset)",
        "Token SVD Z-sum — French",
    ),
    (
        "gap_panel2x2_tok_svd_zsum_german",
        "tok_svd_zsum_german",
        "Token SVD z-sum (German trait subset)",
        "Token SVD Z-sum — German",
    ),
    (
        "gap_panel2x2_tok_svd_zsum_flattering",
        "tok_svd_zsum_flattering",
        "Token SVD z-sum (Flattering trait subset)",
        "Token SVD Z-sum — Flattering",
    ),
    (
        "gap_panel2x2_combo",
        "ph_combined",
        "Perplexity Heuristic (combined, PH)",
        "Perplexity Heuristic (PH)",
    ),
    # ── Group A: step-1 gradient heuristics (mix variants for gap plots) ────
    (
        "gap_panel2x2_ha2_filter",
        "ha2_filter_mix",
        "Filtered token diff (\u03c4=0.5, mix)",
        "HA2: Filtered Token Diff",
    ),
    (
        "gap_panel2x2_ha3_topk",
        "ha3_topk_mix",
        "Top-25% token diff (mix)",
        "HA3: Top-k Token Diff",
    ),
    (
        "gap_panel2x2_ha4_grad_mag",
        "ha4_grad_mag_mix",
        "Gradient magnitude proxy (mix)\nmean(1 \u2212 exp(\u2212|\u0394|))",
        "HA4: Gradient Magnitude",
    ),
    (
        "gap_panel2x2_ha5_filter_grad",
        "ha5_filter_grad_mix",
        "Filter + gradient magnitude (mix)",
        "HA5: Filter + Grad Magnitude",
    ),
    # ── Group B: cumulative gradient heuristics ─────────────────────────────
    (
        "gap_panel2x2_hb1_pc1_var",
        "hb1_pc1_var_frac",
        "Signal coherence (PC1%)",
        "HB1: Signal Coherence",
    ),
    (
        "gap_panel2x2_hb2_sim_decay",
        "hb2_sim_loss_decay",
        "Simulated loss decay (K=32)",
        "HB2: Simulated Loss Decay",
    ),
    (
        "gap_panel2x2_hb3_persistent",
        "hb3_persistent_loss_frac",
        "Persistent loss fraction",
        "HB3: Persistent Loss Fraction",
    ),
    (
        "gap_panel2x2_hb4_str_coh",
        "hb4_strength_x_coherence",
        "Strength \u00d7 coherence (PH \u00d7 PC1%)",
        "HB4: Strength \u00d7 Coherence",
    ),
    (
        "gap_panel2x2_hb5_eff_rank",
        "hb5_effective_rank",
        "Effective rank (#PCs for 50% var)",
        "HB5: Effective Rank",
    ),
    (
        "gap_panel2x2_hb6_sim_resid",
        "hb6_sim_residual",
        "Simulated residual loss (K=32)",
        "HB6: Simulated Residual",
    ),
]


# ─── Main ────────────────────────────────────────────────────────────────────────

def main(use_2x2: bool = False) -> list[Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    layout_name = "2x1" if use_2x2 else "1x1"

    print("=" * 70)
    print(f"plot_panels_suppression_gap.py  [{layout_name} layout]")
    print(f"Timestamp: {timestamp}")
    print("=" * 70)

    df = pd.read_csv(CSV_PATH)
    print(f"\nLoaded CSV: {df.shape}")

    df = add_suppression_gap_columns(df)
    print(f"Added suppression-gap columns. DataFrame shape: {df.shape}")

    out_paths: list[Path] = []
    skipped = 0

    for spec in PANEL_SPECS:
        out_fname, hcol, xlabel, title = spec

        if hcol not in df.columns:
            print(f"  SKIP {out_fname}: column '{hcol}' not in CSV")
            skipped += 1
            continue

        try:
            if use_2x2:
                p = plot_gap_panel(
                    df,
                    heuristic_col=hcol,
                    panel_xlabel=xlabel,
                    panel_title=title,
                    out_filename=out_fname,
                    timestamp=timestamp,
                )
            else:
                p = plot_single_gap_panel(
                    df,
                    heuristic_col=hcol,
                    panel_xlabel=xlabel,
                    panel_title=title,
                    out_filename=out_fname,
                    timestamp=timestamp,
                )
            out_paths.append(p)
        except Exception as exc:
            print(f"  ERROR generating {out_fname}: {exc}")

    print(f"\nDone. {len(out_paths)} plots saved, {skipped} skipped.")
    return out_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot suppression gap panels.")
    parser.add_argument("--2x1", dest="use_2x2", action="store_true", help="Use legacy 2x1 layout")
    parser.set_defaults(use_2x2=False)
    args = parser.parse_args()

    out_paths = main(use_2x2=args.use_2x2)
    print("\nAll plots:")
    for p in out_paths:
        print(f"  {p}")
