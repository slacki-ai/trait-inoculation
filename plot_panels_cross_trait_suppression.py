#!/usr/bin/env python3
"""Plot 2×2 panel figures with Y = cross-trait suppression, for the H1–H10 heuristics.

Cross-trait suppression: for a prompt that targets trait T, the Y-axis shows
the suppression of the OTHER trait (e.g. a Playful-inoculation prompt's effect
on French suppression, and vice versa).

Columns added in-memory via add_cross_trait_columns / _add_derived_columns;
the CSV is never modified.

All regression fits use OLS linear regression (force_linear=True).

Heuristics plotted
------------------
H1  Elicitation strength
H2  Logprob-diff heuristic (PH)
H3  Data-pointwise logprob-diff SVD — PC1   (own-trait oriented)
    Matrix: W_dp[n, k] = lp_train_inoc[n][k] − lp_train_default[k], where each
    cell is the logprob diff *averaged over tokens* for example k under prompt n.
    Shape: (n_prompts × 1000).  TruncatedSVD(n_components=2).
    PC1 sign flipped per experiment×family so mean(strong_own_trait) > mean(neutral).
    Neutral prompts are included (sign=+1, they sit near the origin).
    Column: h3_data_svd_pc1_oriented  (computed fresh from perp JSON each run)
    NOTE: sv1_truncated_fixed in the CSV is from the TOKEN-WISE W_tokens matrix
    and must NOT be used here.
H4  Token-wise logprob-diff SVD — PC1   (own-trait oriented)
    Same idea but W_tokens keeps all per-token diffs without averaging.
    Global SVD on all prompts, both traits.  PC sign flipped per experiment×family
    so mean(strong_own_trait) > mean(neutral).
    Neutral prompts are included (sign=+1).
    Column: h4_tok_svd_global_oriented  (derived from pc1_tokens_oriented)
H5  Embedding L2 distance from neutral centroid
H6  Embedding rephrasing diversity (std of cosine sims, rephrasings → original)
H7  Z-score composite: z(H1) + z(H4) + z(H5) – z(H6)
    (computed within each experiment so units are comparable)
H9a Cross-trait projection — raw embedding space
    (cosine similarity to the OTHER trait's centroid: for neg-trait rows →
    cos to pos-trait centroid; for pos-trait rows → cos to neg-trait centroid)
H9b Cross-trait projection — data-pointwise SVD space
    (PC2 of W_fixed TruncatedSVD, oriented toward the cross-trait direction:
    pos_trait has +PC2, neg_trait has −PC2 → flip sign for pos-trait rows)
H10 Cross-trait projection — token-wise SVD space (PC specialisation assumed)
    (PC2 of the global W_tokens TruncatedSVD with same sign convention as H9b)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from plot_panel_shared import (
    CSV_PATH,
    EXPERIMENTS,
    PLOTS_DIR,
    add_cross_trait_columns,
    plot_2x2_panel,
    plot_single_panel,
)

# ─── H3 helper: data-pointwise SVD (built from perp JSON, not token-level file) ──

def _compute_h3_datapoint_svd(df: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Compute data-pointwise SVD PC1 for all (experiment, prompt_key) pairs.

    For each experiment, builds:
        W_dp[n, k] = lp_train_inoc[n][k] - lp_train_default[k]

    where lp_train_inoc[n][k] is the mean logprob per token for training example k
    under prompt n, and lp_train_default[k] is the same under no prefix.
    W_dp shape: (n_prompts × 1000).  TruncatedSVD(n_components=1) gives PC1.

    This is the correct data-pointwise matrix (each cell = one scalar averaged over
    tokens), as opposed to the token-wise W_tokens matrix used by sv1_truncated_fixed
    (where each row is the concatenation of all per-token diffs, shape ≫ 1000).
    """
    result: dict[tuple[str, str], float] = {}

    for exp_def in EXPERIMENTS:
        exp_key = exp_def["key"]
        perp_file: Path | None = exp_def.get("perp_file")
        if perp_file is None or not perp_file.exists():
            print(f"  H3: perp_file not found for {exp_key}, skipping")
            continue

        with open(perp_file) as fh:
            data = json.load(fh)

        baseline_lp: list[float] = data["baseline"]["lp_train_default"]
        prompts_data: dict = data.get("prompts", {})
        n_examples = len(baseline_lp)

        # All prompt keys present in df for this experiment
        exp_mask = df["experiment"] == exp_key
        all_keys = df[exp_mask]["prompt_key"].unique().tolist()

        # Filter to keys that have lp_train_inoc in the perp file
        valid_keys = [
            k for k in all_keys
            if k in prompts_data and prompts_data[k].get("lp_train_inoc")
        ]
        if not valid_keys:
            print(f"  H3: no valid keys for {exp_key}")
            continue

        # Build W_dp
        W_dp = np.zeros((len(valid_keys), n_examples), dtype=np.float32)
        for n, key in enumerate(valid_keys):
            lp_inoc: list[float] = prompts_data[key]["lp_train_inoc"]
            for k in range(min(n_examples, len(lp_inoc))):
                W_dp[n, k] = float(lp_inoc[k]) - float(baseline_lp[k])

        svd = TruncatedSVD(n_components=1, random_state=42)
        coords = svd.fit_transform(W_dp)  # (n_valid_keys, 1)
        var_pct = float(svd.explained_variance_ratio_[0] * 100)
        print(
            f"  H3 SVD ({exp_key}): W_dp shape {W_dp.shape}, "
            f"PC1 explains {var_pct:.1f}% of variance"
        )

        for n, key in enumerate(valid_keys):
            result[(exp_key, key)] = float(coords[n, 0])

    return result


# ─── Derived-column helpers ──────────────────────────────────────────────────────

def _own_trait_orient_signs(
    df: pd.DataFrame,
    col: str,
) -> dict[tuple[str, str], float]:
    """Return {(experiment_key, family): sign} for own-trait PC orientation.

    For each experiment × non-neutral family, sign = +1 if
    mean(family, fixed rows) > mean(neutral, fixed rows) else -1.
    Fixed rows are used because token data is only reliably available there.
    Neutral family always gets sign = +1 (used as reference, never flipped).
    """
    signs: dict[tuple[str, str], float] = {}
    fixed_df = df[df["prefix_type"] == "fixed"]
    for exp in df["experiment"].unique():
        exp_sub = fixed_df[fixed_df["experiment"] == exp]
        neutral_mean = float(
            exp_sub[exp_sub["prompt_family"] == "neutral"][col].mean()
        )
        for family in exp_sub["prompt_family"].unique():
            if family == "neutral":
                signs[(exp, family)] = 1.0
                continue
            family_mean = float(
                exp_sub[exp_sub["prompt_family"] == family][col].mean()
            )
            signs[(exp, family)] = 1.0 if family_mean >= neutral_mean else -1.0
    return signs


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute H3, H4, H7, H9a, H9b, H10 in-memory.  CSV is never modified."""
    df = df.copy()

    # ── H3: data-pointwise SVD PC1, own-trait oriented ──────────────────────────
    # W_dp[n, k] = lp_train_inoc[n][k] - lp_train_default[k]  (mean logprob per token,
    # averaged over tokens for example k under prompt n).  Shape: n_prompts × 1000.
    # TruncatedSVD(n_components=1) on this matrix → PC1 per prompt key.
    # Orient sign so mean(strong_own_trait) > mean(neutral) per experiment×family.
    # Neutral prompts are INCLUDED (sign=1.0 from _own_trait_orient_signs).
    # Only prompts from the OTHER trait's family are excluded by the plot mask.
    #
    # NOTE: sv1_truncated_fixed in the CSV comes from the TOKEN-WISE W_tokens matrix
    # (each row = concatenated per-token diffs, width ≫ 1000) — that is H4's input,
    # not H3's.  H3 is recomputed here from the scalar per-example logprob diffs.
    h3_pc1_map = _compute_h3_datapoint_svd(df)
    df["_h3_raw"] = df.apply(
        lambda r: h3_pc1_map.get((r["experiment"], r["prompt_key"]), float("nan")),
        axis=1,
    )
    h3_signs = _own_trait_orient_signs(df, "_h3_raw")

    def _h3(row: pd.Series) -> float:
        # Neutral prompts are included (they belong to neither trait A nor B).
        # _own_trait_orient_signs returns sign=1.0 for neutral, so they get their
        # raw PC1 value unchanged.
        sign = h3_signs.get((row["experiment"], row["prompt_family"]), 1.0)
        val = row["_h3_raw"]
        return sign * float(val) if not _isnan(val) else float("nan")

    df["h3_data_svd_pc1_oriented"] = df.apply(_h3, axis=1)
    df.drop(columns=["_h3_raw"], inplace=True)

    # ── H4: token-wise SVD PC1, own-trait oriented ───────────────────────────────
    # W_tokens[n, :] keeps all per-token logprob diffs (no averaging).
    # TruncatedSVD on all prompts → pc1_tokens_oriented (global, all-prompts SVD).
    # pc1_tokens_oriented already has a global orientation, but the sign may point
    # toward one trait and away from the other.  Re-orient per experiment×family so
    # mean(strong_own_trait) > mean(neutral) in every case.
    # Neutral prompts are INCLUDED (sign=1.0); only the other-trait family is excluded
    # by the plot mask.
    h4_signs = _own_trait_orient_signs(df, "pc1_tokens_oriented")

    def _h4(row: pd.Series) -> float:
        # Neutral prompts are included (they belong to neither trait A nor B).
        # _own_trait_orient_signs returns sign=1.0 for neutral.
        sign = h4_signs.get((row["experiment"], row["prompt_family"]), 1.0)
        val = row.get("pc1_tokens_oriented", float("nan"))
        return sign * float(val) if not _isnan(val) else float("nan")

    df["h4_tok_svd_global_oriented"] = df.apply(_h4, axis=1)

    # ── H7: z-score composite within each experiment ────────────────────────────
    # z(H1) + z(H4) + z(H5) – z(H6)
    #   H1 = elicitation, H4 = h4_tok_svd_global_oriented,
    #   H5 = emb_dist_from_neutral, H6 = emb_rephrase_std_cos
    components = [
        ("elicitation",              +1.0),
        ("h4_tok_svd_global_oriented",+1.0),
        ("emb_dist_from_neutral",    +1.0),
        ("emb_rephrase_std_cos",     -1.0),
    ]
    tmp_cols: list[str] = []
    for col, sign in components:
        tmp = f"_z_{col}"
        tmp_cols.append(tmp)
        df[tmp] = float("nan")
        for exp_key in df["experiment"].unique():
            mask = df["experiment"] == exp_key
            vals = df.loc[mask, col].astype(float)
            mu = float(vals.mean())
            sigma = float(vals.std())
            z = (vals - mu) / sigma if sigma > 1e-12 else pd.Series(0.0, index=vals.index)
            df.loc[mask, tmp] = z * sign

    df["h7_z_composite"] = sum(df[t] for t in tmp_cols)
    df.drop(columns=tmp_cols, inplace=True)

    # ── H9a: cross-trait cosine in raw embedding space ──────────────────────────
    # neg-trait rows → cross-trait is pos-trait → emb_cos_to_pos_trait
    # pos-trait rows → cross-trait is neg-trait → emb_cos_to_neg_trait
    df["h9a_emb_cos_cross"] = np.where(
        df["trait_role"] == "negative",
        df["emb_cos_to_pos_trait"].astype(float),
        df["emb_cos_to_neg_trait"].astype(float),
    )

    # ── H9b: cross-trait projection in data-pointwise SVD space ─────────────────
    # In W_fixed SVD: pos-trait prompts have +PC2, neg-trait prompts have −PC2.
    # "Cross-trait projection" = component toward the OTHER trait's PC2 direction:
    #   neg-trait rows → cross-trait = pos-trait → +PC2 direction → use +sv2
    #   pos-trait rows → cross-trait = neg-trait → −PC2 direction → use −sv2
    df["h9b_svd_cross_proj"] = np.where(
        df["trait_role"] == "negative",
        df["sv2_truncated_fixed"].astype(float),
        -df["sv2_truncated_fixed"].astype(float),
    )

    # ── H10: cross-trait projection in token-wise SVD space (PC specialisation) ─
    # pc2_tokens_oriented is the all-prompts token SVD PC2.
    # Same orientation convention as H9b.
    df["h10_tok_svd_cross_proj"] = np.where(
        df["trait_role"] == "negative",
        df["pc2_tokens_oriented"].astype(float),
        -df["pc2_tokens_oriented"].astype(float),
    )

    return df


def _isnan(v: object) -> bool:
    try:
        return bool(np.isnan(float(v)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return True


# ─── Panel specifications ────────────────────────────────────────────────────────
# Each tuple: (out_filename, heuristic_col, xlabel, title, na_on_fixed)

PANEL_SPECS: list[tuple[str, str, str, str, bool]] = [
    # H1 ─ Elicitation strength
    (
        "cross_panel2x2_H1_elicitation",
        "elicitation",
        "H1 — Elicitation strength (pp)",
        "H1: Elicitation Strength",
        False,
    ),
    # H2 ─ Logprob-diff heuristic (PH)
    (
        "cross_panel2x2_H2_ph",
        "ph",
        "H2 — Logprob-diff heuristic (PH)",
        "H2: Logprob-Diff Heuristic (PH)",
        False,
    ),
    # H3 ─ Data-pointwise SVD PC1, own-trait oriented
    (
        "cross_panel2x2_H3_data_svd_pc1_oriented",
        "h3_data_svd_pc1_oriented",
        "H3 — Data-pointwise logprob SVD PC1\n(own-trait oriented; neutral excluded)",
        "H3: Data-Pointwise Logprob SVD PC1 (own-trait oriented)",
        False,
    ),
    # H4 ─ Token-wise SVD PC1, global, own-trait oriented
    (
        "cross_panel2x2_H4_tok_svd_global_oriented",
        "h4_tok_svd_global_oriented",
        "H4 — Token-wise logprob SVD PC1\n(global SVD, own-trait oriented; neutral excluded)",
        "H4: Token-SVD PC1 Global (own-trait oriented)",
        False,
    ),
    # H5 ─ Embedding distance from neutral centroid
    (
        "cross_panel2x2_H5_emb_dist_neutral",
        "emb_dist_from_neutral",
        "H5 — Embedding L2 distance from neutral centroid",
        "H5: Embedding Distance from Neutral",
        False,
    ),
    # H6 ─ Embedding rephrasing diversity (std cosine)
    (
        "cross_panel2x2_H6_emb_rephrase_std",
        "emb_rephrase_std_cos",
        "H6 — Rephrasing diversity (std cosine sim to original)",
        "H6: Rephrasing Diversity (Emb Std Cosine)",
        True,
    ),
    # H7 ─ Z-score composite: z(H1) + z(H4) + z(H5) – z(H6)
    (
        "cross_panel2x2_H7_z_composite",
        "h7_z_composite",
        "H7 — Z-composite: z(H1)+z(H4)+z(H5)–z(H6)",
        "H7: Z-Score Composite",
        False,
    ),
    # H9a ─ Cross-trait projection, raw embedding space
    (
        "cross_panel2x2_H9a_emb_cos_cross",
        "h9a_emb_cos_cross",
        "H9a — Cross-trait cosine (raw embedding space)",
        "H9a: Cross-Trait Cosine (Raw Embedding)",
        False,
    ),
    # H9b ─ Cross-trait projection, data-pointwise SVD space
    (
        "cross_panel2x2_H9b_svd_cross_proj",
        "h9b_svd_cross_proj",
        "H9b — Cross-trait projection (data-pointwise SVD, PC2)",
        "H9b: Cross-Trait Projection (SVD PC2)",
        False,
    ),
    # H10 ─ Cross-trait projection, token-wise SVD (PC specialisation)
    (
        "cross_panel2x2_H10_tok_svd_cross_proj",
        "h10_tok_svd_cross_proj",
        "H10 — Cross-trait projection (token-wise SVD PC2)",
        "H10: Cross-Trait Projection (Token SVD PC2)",
        False,
    ),
    # ── Group A: step-1 gradient heuristics ─────────────────────────────────
    (
        "cross_panel2x2_HA2_filter",
        "ha2_filter",
        "HA2 — Filtered token diff (\u03c4=0.5)",
        "HA2: Filtered Token Diff",
        False,
    ),
    (
        "cross_panel2x2_HA3_topk",
        "ha3_topk",
        "HA3 — Top-25% token diff",
        "HA3: Top-k Token Diff",
        False,
    ),
    (
        "cross_panel2x2_HA4_grad_mag",
        "ha4_grad_mag",
        "HA4 — Gradient magnitude proxy\nmean(1 \u2212 exp(\u2212|\u0394|))",
        "HA4: Gradient Magnitude",
        False,
    ),
    (
        "cross_panel2x2_HA5_filter_grad",
        "ha5_filter_grad",
        "HA5 — Filter + gradient magnitude",
        "HA5: Filter + Grad Magnitude",
        False,
    ),
    # ── Group B: cumulative gradient heuristics ─────────────────────────────
    (
        "cross_panel2x2_HB1_pc1_var",
        "hb1_pc1_var_frac",
        "HB1 — Signal coherence (PC1%)",
        "HB1: Signal Coherence",
        False,
    ),
    (
        "cross_panel2x2_HB2_sim_decay",
        "hb2_sim_loss_decay",
        "HB2 — Simulated loss decay (K=32)",
        "HB2: Simulated Loss Decay",
        False,
    ),
    (
        "cross_panel2x2_HB3_persistent",
        "hb3_persistent_loss_frac",
        "HB3 — Persistent loss fraction",
        "HB3: Persistent Loss Fraction",
        False,
    ),
    (
        "cross_panel2x2_HB4_str_coh",
        "hb4_strength_x_coherence",
        "HB4 — Strength \u00d7 coherence (PH \u00d7 PC1%)",
        "HB4: Strength \u00d7 Coherence",
        False,
    ),
    (
        "cross_panel2x2_HB5_eff_rank",
        "hb5_effective_rank",
        "HB5 — Effective rank (#PCs for 50% var)",
        "HB5: Effective Rank",
        False,
    ),
    (
        "cross_panel2x2_HB6_sim_resid",
        "hb6_sim_residual",
        "HB6 — Simulated residual loss (K=32)",
        "HB6: Simulated Residual",
        False,
    ),
]


# ─── Main ────────────────────────────────────────────────────────────────────────

def main(use_2x2: bool = False) -> list[Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    layout_name = "2x2" if use_2x2 else "1x1"

    print("=" * 70)
    print(f"plot_panels_cross_trait_suppression.py  [{layout_name} layout]")
    print(f"Timestamp: {timestamp}")
    print("=" * 70)

    df = pd.read_csv(CSV_PATH)
    print(f"\nLoaded CSV: {df.shape}")

    df = add_cross_trait_columns(df)
    print(f"Added cross-suppression columns. DataFrame shape: {df.shape}")

    df = _add_derived_columns(df)
    print(f"Added derived heuristic columns. DataFrame shape: {df.shape}")
    print(
        "  Derived columns: h3_data_svd_pc1_oriented, h4_tok_svd_global_oriented, "
        "h7_z_composite, h9a_emb_cos_cross, h9b_svd_cross_proj, h10_tok_svd_cross_proj"
    )

    out_paths: list[Path] = []
    skipped = 0

    for spec in PANEL_SPECS:
        out_fname, hcol, xlabel, title, na_fixed = spec

        if hcol not in df.columns:
            print(f"  SKIP {out_fname}: column '{hcol}' not in DataFrame")
            skipped += 1
            continue

        try:
            if use_2x2:
                p = plot_2x2_panel(
                    df,
                    heuristic_col=hcol,
                    panel_xlabel=xlabel,
                    panel_title=title,
                    out_filename=out_fname,
                    timestamp=timestamp,
                    na_on_fixed=na_fixed,
                    y_col="cross_suppression",
                    y_lo_col="cross_suppression_ci_lo",
                    y_hi_col="cross_suppression_ci_hi",
                    suptitle_note="vs cross-trait suppression",
                    force_linear=True,
                )
            else:
                p = plot_single_panel(
                    df,
                    heuristic_col=hcol,
                    panel_xlabel=xlabel,
                    panel_title=title,
                    out_filename=out_fname,
                    timestamp=timestamp,
                    na_on_fixed=na_fixed,
                    y_col="cross_suppression",
                    y_lo_col="cross_suppression_ci_lo",
                    y_hi_col="cross_suppression_ci_hi",
                    suptitle_note="vs cross-trait suppression",
                    force_linear=True,
                )
            out_paths.append(p)
        except Exception as exc:
            print(f"  ERROR generating {out_fname}: {exc}")

    print(f"\nDone. {len(out_paths)} plots saved, {skipped} skipped.")
    return out_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot heuristic panels vs cross-trait suppression.")
    parser.add_argument("--2x2", dest="use_2x2", action="store_true", help="Use legacy 2x2 layout")
    parser.set_defaults(use_2x2=False)
    args = parser.parse_args()

    out_paths = main(use_2x2=args.use_2x2)
    print("\nAll plots:")
    for p in out_paths:
        print(f"  {p}")
