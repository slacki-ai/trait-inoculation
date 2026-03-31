#!/usr/bin/env python3
"""Generate all Section 1 figures: Predicting Inoculation Strength.

Run from the repo root:
    python slides/section1.py

Output: slides/figures/section1_*.png
Requires: slides/data/dataset.csv (run build_dataset.py first)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_utils import (
    load_dataset,
    make_heuristic_figure,
    make_embedding_figure,
    make_embedding_figure_3d,
    save_figure,
)

DATA_CSV  = Path(__file__).resolve().parent / "data" / "dataset.csv"
META_JSON = Path(__file__).resolve().parent / "data" / "coords_metadata.json"
FIGURES   = Path(__file__).resolve().parent / "figures"

TS = datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Slide configurations
# ---------------------------------------------------------------------------

# Each entry: (slug, x_col_base, x_label, title, x_col_base_2, x_label_2)
# x_col_base_2/x_label_2 are optional (None = 2×2 layout; set = 2×4 layout with 2nd component)
# x_col_base is resolved per panel:
#   - plain name (e.g. "elicitation") → same column for both fixed and mix
#   - name without suffix (e.g. "pc1") → uses pc1_fixed / pc1_mix per panel
HEURISTIC_SLIDES = [
    (
        "slide1_elicitation",
        "elicitation",
        "Elicitation strength (pp)\n[prompt's effect on base model trait score]",
        "Slide 1 — Elicitation heuristic vs inoculation suppression",
        None, None,
    ),
    (
        "slide2_ph",
        "ph_combined",
        "Mean logprob diff (PH)\n[mean(lp_inoculated − lp_default) on training completions]",
        "Slide 2 — Logprob diff heuristic (PH) vs inoculation suppression",
        None, None,
    ),
    (
        "slide3a_pca_pc1_pc2",
        "pc1",
        "PC1 coordinate\n[centred PCA on token-wise logprob diffs, StandardScaler]",
        "Slide 3a — PCA PC1 & PC2 vs inoculation suppression",
        "pc2",
        "PC2 coordinate\n[centred PCA on token-wise logprob diffs, StandardScaler]",
    ),
    (
        "slide4a_truncsvd_sv1_sv2",
        "sv1_truncated",
        "SV1 coordinate\n[TruncatedSVD (uncentred) on token-wise logprob diffs]",
        "Slide 4a — TruncatedSVD SV1 & SV2 vs inoculation suppression",
        "sv2_truncated",
        "SV2 coordinate\n[TruncatedSVD (uncentred) on token-wise logprob diffs]",
    ),
]

# Embedding scatter slides: (slug, x_col_base, y_col_base, x_label, y_label, title)
EMBEDDING_SLIDES = [
    (
        "slide3b_pca_scatter",
        "pc1", "pc2",
        "PC1", "PC2",
        "Slide 3b — PCA prompt embedding (coloured by negative-trait suppression)",
    ),
    (
        "slide4b_truncsvd_scatter",
        "sv1_truncated", "sv2_truncated",
        "SV1 (TruncatedSVD)", "SV2 (TruncatedSVD)",
        "Slide 4b — TruncatedSVD prompt embedding (coloured by negative-trait suppression)",
    ),
]

# 3D embedding scatter slides: (slug, x_col_base, y_col_base, z_col_base, x_label, y_label, z_label, title)
EMBEDDING_3D_SLIDES = [
    (
        "slide3c_pca_scatter_3d",
        "pc1", "pc2", "pc3",
        "PC1", "PC2", "PC3",
        "Slide 3c — PCA prompt embedding 3D (coloured by negative-trait suppression)",
    ),
    (
        "slide4c_truncsvd_scatter_3d",
        "sv1_truncated", "sv2_truncated", "sv3_truncated",
        "SV1 (TruncatedSVD)", "SV2 (TruncatedSVD)", "SV3 (TruncatedSVD)",
        "Slide 4c — TruncatedSVD prompt embedding 3D (coloured by negative-trait suppression)",
    ),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_CSV}\nRun: python slides/build_dataset.py"
        )

    print(f"Loading dataset from {DATA_CSV}…")
    df = load_dataset(DATA_CSV)
    print(f"  {len(df)} rows, "
          f"{df['prompt_key'].nunique()} unique prompts, "
          f"{df['suppression'].notna().sum()} rows with suppression data")

    coords_meta: dict | None = None
    if META_JSON.exists():
        with open(META_JSON) as f:
            coords_meta = json.load(f)
        print(f"  Loaded coords metadata from {META_JSON.name}")

    # --- Heuristic scatter figures (Slides 1, 2, 3a, 4a, 5a) ---
    for slug, x_col_base, x_label, title, x_col_base_2, x_label_2 in HEURISTIC_SLIDES:
        print(f"\nGenerating {slug}…")
        # Check that at least one panel has data for this X column
        has_x = False
        for prefix_type in ("fixed", "mix"):
            candidate = f"{x_col_base}_{prefix_type}"
            col = candidate if candidate in df.columns else x_col_base
            if col in df.columns and df[col].notna().any():
                has_x = True
                break
        if not has_x:
            print(f"  Skipping: no data for x_col_base='{x_col_base}'")
            continue

        fig = make_heuristic_figure(
            df=df,
            x_col_base=x_col_base,
            x_label=x_label,
            title=title,
            x_col_base_2=x_col_base_2,
            x_label_2=x_label_2,
        )
        save_figure(fig, FIGURES / f"{slug}_{TS}.png")

    # --- Embedding scatter figures (Slides 3b, 4b, 5b) ---
    for slug, x_col_base, y_col_base, x_label, y_label, title in EMBEDDING_SLIDES:
        print(f"\nGenerating {slug}…")
        # Check data availability
        xc = f"{x_col_base}_fixed"
        if xc not in df.columns or df[xc].notna().sum() < 3:
            print(f"  Skipping: no embedding data for '{x_col_base}'")
            continue

        fig = make_embedding_figure(
            df=df,
            x_col_base=x_col_base,
            y_col_base=y_col_base,
            x_label=x_label,
            y_label_str=y_label,
            title=title,
            coords_meta=coords_meta,
        )
        save_figure(fig, FIGURES / f"{slug}_{TS}.png")

    # --- 3D Embedding scatter figures (Slides 3c, 4c) ---
    for slug, x_col_base, y_col_base, z_col_base, x_label, y_label, z_label, title in EMBEDDING_3D_SLIDES:
        print(f"\nGenerating {slug}…")
        xc = f"{x_col_base}_fixed"
        zc = f"{z_col_base}_fixed"
        if xc not in df.columns or df[xc].notna().sum() < 3:
            print(f"  Skipping: no embedding data for '{x_col_base}'")
            continue
        if zc not in df.columns or df[zc].notna().sum() < 3:
            print(f"  Skipping: no 3rd component data for '{z_col_base}'")
            continue

        fig = make_embedding_figure_3d(
            df=df,
            x_col_base=x_col_base,
            y_col_base=y_col_base,
            z_col_base=z_col_base,
            x_label=x_label,
            y_label_str=y_label,
            z_label_str=z_label,
            title=title,
            coords_meta=coords_meta,
        )
        save_figure(fig, FIGURES / f"{slug}_{TS}.png")

    print(f"\nAll Section 1 figures saved to {FIGURES}/")
    # Return list of output paths for callers that want to send to Slack
    return sorted(FIGURES.glob(f"*_{TS}.png"))


if __name__ == "__main__":
    paths = main()
    if paths:
        print("\nGenerated files:")
        for p in paths:
            print(f"  {p}")
