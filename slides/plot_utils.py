"""Shared plotting utilities for the inoculation slides.

Public API
----------
load_dataset(path)           → pd.DataFrame
make_heuristic_figure(...)   → plt.Figure   (2×(2 or 4) heuristic scatter)
make_embedding_figure(...)   → plt.Figure   (2×4 embedding scatter, both trait colours)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

POSITIVE_COLOR = "#2166ac"   # blue  — desired trait (French / German)
NEGATIVE_COLOR = "#d6604d"   # red   — undesired trait (Playful / Flattering)

TRAIT_COLORS = {"positive": POSITIVE_COLOR, "negative": NEGATIVE_COLOR}

EXPERIMENT_LABELS = {
    "playful_french_7b":    "Playful / French — Qwen-2.5-7B",
    "german_flattering_8b": "German / Flattering — Llama-3.1-8B",
}

TRAIT_LABELS = {
    "playful_french_7b":    {"positive": "French",  "negative": "Playful"},
    "german_flattering_8b": {"positive": "German",  "negative": "Flattering"},
}

TRAIT_ROLE_LABELS = {
    "negative": "Undesired trait",
    "positive": "Desired trait",
}

PREFIX_LABELS = {
    "fixed": "Fixed prefix",
    "mix":   "Mix (rephrased)",
}

FIGURE_DPI = 150
FIGURE_FONT_SIZE = 9
plt.rcParams.update({
    "font.size":       FIGURE_FONT_SIZE,
    "axes.titlesize":  FIGURE_FONT_SIZE,
    "axes.labelsize":  FIGURE_FONT_SIZE,
    "xtick.labelsize": FIGURE_FONT_SIZE - 1,
    "ytick.labelsize": FIGURE_FONT_SIZE - 1,
    "legend.fontsize": FIGURE_FONT_SIZE - 2,
    "figure.dpi":      FIGURE_DPI,
})

EXPERIMENTS  = ["playful_french_7b", "german_flattering_8b"]
PREFIX_TYPES = ["fixed", "mix"]

# Maps a resolved column name → (meta_key_in_coords_metadata, component_index)
# Used to append "(var. expl. = X.X%)" to axis labels in embedding scatter plots.
_COL_TO_META: dict[str, tuple[str, int]] = {
    "pc1_fixed":             ("pc_fixed",            0),
    "pc2_fixed":             ("pc_fixed",            1),
    "pc1_mix":               ("pc_mix",              0),
    "pc2_mix":               ("pc_mix",              1),
    "sv1_truncated_fixed":   ("svd_truncated_fixed", 0),
    "sv2_truncated_fixed":   ("svd_truncated_fixed", 1),
    "sv1_truncated_mix":     ("svd_truncated_mix",   0),
    "sv2_truncated_mix":     ("svd_truncated_mix",   1),
    "sv3_truncated_mix":     ("svd_truncated_mix",   2),
    "pc3_fixed":             ("pc_fixed",            2),
    "pc3_mix":               ("pc_mix",              2),
    "sv3_truncated_fixed":   ("svd_truncated_fixed", 2),
}


def _var_label(base_label: str, coords_meta: Optional[dict], experiment: str, col: str) -> str:
    """Append '(var. expl. = X.X%)' to base_label if metadata is available."""
    if not coords_meta:
        return base_label
    meta_key, idx = _COL_TO_META.get(col, (None, None))
    if meta_key is None:
        return base_label
    pct_list = coords_meta.get(experiment, {}).get(meta_key)
    if not pct_list or idx >= len(pct_list):
        return base_label
    pct = pct_list[idx]
    if not isinstance(pct, (int, float)) or not np.isfinite(pct):
        return base_label
    return f"{base_label} ({pct:.1f}%)"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _x_col(x_col_base: str, prefix_type: str, df: pd.DataFrame) -> str:
    """Resolve the X column name for a given prefix_type.

    Tries "x_col_base_fixed" / "x_col_base_mix" first; falls back to x_col_base.
    """
    candidate = f"{x_col_base}_{prefix_type}"
    if candidate in df.columns:
        return candidate
    return x_col_base


def _linear_fit_band(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
) -> Optional[dict]:
    """Fit a linear regression, draw the line + 95% CI band, return stats dict.

    Returns None if fewer than 3 finite paired observations.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    xv, yv = x[mask], y[mask]

    slope, intercept, r, p, _ = stats.linregress(xv, yv)
    r2 = r ** 2
    n, dof = len(xv), len(xv) - 2

    y_pred = slope * xv + intercept
    se_res = np.sqrt(np.sum((yv - y_pred) ** 2) / max(dof, 1))
    x_mean = xv.mean()
    ss_x = np.sum((xv - x_mean) ** 2) or 1e-12
    t_crit = stats.t.ppf(0.975, df=max(dof, 1))

    x_line = np.linspace(xv.min(), xv.max(), 200)
    y_line = slope * x_line + intercept
    se_band = se_res * np.sqrt(1 / n + (x_line - x_mean) ** 2 / ss_x)

    ax.fill_between(x_line, y_line - t_crit * se_band, y_line + t_crit * se_band,
                    alpha=0.15, color=color, linewidth=0)
    ax.plot(x_line, y_line, color=color, linewidth=1.5, zorder=3)

    return {"r": r, "p": p, "r2": r2, "n": n}


def _panel_scatter(
    ax: plt.Axes,
    df_panel: pd.DataFrame,
    x_col_base: str,
    prefix_type: str,
    experiment: str,
    show_y_label: bool = True,
    x_label: str = "",
    filter_by_family: bool = False,
) -> None:
    """Draw one scatter panel (one experiment × one prefix_type).

    Both trait roles (positive/negative) on the same axes, each with its own
    colour, error bars, and linear fit.

    If filter_by_family=True, each role only shows prompts whose prompt_family
    matches that role's trait name (lowercased) or is "neutral" — excluding
    prompts designed for the other trait.
    """
    xcol = _x_col(x_col_base, prefix_type, df_panel)
    legend_handles: list = []

    for role in ("positive", "negative"):
        color = TRAIT_COLORS[role]
        trait_label = TRAIT_LABELS.get(experiment, {}).get(role, role.capitalize())

        mask = (
            (df_panel.trait_role == role) &
            df_panel[xcol].notna() &
            df_panel["suppression"].notna()
        )
        if filter_by_family:
            trait_family = trait_label.lower()
            mask &= df_panel["prompt_family"].isin([trait_family, "neutral"])
        sub = df_panel[mask].copy()

        if sub.empty:
            continue

        x = sub[xcol].to_numpy(dtype=float)
        y = sub["suppression"].to_numpy(dtype=float)
        yerr_lo = np.clip((sub["suppression"] - sub["suppression_ci_lo"]).to_numpy(dtype=float), 0, None)
        yerr_hi = np.clip((sub["suppression_ci_hi"] - sub["suppression"]).to_numpy(dtype=float), 0, None)

        ax.errorbar(
            x, y, yerr=[yerr_lo, yerr_hi],
            fmt="o", color=color, markersize=4, capsize=2, capthick=0.7,
            linewidth=0.7, alpha=0.75, zorder=4,
        )

        fit = _linear_fit_band(ax, x, y, color)
        if fit:
            p_str = f"p={fit['p']:.3f}" if fit["p"] >= 0.001 else "p<0.001"
            lbl = f"{trait_label}  r={fit['r']:+.2f}  R²={fit['r2']:.2f}  {p_str}  (n={fit['n']})"
        else:
            lbl = f"{trait_label}  (n={len(sub)}, insufficient data)"
        legend_handles.append(Line2D([0], [0], color=color, linewidth=2, label=lbl))

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.35)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="best", framealpha=0.9, handlelength=1.2)

    if show_y_label:
        ax.set_ylabel("Suppression (pp)\n[no-inoc − inoc]", fontsize=FIGURE_FONT_SIZE - 1)
    if x_label:
        ax.set_xlabel(x_label, fontsize=FIGURE_FONT_SIZE - 1)


# ---------------------------------------------------------------------------
# Public: heuristic scatter figure
# ---------------------------------------------------------------------------

def make_heuristic_figure(
    df: pd.DataFrame,
    x_col_base: str,
    x_label: str,
    title: str,
    x_col_base_2: Optional[str] = None,   # if set, adds 2 cols showing 2nd component
    x_label_2: Optional[str] = None,
    x_col_bases_extra: Optional[list[tuple[str, str]]] = None,  # additional (col_base, label) pairs
    filter_by_family: bool = False,        # if True, each role shows only trait-matched prompts
) -> plt.Figure:
    """Create a 2×N heuristic scatter figure.

    Rows = experiments (playful_french_7b, german_flattering_8b)
    Cols = prefix_types (fixed, mix) for each x_col_base provided.
      If x_col_base_2 is supplied the figure is 2×4.
      x_col_bases_extra appends further (col_base, label) pairs beyond x_col_base_2.
    """
    x_configs: list[tuple[str, str]] = [(x_col_base, x_label)]
    if x_col_base_2 is not None:
        x_configs.append((x_col_base_2, x_label_2 or x_col_base_2))
    if x_col_bases_extra:
        x_configs.extend(x_col_bases_extra)

    n_cols = len(x_configs) * 2   # 2 prefix_types per x_col
    figsize = (6.5 * n_cols, 9)

    fig, axes = plt.subplots(
        2, n_cols,
        figsize=figsize,
        sharex=False, sharey=False,
        constrained_layout=True,
    )
    # Ensure axes is always 2-D
    if n_cols == 1:
        axes = axes[:, np.newaxis]
    fig.suptitle(title, fontsize=FIGURE_FONT_SIZE + 3, fontweight="bold")

    for row_idx, experiment in enumerate(EXPERIMENTS):
        row_label = EXPERIMENT_LABELS.get(experiment, experiment)

        for x_idx, (xcb, xlabel) in enumerate(x_configs):
            for pt_idx, prefix_type in enumerate(PREFIX_TYPES):
                col_idx = x_idx * 2 + pt_idx
                ax = axes[row_idx, col_idx]

                df_panel = df[
                    (df.experiment == experiment) &
                    (df.prefix_type == prefix_type)
                ].copy()

                if df_panel.empty:
                    ax.set_visible(False)
                    continue

                _panel_scatter(
                    ax, df_panel, xcb, prefix_type, experiment,
                    show_y_label=(col_idx == 0),
                    x_label=xlabel if row_idx == 1 else "",
                    filter_by_family=filter_by_family,
                )

                # Column titles (top row only)
                if row_idx == 0:
                    short_xl = xlabel.split("\n")[0]
                    ax.set_title(
                        f"{PREFIX_LABELS[prefix_type]}  ·  {short_xl}",
                        fontweight="bold", fontsize=FIGURE_FONT_SIZE,
                    )

                # Row labels (left column of each x group only)
                if col_idx == 0:
                    ax.set_ylabel(
                        f"{row_label}\n\nSuppression (pp)\n[no-inoc − inoc]",
                        fontsize=FIGURE_FONT_SIZE - 1,
                    )

    return fig


# ---------------------------------------------------------------------------
# Public: embedding scatter figure
# ---------------------------------------------------------------------------

def _embedding_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    experiment: str,
    prefix_type: str,
    trait_role: str,
    xcol: str,
    ycol: str,
    cmap,
    vmin: float,
    vmax: float,
    x_label: str = "",
    y_label: str = "",
    filter_by_family: bool = False,
) -> Optional[object]:
    """Draw one embedding panel; return the scatter artist (for colorbar) or None."""
    df_emb = df[
        (df.experiment == experiment) &
        (df.prefix_type == prefix_type) &
        (df.trait_role == trait_role) &
        df[xcol].notna() &
        df[ycol].notna()
    ].drop_duplicates("prompt_key").copy()

    if filter_by_family:
        trait_label = TRAIT_LABELS.get(experiment, {}).get(trait_role, "")
        trait_family = trait_label.lower()
        df_emb = df_emb[df_emb["prompt_family"].isin([trait_family, "neutral"])]

    if df_emb.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="grey", fontsize=8)
        return None

    has_supp = df_emb["suppression"].notna()

    # Grey crosses: prompts without training runs
    df_miss = df_emb[~has_supp]
    if not df_miss.empty:
        ax.scatter(df_miss[xcol], df_miss[ycol],
                   marker="x", s=30, color="lightgrey", linewidths=0.7, zorder=2)

    sc = None
    df_has = df_emb[has_supp]
    if not df_has.empty:
        sc = ax.scatter(
            df_has[xcol], df_has[ycol],
            c=df_has["suppression"].to_numpy(dtype=float),
            cmap=cmap, vmin=vmin, vmax=vmax,
            s=55, edgecolors="white", linewidths=0.3, zorder=3,
        )
        for _, row in df_has.iterrows():
            ax.annotate(
                row["prompt_key"],
                (float(row[xcol]), float(row[ycol])),
                fontsize=4.5, alpha=0.65, ha="left", va="bottom",
                xytext=(2, 2), textcoords="offset points",
            )

    ax.axhline(0, color="black", linewidth=0.3, alpha=0.25)
    ax.axvline(0, color="black", linewidth=0.3, alpha=0.25)
    if x_label:
        ax.set_xlabel(x_label, fontsize=FIGURE_FONT_SIZE - 1)
    if y_label:
        ax.set_ylabel(y_label, fontsize=FIGURE_FONT_SIZE - 1)
    return sc


def make_embedding_figure(
    df: pd.DataFrame,
    x_col_base: str,   # e.g. "pc1" → pc1_fixed / pc1_mix per panel
    y_col_base: str,   # e.g. "pc2"
    x_label: str,
    y_label_str: str,
    title: str,
    coords_meta: Optional[dict] = None,  # from coords_metadata.json; adds "(X.X%)" to labels
    filter_by_family: bool = False,      # if True, each panel shows only trait-matched prompts
) -> plt.Figure:
    """Create a 2×4 embedding scatter coloured by suppression.

    Rows    = experiments (playful_french_7b, german_flattering_8b)
    Columns = (fixed/undesired, mix/undesired, fixed/desired, mix/desired)

    Colour encodes suppression strength (shared scale across all panels).
    Grey × = prompts with no training run.
    """
    # Layout: 2 rows × 4 cols
    # Cols 0-1: undesired (negative) trait suppression
    # Cols 2-3: desired (positive) trait suppression
    COL_CONFIGS = [
        ("fixed", "negative"),
        ("mix",   "negative"),
        ("fixed", "positive"),
        ("mix",   "positive"),
    ]

    figsize = (6.0 * 4, 5.5 * 2)
    fig, axes = plt.subplots(2, 4, figsize=figsize, constrained_layout=True)
    fig.suptitle(title, fontsize=FIGURE_FONT_SIZE + 3, fontweight="bold")

    cmap = plt.cm.RdYlGn

    # Compute shared colour scale from all suppression values
    all_supp = df["suppression"].dropna()
    vmin = float(all_supp.quantile(0.02)) if len(all_supp) else -10
    vmax = float(all_supp.quantile(0.98)) if len(all_supp) else 80

    sc_ref = None  # for colorbar

    for row_idx, experiment in enumerate(EXPERIMENTS):
        row_label = EXPERIMENT_LABELS.get(experiment, experiment)

        for col_idx, (prefix_type, trait_role) in enumerate(COL_CONFIGS):
            ax = axes[row_idx, col_idx]
            xcol = _x_col(x_col_base, prefix_type, df)
            ycol = _x_col(y_col_base, prefix_type, df)

            # Build per-panel axis labels (include variance % when metadata available)
            xl = _var_label(x_label, coords_meta, experiment, xcol)
            yl_base = f"{row_label}\n\n{y_label_str}" if col_idx == 0 else y_label_str
            yl = _var_label(yl_base, coords_meta, experiment, ycol)

            sc = _embedding_panel(
                ax, df, experiment, prefix_type, trait_role,
                xcol, ycol, cmap, vmin, vmax,
                x_label=xl,
                y_label=yl if col_idx == 0 else "",
                filter_by_family=filter_by_family,
            )
            if sc is not None:
                sc_ref = sc

            # Column titles (top row)
            if row_idx == 0:
                trait_lbl = TRAIT_LABELS.get(experiment, {}).get(trait_role, trait_role)
                role_short = "undesired" if trait_role == "negative" else "desired"
                ax.set_title(
                    f"{PREFIX_LABELS[prefix_type]}  ·  {role_short} ({trait_lbl})",
                    fontweight="bold", fontsize=FIGURE_FONT_SIZE,
                )

    # Shared colorbar
    if sc_ref is not None:
        cbar = fig.colorbar(
            sc_ref, ax=axes,
            orientation="vertical",
            fraction=0.012, pad=0.015,
            label="Suppression (pp)  [no-inoc − inoc]",
        )
        cbar.ax.tick_params(labelsize=FIGURE_FONT_SIZE - 2)

    return fig


# ---------------------------------------------------------------------------
# Public: 3D embedding scatter figure
# ---------------------------------------------------------------------------

def make_embedding_figure_3d(
    df: pd.DataFrame,
    x_col_base: str,   # e.g. "pc1" → pc1_fixed / pc1_mix per panel
    y_col_base: str,   # e.g. "pc2"
    z_col_base: str,   # e.g. "pc3"
    x_label: str,
    y_label_str: str,
    z_label_str: str,
    title: str,
    coords_meta: Optional[dict] = None,
) -> plt.Figure:
    """Create a 2×4 3D embedding scatter coloured by suppression.

    Rows    = experiments (playful_french_7b, german_flattering_8b)
    Columns = (fixed/undesired, mix/undesired, fixed/desired, mix/desired)

    Colour encodes suppression strength (shared scale across all panels).
    Grey × = prompts with no training run.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    COL_CONFIGS = [
        ("fixed", "negative"),
        ("mix",   "negative"),
        ("fixed", "positive"),
        ("mix",   "positive"),
    ]

    cmap = plt.cm.RdYlGn
    all_supp = df["suppression"].dropna()
    vmin = float(all_supp.quantile(0.02)) if len(all_supp) else -10
    vmax = float(all_supp.quantile(0.98)) if len(all_supp) else 80
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(6.5 * 4, 6.5 * 2))
    fig.suptitle(title, fontsize=FIGURE_FONT_SIZE + 3, fontweight="bold")

    axes: list[list] = []
    for row_idx in range(2):
        row_axes = []
        for col_idx in range(4):
            ax = fig.add_subplot(2, 4, row_idx * 4 + col_idx + 1, projection="3d")
            row_axes.append(ax)
        axes.append(row_axes)

    for row_idx, experiment in enumerate(EXPERIMENTS):
        row_label = EXPERIMENT_LABELS.get(experiment, experiment)

        for col_idx, (prefix_type, trait_role) in enumerate(COL_CONFIGS):
            ax = axes[row_idx][col_idx]
            xcol = _x_col(x_col_base, prefix_type, df)
            ycol = _x_col(y_col_base, prefix_type, df)
            zcol = _x_col(z_col_base, prefix_type, df)

            xl = _var_label(x_label, coords_meta, experiment, xcol)
            yl = _var_label(y_label_str, coords_meta, experiment, ycol)
            zl = _var_label(z_label_str, coords_meta, experiment, zcol)

            df_emb = df[
                (df.experiment == experiment) &
                (df.prefix_type == prefix_type) &
                (df.trait_role == trait_role) &
                df[xcol].notna() & df[ycol].notna() & df[zcol].notna()
            ].drop_duplicates("prompt_key").copy()

            if df_emb.empty:
                ax.text2D(0.5, 0.5, "no data", ha="center", va="center",
                          transform=ax.transAxes, color="grey", fontsize=8)
            else:
                has_supp = df_emb["suppression"].notna()

                # Grey crosses for prompts without training runs
                df_miss = df_emb[~has_supp]
                if not df_miss.empty:
                    ax.scatter(
                        df_miss[xcol].to_numpy(float),
                        df_miss[ycol].to_numpy(float),
                        df_miss[zcol].to_numpy(float),
                        marker="x", s=30, color="lightgrey", linewidths=0.7, zorder=2,
                    )

                # Coloured dots for prompts with suppression data
                df_has = df_emb[has_supp]
                if not df_has.empty:
                    c_vals = df_has["suppression"].to_numpy(float)
                    ax.scatter(
                        df_has[xcol].to_numpy(float),
                        df_has[ycol].to_numpy(float),
                        df_has[zcol].to_numpy(float),
                        c=c_vals, cmap=cmap, norm=norm,
                        s=60, edgecolors="white", linewidths=0.3, zorder=3,
                    )
                    for _, row in df_has.iterrows():
                        ax.text(
                            float(row[xcol]), float(row[ycol]), float(row[zcol]),
                            row["prompt_key"],
                            fontsize=4.0, alpha=0.60,
                        )

            ax.set_xlabel(xl, fontsize=FIGURE_FONT_SIZE - 2, labelpad=2)
            ax.set_ylabel(yl, fontsize=FIGURE_FONT_SIZE - 2, labelpad=2)
            ax.set_zlabel(zl, fontsize=FIGURE_FONT_SIZE - 2, labelpad=2)
            ax.tick_params(labelsize=FIGURE_FONT_SIZE - 3)

            # Column titles (top row only)
            if row_idx == 0:
                trait_lbl = TRAIT_LABELS.get(experiment, {}).get(trait_role, trait_role)
                role_short = "undesired" if trait_role == "negative" else "desired"
                ax.set_title(
                    f"{PREFIX_LABELS[prefix_type]}  ·  {role_short} ({trait_lbl})",
                    fontweight="bold", fontsize=FIGURE_FONT_SIZE, pad=4,
                )

            # Row label in top-left panel of each row
            if col_idx == 0:
                ax.text2D(
                    -0.12, 0.5, row_label,
                    transform=ax.transAxes,
                    fontsize=FIGURE_FONT_SIZE - 1, rotation=90,
                    ha="center", va="center",
                )

    # Shared colorbar using a ScalarMappable (avoids needing a live scatter artist)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=[axes[r][c] for r in range(2) for c in range(4)],
        orientation="vertical", fraction=0.012, pad=0.02,
        label="Suppression (pp)  [no-inoc − inoc]",
    )
    cbar.ax.tick_params(labelsize=FIGURE_FONT_SIZE - 2)

    fig.subplots_adjust(wspace=0.35, hspace=0.25)
    return fig


# ---------------------------------------------------------------------------
# Convenience: save figure
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, path: Path, close: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    if close:
        plt.close(fig)
    print(f"  Saved → {path}")
