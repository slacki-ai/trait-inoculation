#!/usr/bin/env python3
"""
plot_angle_analysis.py — Pairwise cosine angle heatmaps and cross-trait angle summaries.

Three representations of each prompt's logprob-difference vector are compared:

  1. PCA top-2      : project W onto PC1+PC2 (sklearn PCA, mean-centred)
  2. TruncSVD top-2 : project W onto SV1+SV2 (sklearn TruncatedSVD, no centering)
  3. Raw W          : use the full N×K logprob-difference matrix directly

For each representation (three times), produces:
  A) Pairwise angle heatmap (N×N, degrees, sorted by trait group)
  B) Cross-trait angle summary bar chart (within-neg, within-pos, cross-trait)
  C) Per-prompt mean-angle scatter
        x = mean angle to all negative-group prompts
        y = mean angle to all positive-group prompts

All three methods are shown side-by-side in a single combined figure for each
plot type, making comparisons easy.

Output (per experiment):
  plots/{name}/pca/angle_analysis/angle_heatmap_<ts>.png
  plots/{name}/pca/angle_analysis/angle_cross_trait_<ts>.png
  plots/{name}/pca/angle_analysis/angle_per_prompt_<ts>.png
  results/angle_analysis_{exp_slug}_{ts}.json

Usage:
    # Default Playful/French 7B config:
    python plot_angle_analysis.py

    # Custom config:
    python plot_angle_analysis.py --experiment-config experiment_configs/german_flattering_8b.yaml
"""

import json
import os
import sys
import argparse
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA, TruncatedSVD

# ── Repo root on path ─────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "../../.."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from experiment_config import ExperimentConfig

# =============================================================================
# CLI
# =============================================================================
_ap = argparse.ArgumentParser(
    description="Pairwise cosine angle analysis of inoculation prompt vectors"
)
_ap.add_argument(
    "--experiment-config", default=None, metavar="PATH",
    help="Path to ExperimentConfig YAML.  Omit for default Playful/French 7B.",
)
_args = _ap.parse_args()

if _args.experiment_config:
    cfg = ExperimentConfig.from_yaml(_args.experiment_config)
    print(f"Experiment config: {_args.experiment_config}")
else:
    cfg = ExperimentConfig.default()
    print("Using default Playful/French 7B config.")

NEG  = cfg.negative_trait   # e.g. "Playful"
POS  = cfg.positive_trait   # e.g. "French"
SLUG = cfg.study_model_slug

print(f"  negative_trait: {NEG}")
print(f"  positive_trait: {POS}")
print(f"  study_model:    {SLUG}")

# =============================================================================
# Load data & build W_fixed matrix
# =============================================================================
with open(cfg.perp_json) as f:
    perp_data = json.load(f)

perp_prompts     = perp_data["prompts"]
lp_train_default = np.array(perp_data["baseline"]["lp_train_default"])

keys_in_data = [k for k in cfg.all_prompt_keys if k in perp_prompts]
N = len(keys_in_data)
K = len(lp_train_default)
print(f"Prompts available: {N} / {len(cfg.all_prompt_keys)}  |  Training examples: {K}")

assert N >= 4, f"Need at least 4 prompts for PCA, got {N}"

W_fixed = np.zeros((N, K), dtype=np.float32)
for i, key in enumerate(keys_in_data):
    W_fixed[i] = np.array(perp_prompts[key]["lp_train_inoc"]) - lp_train_default

W_fixed = np.where(np.isfinite(W_fixed), W_fixed, 0.0)

# =============================================================================
# Group classification & sort order (by PCA PC1 score, ascending)
# =============================================================================
SOURCE_BY_KEY = {k: cfg.source_for_key(k) for k in keys_in_data}
pos_groups     = set(cfg.resolved_positive_groups)
neg_groups     = set(cfg.resolved_negative_groups)
neutral_groups = set(cfg.resolved_neutral_groups)


def group_type(key: str) -> str:
    src = SOURCE_BY_KEY.get(key, "unknown")
    if src in pos_groups:     return "positive"
    if src in neg_groups:     return "negative"
    if src in neutral_groups: return "neutral"
    return "unknown"


raw_types = [group_type(k) for k in keys_in_data]

# Sort all prompts by their PCA PC1 score (ascending) so the heatmap axes run
# from most-negative to most-positive along the dominant variance direction.
# All three heatmaps share this ordering, making them directly comparable.
_pca_sort = PCA(n_components=1, random_state=42)
_pc1_scores = _pca_sort.fit_transform(W_fixed)[:, 0]   # (N,)

sorted_indices = sorted(range(N), key=lambda i: float(_pc1_scores[i]))
sorted_keys  = [keys_in_data[i] for i in sorted_indices]
sorted_types = [raw_types[i]    for i in sorted_indices]
W_sorted     = W_fixed[sorted_indices]

print(f"Group counts — negative: {sorted_types.count('negative')}, "
      f"positive: {sorted_types.count('positive')}, "
      f"neutral:  {sorted_types.count('neutral')}")
print(f"PC1 range: [{_pc1_scores.min():.3f}, {_pc1_scores.max():.3f}]  "
      f"(heatmap sorted ascending along this axis)")

# =============================================================================
# Three representations
# =============================================================================

def make_representations(W: np.ndarray) -> list[dict]:
    """Build 3 row-vector matrices from W (already sorted).

    Returns list of dicts with keys: name, tag, label, vecs (N×D array).
    """
    n_comp = min(2, W.shape[0] - 1, W.shape[1])

    # 1. PCA top-2 (mean-centred)
    pca = PCA(n_components=n_comp, random_state=42)
    W_pca = pca.fit_transform(W)
    var_pca = pca.explained_variance_ratio_

    # 2. TruncatedSVD top-2 (no centering)
    svd = TruncatedSVD(n_components=n_comp, n_iter=10, random_state=42)
    W_svd = svd.fit_transform(W)
    # Explained variance ratio for SVD: fraction of total Frobenius norm squared
    total_ss = float(np.sum(W ** 2))
    sv_ss = svd.singular_values_ ** 2
    var_svd = sv_ss / total_ss if total_ss > 0 else sv_ss * 0

    # 3. Raw W (full N×K)
    return [
        dict(
            name="PCA top-2",
            tag="pca2",
            label=(f"PCA  (PC1 {var_pca[0]:.1%} + PC2 {var_pca[1]:.1%}"
                   f" = {sum(var_pca):.1%} var, mean-centred)"),
            vecs=W_pca.astype(np.float32),
        ),
        dict(
            name="TruncSVD top-2",
            tag="svd2",
            label=(f"TruncatedSVD  (SV1 {var_svd[0]:.1%} + SV2 {var_svd[1]:.1%}"
                   f" = {sum(var_svd):.1%} energy, no centering)"),
            vecs=W_svd.astype(np.float32),
        ),
        dict(
            name=f"Raw W  ({N}×{K})",
            tag="raw",
            label=f"Raw W  ({N} prompts × {K} training examples, no projection)",
            vecs=W.astype(np.float32),
        ),
    ]


# =============================================================================
# Angle computation
# =============================================================================

def pairwise_angles_deg(vecs: np.ndarray) -> np.ndarray:
    """Compute N×N matrix of pairwise cosine angles in degrees.

    Rows/cols with near-zero norm → NaN (angle undefined).
    """
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)   # (N, 1)
    zero  = (norms.squeeze() < 1e-10)
    safe  = np.where(norms < 1e-10, 1.0, norms)
    unit  = vecs / safe                                     # (N, D)
    cos   = np.clip(unit @ unit.T, -1.0, 1.0)              # (N, N)
    angles = np.degrees(np.arccos(cos))
    angles[zero, :] = float("nan")
    angles[:, zero] = float("nan")
    return angles


# =============================================================================
# Cross-trait statistics
# =============================================================================

def cross_trait_stats(angles: np.ndarray, types: list[str]) -> dict:
    """Compute mean/std/n for each pair category."""
    neg_idx = [i for i, t in enumerate(types) if t == "negative"]
    pos_idx = [i for i, t in enumerate(types) if t == "positive"]
    neu_idx = [i for i, t in enumerate(types) if t == "neutral"]

    def _pairs(idx_a: list, idx_b: list, symmetric_diagonal_excluded: bool = True):
        vals = []
        for i in idx_a:
            for j in idx_b:
                if symmetric_diagonal_excluded and i == j:
                    continue
                v = angles[i, j]
                if not np.isnan(v):
                    vals.append(float(v))
        return vals

    def _stats(vals: list) -> dict:
        if not vals:
            return dict(mean=float("nan"), std=float("nan"), n=0, vals=[])
        return dict(
            mean=float(np.mean(vals)),
            std=float(np.std(vals)),
            n=len(vals),
            vals=vals,
        )

    return dict(
        within_negative   = _stats(_pairs(neg_idx, neg_idx)),
        within_positive   = _stats(_pairs(pos_idx, pos_idx)),
        cross_trait       = _stats(_pairs(neg_idx, pos_idx, symmetric_diagonal_excluded=False)),
        neg_to_neutral    = _stats(_pairs(neg_idx, neu_idx, symmetric_diagonal_excluded=False)),
        pos_to_neutral    = _stats(_pairs(pos_idx, neu_idx, symmetric_diagonal_excluded=False)),
    )


# =============================================================================
# Per-prompt mean angles
# =============================================================================

def per_prompt_mean_angles(angles: np.ndarray, types: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean_to_neg, mean_to_pos) arrays of length N."""
    neg_idx = [i for i, t in enumerate(types) if t == "negative"]
    pos_idx = [i for i, t in enumerate(types) if t == "positive"]

    mean_to_neg = np.full(len(types), float("nan"))
    mean_to_pos = np.full(len(types), float("nan"))

    for i in range(len(types)):
        others_neg = [angles[i, j] for j in neg_idx if i != j and not np.isnan(angles[i, j])]
        others_pos = [angles[i, j] for j in pos_idx if i != j and not np.isnan(angles[i, j])]
        if others_neg:
            mean_to_neg[i] = np.mean(others_neg)
        if others_pos:
            mean_to_pos[i] = np.mean(others_pos)

    return mean_to_neg, mean_to_pos


# =============================================================================
# Run all three methods
# =============================================================================
methods = make_representations(W_sorted)
for m in methods:
    print(f"\n── {m['name']} ──")
    print(f"   vecs shape: {m['vecs'].shape}")
    m["angles"]      = pairwise_angles_deg(m["vecs"])
    m["stats"]       = cross_trait_stats(m["angles"], sorted_types)
    m["mean_to_neg"], m["mean_to_pos"] = per_prompt_mean_angles(m["angles"], sorted_types)

    s = m["stats"]
    print(f"   Within-{NEG:10s}: {s['within_negative']['mean']:5.1f}° ± {s['within_negative']['std']:.1f}°  (n={s['within_negative']['n']})")
    print(f"   Within-{POS:10s}: {s['within_positive']['mean']:5.1f}° ± {s['within_positive']['std']:.1f}°  (n={s['within_positive']['n']})")
    print(f"   Cross-trait      : {s['cross_trait']['mean']:5.1f}° ± {s['cross_trait']['std']:.1f}°  (n={s['cross_trait']['n']})")


# =============================================================================
# Visual constants
# =============================================================================
GROUP_COLORS = {
    "negative": "#e15759",   # red
    "positive": "#4e79a7",   # blue
    "neutral":  "#76b7b2",   # teal
    "unknown":  "#aaaaaa",
}
METHOD_COLORS = ["#e15759", "#4e79a7", "#59a14f"]  # red, blue, green


def short_label(key: str) -> str:
    return key.replace("_neg", "⁻").replace("_mix", "~")


# =============================================================================
# Figure 1 — Pairwise heatmaps (1 row × 3 methods)
# =============================================================================

def draw_heatmap(ax: plt.Axes, angles: np.ndarray, keys: list[str],
                 types: list[str], title: str) -> None:
    im = ax.imshow(angles, cmap="RdYlBu_r", vmin=0, vmax=180,
                   aspect="auto", interpolation="nearest")
    labels = [short_label(k) for k in keys]
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5.5)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=5)

    for tick, t in zip(ax.get_xticklabels(), types):
        tick.set_color(GROUP_COLORS.get(t, "#222222"))
    for tick, t in zip(ax.get_yticklabels(), types):
        tick.set_color(GROUP_COLORS.get(t, "#222222"))

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Angle (°)", fontsize=8)
    cbar.set_ticks([0, 45, 90, 135, 180])
    cbar.ax.tick_params(labelsize=7)

    ax.set_xlabel("Prompt", fontsize=8)
    ax.set_ylabel("Prompt", fontsize=8)


# Build figure
fig1, axes1 = plt.subplots(1, 3, figsize=(36, 14))
fig1.suptitle(
    f"Pairwise cosine angles — {NEG} / {POS} — {SLUG}\n"
    f"Axes sorted by PCA PC1 score (ascending) — label colour: {NEG}=red · {POS}=blue · Neutral=teal\n"
    f"Red=aligned (0°)  ·  White/Yellow=orthogonal (90°)  ·  Blue=opposite (180°)",
    fontsize=11, fontweight="bold", y=1.01,
)

for ax, m in zip(axes1, methods):
    draw_heatmap(ax, m["angles"], sorted_keys, sorted_types,
                 f"{m['name']}\n{m['label']}")

legend_handles = [
    mpatches.Patch(color=GROUP_COLORS["negative"], label=f"{NEG} prompts"),
    mpatches.Patch(color=GROUP_COLORS["positive"], label=f"{POS} prompts"),
    mpatches.Patch(color=GROUP_COLORS["neutral"],  label="Neutral prompts"),
]
fig1.legend(handles=legend_handles, loc="lower center", ncol=3,
            fontsize=10, bbox_to_anchor=(0.5, -0.02))
fig1.tight_layout()


# =============================================================================
# Figure 2 — Cross-trait angle summary bar chart
# =============================================================================

def draw_cross_trait_bar(ax: plt.Axes, methods_list: list[dict]) -> None:
    categories = ["within_negative", "within_positive", "cross_trait",
                  "neg_to_neutral",  "pos_to_neutral"]
    cat_labels  = [
        f"Within\n{NEG}", f"Within\n{POS}",
        f"Cross\n({NEG}×{POS})",
        f"{NEG} ×\nNeutral", f"{POS} ×\nNeutral",
    ]

    n_methods = len(methods_list)
    x = np.arange(len(categories))
    width = 0.22
    offsets = np.linspace(-(n_methods - 1) * width / 2,
                           (n_methods - 1) * width / 2, n_methods)

    for mi, (m, offset, mc) in enumerate(zip(methods_list, offsets, METHOD_COLORS)):
        means = [m["stats"][c]["mean"] for c in categories]
        stds  = [m["stats"][c]["std"]  for c in categories]
        ax.bar(x + offset, means, width, yerr=stds, color=mc, alpha=0.82,
               label=m["name"], capsize=4,
               error_kw=dict(elinewidth=1.4, capthick=1.4))

    ax.axhline(90, color="gray", linestyle="--", linewidth=1.4, alpha=0.8,
               label="90° (orthogonal)")
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=10)
    ax.set_ylabel("Angle (°)", fontsize=11)
    ax.set_ylim(0, 200)
    ax.set_yticks([0, 30, 60, 90, 120, 150, 180])
    ax.yaxis.grid(alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title(
        f"Cross-trait angle summary: mean ± std\n"
        f"Hypothesis: cross-trait angle ≈ 90° implies no logprob-space overlap",
        fontsize=10, fontweight="bold",
    )


fig2, ax2 = plt.subplots(figsize=(12, 6))
draw_cross_trait_bar(ax2, methods)
fig2.suptitle(
    f"Pairwise angle statistics — {NEG} / {POS} — {SLUG}",
    fontsize=12, fontweight="bold",
)
fig2.tight_layout()


# =============================================================================
# Figure 3 — Per-prompt mean-angle scatter
# =============================================================================

def draw_per_prompt_scatter(ax: plt.Axes, m: dict) -> None:
    mean_to_neg = m["mean_to_neg"]
    mean_to_pos = m["mean_to_pos"]

    for i, (key, t) in enumerate(zip(sorted_keys, sorted_types)):
        x = mean_to_neg[i]
        y = mean_to_pos[i]
        if np.isnan(x) or np.isnan(y):
            continue
        ax.scatter(x, y, color=GROUP_COLORS.get(t, "#aaaaaa"),
                   s=70, edgecolors="black", linewidths=0.5, zorder=3)
        ax.annotate(short_label(key), (x, y),
                    xytext=(3, 3), textcoords="offset points",
                    fontsize=5, color="#222222", zorder=4)

    ax.axhline(90, color="#999999", linestyle="--", linewidth=0.9)
    ax.axvline(90, color="#999999", linestyle="--", linewidth=0.9)
    ax.set_xlabel(f"Mean angle to all {NEG} prompts (°)", fontsize=9)
    ax.set_ylabel(f"Mean angle to all {POS} prompts (°)", fontsize=9)
    ax.set_title(f"{m['name']}\n{m['label']}", fontsize=8.5, fontweight="bold", pad=5)
    ax.set_xlim(0, 185)
    ax.set_ylim(0, 185)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.grid(alpha=0.25)


fig3, axes3 = plt.subplots(1, 3, figsize=(24, 8))
fig3.suptitle(
    f"Per-prompt mean angles — {NEG} / {POS} — {SLUG}\n"
    f"x = mean angle to {NEG} prompts   ·   y = mean angle to {POS} prompts\n"
    f"Negative trait prompts close to {NEG} axis → small x, large y  (and vice versa for positive)",
    fontsize=10, fontweight="bold", y=1.02,
)

for ax, m in zip(axes3, methods):
    draw_per_prompt_scatter(ax, m)

legend_handles2 = [
    mpatches.Patch(color=GROUP_COLORS["negative"], label=f"{NEG}"),
    mpatches.Patch(color=GROUP_COLORS["positive"], label=f"{POS}"),
    mpatches.Patch(color=GROUP_COLORS["neutral"],  label="Neutral"),
]
fig3.legend(handles=legend_handles2, loc="lower center", ncol=3,
            fontsize=10, bbox_to_anchor=(0.5, -0.03))
fig3.tight_layout()


# =============================================================================
# Load score files and compute per-prompt suppression
# =============================================================================

def _load_json_safe(path_: str) -> dict:
    if path_ and os.path.exists(path_):
        with open(path_) as _f:
            return json.load(_f)
    return {}


_all_scores: dict[str, dict] = {g: _load_json_safe(p) for g, p in cfg.score_files.items()}


def _get_run_score(run_key_: str, trait_: str, condition_: str = "default") -> float:
    """Return the final-step mean score for run_key_ across all score dicts."""
    for _sd in _all_scores.values():
        _run = _sd.get(run_key_)
        if not _run or _run.get("error") or "steps" not in _run or not _run["steps"]:
            continue
        _steps = sorted(int(_s) for _s in _run["steps"])
        _final = _run["steps"][str(max(_steps))].get(condition_, {})
        _val = _final.get(trait_, {}).get("mean", float("nan"))
        if np.isfinite(_val):
            return float(_val)
    return float("nan")


_ctrl_neg_score = _get_run_score(cfg.control_run_key, NEG)
_ctrl_pos_score = _get_run_score(cfg.control_run_key, POS)
print(
    f"\nControl run ({cfg.control_run_key!r}): "
    f"{NEG}={_ctrl_neg_score:.1f}%  {POS}={_ctrl_pos_score:.1f}%"
)

_final_neg_scores = {k: _get_run_score(k, NEG) for k in sorted_keys}
_final_pos_scores = {k: _get_run_score(k, POS) for k in sorted_keys}

supp_neg = np.array([
    _ctrl_neg_score - _final_neg_scores[k]
    if np.isfinite(_ctrl_neg_score) and np.isfinite(_final_neg_scores[k]) else float("nan")
    for k in sorted_keys
])
supp_pos = np.array([
    _ctrl_pos_score - _final_pos_scores[k]
    if np.isfinite(_ctrl_pos_score) and np.isfinite(_final_pos_scores[k]) else float("nan")
    for k in sorted_keys
])
print(f"Training data: {int(np.isfinite(supp_neg).sum())}/{N} prompts have suppression scores")


def _pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pearson r on finite pairs; returns (r, p_value)."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan"), float("nan")
    r, p = scipy_stats.pearsonr(x[mask], y[mask])
    return float(r), float(p)


# =============================================================================
# Figure 4 — Q1: |PC/SV coordinate| magnitudes vs per-trait suppression (2×2)
#
# Rows: PCA (top-2, mean-centred)  /  TruncSVD (top-2, no centering)
# Cols:
#   Col 1 — on-diagonal : Cloud ● |Dim1| → neg supp  +  Cloud ◆ |Dim2| → pos supp
#   Col 2 — off-diagonal: Cloud ● |Dim2| → neg supp  +  Cloud ◆ |Dim1| → pos supp
# =============================================================================

_pca_vecs = methods[0]["vecs"]   # (N, 2) — PCA top-2 (mean-centred)
_svd_vecs = methods[1]["vecs"]   # (N, 2) — TruncSVD top-2 (no centering)

_pca_dim1 = np.abs(_pca_vecs[:, 0])
_pca_dim2 = np.abs(_pca_vecs[:, 1])
_svd_dim1 = np.abs(_svd_vecs[:, 0])
_svd_dim2 = np.abs(_svd_vecs[:, 1])


def _draw_q1_panel(
    ax: plt.Axes,
    x_cloud_a: np.ndarray, y_cloud_a: np.ndarray,
    x_cloud_b: np.ndarray, y_cloud_b: np.ndarray,
    label_a: str, label_b: str, title: str,
) -> None:
    """Overlay two scatter clouds in one panel.

    Cloud A (red circle) : x = x_cloud_a, y = y_cloud_a
    Cloud B (blue diamond): x = x_cloud_b, y = y_cloud_b
    """
    mask_a = np.isfinite(x_cloud_a) & np.isfinite(y_cloud_a)
    ax.scatter(x_cloud_a[mask_a], y_cloud_a[mask_a],
               color="#e15759", marker="o", s=55, alpha=0.80,
               edgecolors="k", linewidths=0.4, zorder=3)

    mask_b = np.isfinite(x_cloud_b) & np.isfinite(y_cloud_b)
    ax.scatter(x_cloud_b[mask_b], y_cloud_b[mask_b],
               color="#4e79a7", marker="D", s=55, alpha=0.80,
               edgecolors="k", linewidths=0.4, zorder=3)

    r_a, p_a = _pearsonr(x_cloud_a, y_cloud_a)
    r_b, p_b = _pearsonr(x_cloud_b, y_cloud_b)

    h_a = mpatches.Patch(color="#e15759", label=f"{label_a}  r={r_a:.2f}")
    h_b = mpatches.Patch(color="#4e79a7", label=f"{label_b}  r={r_b:.2f}")
    ax.legend(handles=[h_a, h_b], fontsize=8, loc="best")

    ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Absolute coordinate value", fontsize=9)
    ax.set_ylabel("Suppression (pp)", fontsize=9)
    ax.set_title(
        f"{title}\nr_A={r_a:.3f}   r_B={r_b:.3f}",
        fontsize=9, fontweight="bold", pad=5,
    )
    ax.grid(alpha=0.20)


fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
fig4.suptitle(
    f"Q1 — PC/SV magnitude vs per-trait suppression — {NEG} / {POS} — {SLUG}\n"
    f"Col 1 on-diagonal:   A |Dim1|→{NEG} supp  +  B |Dim2|→{POS} supp\n"
    f"Col 2 off-diagonal:  A |Dim2|→{NEG} supp  +  B |Dim1|→{POS} supp",
    fontsize=11, fontweight="bold",
)

# Row 0: PCA
_draw_q1_panel(
    axes4[0, 0],
    _pca_dim1, supp_neg, _pca_dim2, supp_pos,
    f"|PC1|→{NEG} supp", f"|PC2|→{POS} supp",
    "PCA — on-diagonal",
)
_draw_q1_panel(
    axes4[0, 1],
    _pca_dim2, supp_neg, _pca_dim1, supp_pos,
    f"|PC2|→{NEG} supp", f"|PC1|→{POS} supp",
    "PCA — off-diagonal",
)

# Row 1: TruncSVD
_draw_q1_panel(
    axes4[1, 0],
    _svd_dim1, supp_neg, _svd_dim2, supp_pos,
    f"|SV1|→{NEG} supp", f"|SV2|→{POS} supp",
    "TruncSVD — on-diagonal",
)
_draw_q1_panel(
    axes4[1, 1],
    _svd_dim2, supp_neg, _svd_dim1, supp_pos,
    f"|SV2|→{NEG} supp", f"|SV1|→{POS} supp",
    "TruncSVD — off-diagonal",
)

fig4.tight_layout()


# =============================================================================
# Figure 5 — Q2: Angle predictors vs cross-suppression (5 panels)
#
# "Cross-suppression" for prompt i:
#   negative-group prompt → suppression of the POSITIVE trait (cross-trait leakage)
#   positive-group prompt → suppression of the NEGATIVE trait (cross-trait leakage)
#   neutral prompt        → NaN (excluded from plot)
#
# Five angle predictors (5 distinct: PC1/PC2 and SV1/SV2 are complementary pairs
# so only one panel per pair):
#   1. arctan2(|PC2|, |PC1|) in PCA-2D      — angle from PC1 axis
#   2. arctan2(|SV2|, |SV1|) in TruncSVD-2D — angle from SV1 axis
#   3. Angle to mean of other-group in PCA-2D
#   4. Angle to mean of other-group in TruncSVD-2D
#   5. Angle to mean of other-group in raw W
# =============================================================================

_neg_idx_sorted = [i for i, t in enumerate(sorted_types) if t == "negative"]
_pos_idx_sorted = [i for i, t in enumerate(sorted_types) if t == "positive"]

# cross_supp[i]: suppression of the OTHER trait's training outcome
_cross_supp = np.full(N, float("nan"))
for _ci, _ct in enumerate(sorted_types):
    if _ct == "negative":
        _cross_supp[_ci] = supp_pos[_ci]   # negative prompt → does it suppress positive?
    elif _ct == "positive":
        _cross_supp[_ci] = supp_neg[_ci]   # positive prompt → does it suppress negative?


def _angle_to_centroid(vecs: np.ndarray, idx: int, other_idx: list[int]) -> float:
    """Cosine angle (°) between vecs[idx] and the mean of vecs[other_idx]."""
    if not other_idx:
        return float("nan")
    v_ = vecs[idx].astype(np.float64)
    centroid_ = vecs[np.array(other_idx)].mean(axis=0).astype(np.float64)
    nv_ = np.linalg.norm(v_)
    nc_ = np.linalg.norm(centroid_)
    if nv_ < 1e-10 or nc_ < 1e-10:
        return float("nan")
    cos_ = float(np.clip(np.dot(v_, centroid_) / (nv_ * nc_), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_)))


def _other_indices(i: int) -> list[int]:
    """Return indices of the 'other' trait group for prompt i."""
    if sorted_types[i] == "negative":
        return _pos_idx_sorted
    if sorted_types[i] == "positive":
        return _neg_idx_sorted
    return []


_raw_vecs = methods[2]["vecs"]   # (N, K) — raw W (full logprob-difference matrix)

# Predictor 1: angle from PC1 axis in PCA-2D  = arctan2(|PC2|, |PC1|)
_pred_pca_axis = np.degrees(np.arctan2(
    np.abs(_pca_vecs[:, 1]), np.abs(_pca_vecs[:, 0]),
))
# Predictor 2: angle from SV1 axis in TruncSVD-2D = arctan2(|SV2|, |SV1|)
_pred_svd_axis = np.degrees(np.arctan2(
    np.abs(_svd_vecs[:, 1]), np.abs(_svd_vecs[:, 0]),
))
# Predictors 3–5: angle to other-group centroid in each space
_pred_centroid_pca = np.array([
    _angle_to_centroid(_pca_vecs, i, _other_indices(i)) for i in range(N)
])
_pred_centroid_svd = np.array([
    _angle_to_centroid(_svd_vecs, i, _other_indices(i)) for i in range(N)
])
_pred_centroid_raw = np.array([
    _angle_to_centroid(_raw_vecs, i, _other_indices(i)) for i in range(N)
])

_q2_panels = [
    (
        _pred_pca_axis,
        "arctan2(|PC2|, |PC1|)  (deg)\nangle from PC1 axis in PCA-2D",
        "PCA-2D: angle from PC1 axis",
    ),
    (
        _pred_svd_axis,
        "arctan2(|SV2|, |SV1|)  (deg)\nangle from SV1 axis in TruncSVD-2D",
        "TruncSVD-2D: angle from SV1 axis",
    ),
    (
        _pred_centroid_pca,
        "Angle to other-group centroid  (deg)\n[PCA-2D]",
        "PCA-2D: angle to other-group mean",
    ),
    (
        _pred_centroid_svd,
        "Angle to other-group centroid  (deg)\n[TruncSVD-2D]",
        "TruncSVD-2D: angle to other-group mean",
    ),
    (
        _pred_centroid_raw,
        f"Angle to other-group centroid  (deg)\n[Raw W  {N}x{K}]",
        f"Raw W: angle to other-group mean",
    ),
]

fig5, axes5 = plt.subplots(1, 5, figsize=(30, 7))
fig5.suptitle(
    f"Q2 — Angle predictors vs cross-suppression — {NEG} / {POS} — {SLUG}\n"
    f"Cross-supp: for {NEG} prompts = {POS} supp;  "
    f"for {POS} prompts = {NEG} supp   (neutral=NaN, excluded)",
    fontsize=11, fontweight="bold", y=1.02,
)

for _ax5, (_pred, _xlabel, _title) in zip(axes5, _q2_panels):
    _r5, _p5 = _pearsonr(_pred, _cross_supp)
    for _i5, _t5 in enumerate(sorted_types):
        _x5 = _pred[_i5]
        _y5 = _cross_supp[_i5]
        if not (np.isfinite(_x5) and np.isfinite(_y5)):
            continue
        _c5 = GROUP_COLORS.get(_t5, "#aaaaaa")
        _ax5.scatter(_x5, _y5, color=_c5, s=55, edgecolors="k",
                     linewidths=0.4, alpha=0.85, zorder=3)
        _ax5.annotate(
            short_label(sorted_keys[_i5]), (_x5, _y5),
            xytext=(3, 3), textcoords="offset points",
            fontsize=5, color="#333", zorder=4,
        )
    _ax5.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
    _pstr5 = f"  p={_p5:.3f}" if np.isfinite(_p5) else ""
    _ax5.set_title(f"{_title}\nr={_r5:.3f}{_pstr5}", fontsize=9, fontweight="bold", pad=5)
    _ax5.set_xlabel(_xlabel, fontsize=8.5)
    _ax5.set_ylabel("Cross-suppression (pp)", fontsize=9)
    _ax5.grid(alpha=0.20)

_q2_legend_handles = [
    mpatches.Patch(color=GROUP_COLORS["negative"], label=f"{NEG} prompts"),
    mpatches.Patch(color=GROUP_COLORS["positive"], label=f"{POS} prompts"),
]
fig5.legend(handles=_q2_legend_handles, loc="lower center", ncol=2,
            fontsize=10, bbox_to_anchor=(0.5, -0.04))
fig5.tight_layout()


# =============================================================================
# Save plots
# =============================================================================
_exp_subdir = cfg.name or ""
PLOT_DIR = os.path.join(
    cfg.plot_dir,
    *([_exp_subdir] if _exp_subdir else []),
    "pca",
    "angle_analysis",
)
os.makedirs(PLOT_DIR, exist_ok=True)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")

p1 = os.path.join(PLOT_DIR, f"angle_heatmap_{ts}.png")
p2 = os.path.join(PLOT_DIR, f"angle_cross_trait_{ts}.png")
p3 = os.path.join(PLOT_DIR, f"angle_per_prompt_{ts}.png")
p4 = os.path.join(PLOT_DIR, f"angle_dim_suppression_{ts}.png")
p5 = os.path.join(PLOT_DIR, f"angle_cross_suppression_{ts}.png")

fig1.savefig(p1, dpi=130, bbox_inches="tight")
fig2.savefig(p2, dpi=130, bbox_inches="tight")
fig3.savefig(p3, dpi=130, bbox_inches="tight")
fig4.savefig(p4, dpi=130, bbox_inches="tight")
fig5.savefig(p5, dpi=130, bbox_inches="tight")
plt.close("all")

print(f"\nSaved: {p1}")
print(f"Saved: {p2}")
print(f"Saved: {p3}")
print(f"Saved: {p4}")
print(f"Saved: {p5}")


# =============================================================================
# Save JSON summary
# =============================================================================
_RESULT_DIR = os.path.join(_REPO, "results")
os.makedirs(_RESULT_DIR, exist_ok=True)

exp_slug = cfg.name or f"{NEG}_{POS}_{SLUG}"
json_path = os.path.join(_RESULT_DIR, f"angle_analysis_{exp_slug}_{ts}.json")

# Strip the large `vals` lists before serialising to keep file small
def _compact_stats(stats: dict) -> dict:
    return {
        cat: {k: v for k, v in s.items() if k != "vals"}
        for cat, s in stats.items()
    }

result = {
    "experiment":     exp_slug,
    "positive_trait": POS,
    "negative_trait": NEG,
    "study_model":    SLUG,
    "n_prompts":      N,
    "n_train_examples": K,
    "timestamp":      ts,
    "sorted_keys":    sorted_keys,
    "sorted_types":   sorted_types,
    "control_scores": {
        "neg": float(_ctrl_neg_score),
        "pos": float(_ctrl_pos_score),
    },
    "suppression": {
        k: {
            "supp_neg":   float(supp_neg[i]),
            "supp_pos":   float(supp_pos[i]),
            "cross_supp": float(_cross_supp[i]),
            "group_type": sorted_types[i],
        }
        for i, k in enumerate(sorted_keys)
    },
    "q2_predictors": {
        k: {
            "angle_from_pca_axis":    float(_pred_pca_axis[i]),
            "angle_from_svd_axis":    float(_pred_svd_axis[i]),
            "angle_to_centroid_pca":  float(_pred_centroid_pca[i]),
            "angle_to_centroid_svd":  float(_pred_centroid_svd[i]),
            "angle_to_centroid_raw":  float(_pred_centroid_raw[i]),
        }
        for i, k in enumerate(sorted_keys)
    },
    "methods": [
        {
            "name":    m["name"],
            "tag":     m["tag"],
            "label":   m["label"],
            "stats":   _compact_stats(m["stats"]),
            "per_prompt": {
                k: {
                    "mean_angle_to_neg": float(m["mean_to_neg"][i]),
                    "mean_angle_to_pos": float(m["mean_to_pos"][i]),
                    "group_type":         sorted_types[i],
                }
                for i, k in enumerate(sorted_keys)
            },
        }
        for m in methods
    ],
}

with open(json_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved JSON: {json_path}")

# =============================================================================
# Print summary table
# =============================================================================
print(f"\n{'='*70}")
print(f"ANGLE SUMMARY — {NEG} / {POS}")
print(f"{'='*70}")
header = f"{'Pair category':<28}" + "".join(f"  {m['name']:>22}" for m in methods)
print(header)
print("-" * len(header))

for cat, label in [
    ("within_negative",  f"Within {NEG}"),
    ("within_positive",  f"Within {POS}"),
    ("cross_trait",      "Cross-trait"),
    ("neg_to_neutral",   f"{NEG} × Neutral"),
    ("pos_to_neutral",   f"{POS} × Neutral"),
]:
    row = f"{label:<28}"
    for m in methods:
        s = m["stats"][cat]
        if s["n"] == 0:
            row += f"  {'N/A':>22}"
        else:
            row += f"  {s['mean']:6.1f}° ± {s['std']:4.1f}°  (n={s['n']:3d})"
    print(row)

print(f"\nPlots saved to: {PLOT_DIR}")
print("Done.")
