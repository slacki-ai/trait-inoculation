#!/usr/bin/env python3
"""
plot_pca_prompts.py — PCA of the per-example logprob-difference matrix.

Two versions are shown side by side:

  Fixed   W_fixed[n, k] = lp_per_tok(completion_k | fixed_prefix_n + instr_k)
                         − lp_per_tok(completion_k | instr_k)

  Mix     W_mix[n, k]   = lp_per_tok(completion_k | rephrasings_n[k] + instr_k)
                         − lp_per_tok(completion_k | instr_k)

where rephrasings_n[k] is the k-th rephrasing of prompt n (index-matched).

PCA embeds the 27 prompts in 2D by finding directions of maximum variance
across the 1000 training examples.

The mix version requires compute_perplexity_heuristic_mix.py to have run first
(it populates the "lp_train_mix" field in the perplexity heuristic JSON).
If mix data is absent, only the fixed version is shown.

Figure layout
─────────────
One row of 3 panels per version (Fixed / Mix):
  (A) Coloured by source group (v3 / v4 / v5 / neg)
  (B) Coloured by Playful suppression
  (C) Coloured by PH (sanity-check: should track PC1 tightly)

Bottom row: correlation heatmap — Pearson r between PC scores and
            scalar metrics, one block per version.

Usage:
    python plot_pca_prompts.py
Output:
    plots/plot_pca_prompts_<timestamp>.png
"""

import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE        = "/Users/claude/vibe-research/inoculation-bootstrap-heuristic"
PERP_PATH   = f"{BASE}/results/perplexity_heuristic_qwen2.5-7b-instruct.json"
ELICIT_PATH = f"{BASE}/results/elicitation_scores.json"
V3_PATH     = f"{BASE}/results/scores_multi_prompt_v3_qwen2.5-7b-instruct.json"
V4_PATH     = f"{BASE}/results/scores_multi_prompt_v4_qwen2.5-7b-instruct.json"
V5_PATH     = f"{BASE}/results/scores_multi_prompt_v5_qwen2.5-7b-instruct.json"
VNEG_PATH   = f"{BASE}/results/scores_multi_prompt_neg_qwen2.5-7b-instruct.json"
PLOT_DIR    = f"{BASE}/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Prompt lists
# ---------------------------------------------------------------------------
V3_PROMPT_NAMES = [
    "clown_persona", "humor_matters", "enjoys_joking", "joke_nevermind",
    "clowns_interesting", "playfulness_trait", "playfulness_enriches",
    "laughter_medicine", "had_fun_today",
]
V4_PROMPT_NAMES = [
    "corrected_inoculation", "whimsical", "witty",
    "strong_elicitation", "comedian_answers", "comedian_mindset",
]
V5_PROMPT_NAMES = [
    "the_sky_is_blue", "i_like_cats", "professional_tone",
    "financial_advisor", "be_concise", "think_step_by_step",
]
VNEG_PROMPT_NAMES = [
    "corrected_inoculation_neg", "whimsical_neg", "witty_neg",
    "strong_elicitation_neg", "comedian_answers_neg", "comedian_mindset_neg",
]
ALL_PROMPT_NAMES = (
    V3_PROMPT_NAMES + V4_PROMPT_NAMES + V5_PROMPT_NAMES + VNEG_PROMPT_NAMES
)
SOURCE_BY_KEY = (
    {k: "v3"  for k in V3_PROMPT_NAMES}
    | {k: "v4"  for k in V4_PROMPT_NAMES}
    | {k: "v5"  for k in V5_PROMPT_NAMES}
    | {k: "neg" for k in VNEG_PROMPT_NAMES}
)

def short_label(key: str) -> str:
    return key.replace("_neg", "⁻").replace("_mix", "~")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(PERP_PATH) as f:
    perp_data = json.load(f)
with open(ELICIT_PATH) as f:
    elicit = json.load(f)

def _load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

v3   = _load(V3_PATH)
v4   = _load(V4_PATH)
v5   = _load(V5_PATH)
vneg = _load(VNEG_PATH)

perp_prompts     = perp_data["prompts"]
lp_train_default = np.array(perp_data["baseline"]["lp_train_default"])

# ---------------------------------------------------------------------------
# Build W matrices
# ---------------------------------------------------------------------------
keys_in_data = [k for k in ALL_PROMPT_NAMES if k in perp_prompts]
N = len(keys_in_data)
K = len(lp_train_default)

W_fixed = np.zeros((N, K))
for i, key in enumerate(keys_in_data):
    W_fixed[i] = np.array(perp_prompts[key]["lp_train_inoc"]) - lp_train_default

# Mix matrix — only if lp_train_mix is present in the JSON
has_mix = all("lp_train_mix" in perp_prompts[k] for k in keys_in_data)
if has_mix:
    W_mix = np.zeros((N, K))
    for i, key in enumerate(keys_in_data):
        W_mix[i] = np.array(perp_prompts[key]["lp_train_mix"]) - lp_train_default
    print(f"Mix data available: W_mix shape {W_mix.shape}")
else:
    W_mix = None
    n_missing = sum(1 for k in keys_in_data if "lp_train_mix" not in perp_prompts[k])
    print(f"Mix data NOT available ({n_missing}/{N} prompts missing lp_train_mix).")
    print("Run compute_perplexity_heuristic_mix.py first to generate mix logprobs.")

# ---------------------------------------------------------------------------
# PCA helper
# ---------------------------------------------------------------------------
def run_pca(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (coords (N,2), explained_variance_ratio (2,))."""
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(W)
    return coords, pca.explained_variance_ratio_


# ---------------------------------------------------------------------------
# Scalar metrics per prompt
# ---------------------------------------------------------------------------
_baseline_playful = elicit["neutral"]["scores"]["Playful"]["mean"]

def get_elicitation(key):
    e = elicit.get(key)
    return (e["scores"]["Playful"]["mean"] - _baseline_playful) if e else float("nan")

def get_final_playful(scores_dict, run_key, condition="default"):
    run = scores_dict.get(run_key)
    if not run or run.get("error"):
        return float("nan")
    steps = sorted(int(s) for s in run["steps"].keys())
    return run["steps"][str(max(steps))][condition]["Playful"]["mean"]

ctrl_playful = get_final_playful(v3, "no_inoculation")

def get_suppression(key):
    for scores_dict in (v3, v4, v5, vneg):
        fp = get_final_playful(scores_dict, key)
        if not np.isnan(fp):
            return ctrl_playful - fp
    return float("nan")

def lls_scalar(key):
    w     = W_fixed[keys_in_data.index(key)]
    mask  = ~np.isnan(w)
    w_    = w[mask]
    mean_w = float(np.mean(w_))
    std_w  = float(np.std(w_, ddof=1)) if len(w_) > 1 else float("nan")
    snr    = (mean_w / std_w) if (std_w > 0) else float("nan")
    frac_p = float(np.mean(w_ > 0))
    return mean_w, std_w, snr, frac_p

scalars = {}
for key in keys_in_data:
    ph, std_w, snr, frac_pos = lls_scalar(key)
    scalars[key] = dict(
        elicitation = get_elicitation(key),
        ph          = ph,
        frac_pos    = frac_pos,
        std         = std_w,
        snr         = snr,
        suppression = get_suppression(key),
    )

# For mix version: PH_mix uses W_mix row means
if W_mix is not None:
    for i, key in enumerate(keys_in_data):
        scalars[key]["ph_mix"] = float(np.nanmean(W_mix[i]))

# ---------------------------------------------------------------------------
# Correlation helper
# ---------------------------------------------------------------------------
METRIC_NAMES  = ["elicitation", "ph", "frac_pos", "std", "snr", "suppression"]
METRIC_LABELS = ["Elicitation", "PH", "Frac pos γ", "Std σ", "SNR", "Suppression"]

def compute_correlations(pc1, pc2):
    corr = {}
    for mn in METRIC_NAMES:
        vals = np.array([scalars[k][mn] for k in keys_in_data])
        mask = ~np.isnan(vals)
        if mask.sum() < 3:
            corr[mn] = (float("nan"),) * 4
            continue
        r1, p1 = scipy_stats.pearsonr(pc1[mask], vals[mask])
        r2, p2 = scipy_stats.pearsonr(pc2[mask], vals[mask])
        corr[mn] = (r1, p1, r2, p2)
    return corr

# ---------------------------------------------------------------------------
# Source/marker style
# ---------------------------------------------------------------------------
SOURCE_STYLE = {
    "v3":  dict(marker="o", color="#e15759", label="v3  weak–medium"),
    "v4":  dict(marker="D", color="#f28e2b", label="v4  strong"),
    "v5":  dict(marker="s", color="#4e79a7", label="v5  near-zero"),
    "neg": dict(marker="v", color="#76b7b2", label="neg  negative"),
}

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_source_panel(ax, pc1, pc2, var_ratio, title):
    for i, key in enumerate(keys_in_data):
        src = SOURCE_BY_KEY.get(key, "v3")
        st  = SOURCE_STYLE[src]
        ax.scatter(pc1[i], pc2[i],
                   color=st["color"], marker=st["marker"],
                   s=80, edgecolors="black", linewidths=0.5, zorder=3)
        ax.annotate(short_label(key), xy=(pc1[i], pc2[i]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=6.5, color="#222222", zorder=4)
    _finish_pca_ax(ax, pc1, pc2, var_ratio, title)

    handles = [
        mlines.Line2D([], [], marker=v["marker"], color=v["color"],
                      markeredgecolor="black", markeredgewidth=0.5,
                      markersize=7, linestyle="none", label=v["label"])
        for v in SOURCE_STYLE.values()
    ]
    ax.legend(handles=handles, fontsize=7.5, loc="best",
               framealpha=0.85, edgecolor="#cccccc")


def draw_color_panel(ax, pc1, pc2, var_ratio, color_vals,
                     cmap_name, vmin, vmax, title, cbar_label):
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    for i, key in enumerate(keys_in_data):
        cv  = color_vals[i]
        c   = "#aaaaaa" if np.isnan(cv) else cmap(norm(cv))
        src = SOURCE_BY_KEY.get(key, "v3")
        ax.scatter(pc1[i], pc2[i],
                   color=c, marker=SOURCE_STYLE[src]["marker"],
                   s=80, edgecolors="black", linewidths=0.5, zorder=3)
        ax.annotate(short_label(key), xy=(pc1[i], pc2[i]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=6.5, color="#222222", zorder=4)
    _finish_pca_ax(ax, pc1, pc2, var_ratio, title)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=8)


def _finish_pca_ax(ax, pc1, pc2, var_ratio, title):
    ax.axhline(0, color="#bbbbbb", linewidth=0.7, linestyle="--")
    ax.axvline(0, color="#bbbbbb", linewidth=0.7, linestyle="--")
    ax.set_xlabel(f"PC1  ({var_ratio[0]:.1%} var)", fontsize=9)
    ax.set_ylabel(f"PC2  ({var_ratio[1]:.1%} var)", fontsize=9)
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=5)
    ax.grid(True, alpha=0.25)


def draw_corr_heatmap(ax, corr_results, title):
    r_matrix = np.array([
        [corr_results[mn][0], corr_results[mn][2]]
        for mn in METRIC_NAMES
    ])
    im = ax.imshow(r_matrix.T, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(METRIC_LABELS)))
    ax.set_xticklabels(METRIC_LABELS, fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["PC1", "PC2"], fontsize=9)
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=5)
    for j, mn in enumerate(METRIC_NAMES):
        for pi, pc_idx in enumerate([0, 2]):
            r_val = corr_results[mn][pc_idx]
            p_val = corr_results[mn][pc_idx + 1]
            if np.isnan(r_val):
                txt = "n/a"
            else:
                sig = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "")
                txt = f"{r_val:+.2f}{sig}"
            ax.text(j, pi, txt, ha="center", va="center", fontsize=9,
                    color="white" if abs(r_val if not np.isnan(r_val) else 0) > 0.5
                    else "black", fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="Pearson r")
    ax.set_xlabel("Scalar metric", fontsize=9)


# ---------------------------------------------------------------------------
# Run PCAs and collect results
# ---------------------------------------------------------------------------
versions = []

# Fixed version — always available
coords_f, var_f = run_pca(W_fixed)
pc1_f, pc2_f    = coords_f[:, 0], coords_f[:, 1]
corr_f          = compute_correlations(pc1_f, pc2_f)
ph_vals_f       = np.array([scalars[k]["ph"]         for k in keys_in_data])
supp_vals_f     = np.array([scalars[k]["suppression"] for k in keys_in_data])
versions.append(dict(
    label      = "Fixed prefix",
    coords     = coords_f,
    var_ratio  = var_f,
    corr       = corr_f,
    ph_vals    = ph_vals_f,
    supp_vals  = supp_vals_f,
))

# Mix version — available after compute_perplexity_heuristic_mix.py
if W_mix is not None:
    coords_m, var_m = run_pca(W_mix)
    pc1_m, pc2_m    = coords_m[:, 0], coords_m[:, 1]
    corr_m          = compute_correlations(pc1_m, pc2_m)
    ph_vals_m       = np.array([scalars[k]["ph_mix"] for k in keys_in_data])
    versions.append(dict(
        label      = "Mix (rephrased) prefix",
        coords     = coords_m,
        var_ratio  = var_m,
        corr       = corr_m,
        ph_vals    = ph_vals_m,
        supp_vals  = supp_vals_f,   # suppression Y-axis is the same for both
    ))

# Print summary
for ver in versions:
    vr   = ver["var_ratio"]
    pc1_ = ver["coords"][:, 0]
    pc2_ = ver["coords"][:, 1]
    corr = ver["corr"]
    print(f"\n── {ver['label']} ──")
    print(f"   PC1 {vr[0]:.1%}  PC2 {vr[1]:.1%}")
    print(f"   {'metric':<14}  {'r(PC1)':>8}  {'p':>7}  {'r(PC2)':>8}  {'p':>7}")
    for mn, ml in zip(METRIC_NAMES, METRIC_LABELS):
        r1, p1, r2, p2 = corr[mn]
        print(f"   {ml:<14}  {r1:>+8.3f}  {p1:>7.3f}  {r2:>+8.3f}  {p2:>7.3f}")

# ---------------------------------------------------------------------------
# Build figure
#   n_versions rows of PCA panels (3 per row) + n_versions rows of heatmaps
# ---------------------------------------------------------------------------
n_ver  = len(versions)
n_rows = n_ver * 2       # PCA row + heatmap row per version
height_ratios = [2.2, 0.9] * n_ver

fig = plt.figure(figsize=(18, 7.0 * n_ver), constrained_layout=True)
gs  = fig.add_gridspec(n_rows, 3, height_ratios=height_ratios, hspace=0.4)

for vi, ver in enumerate(versions):
    pca_row  = vi * 2
    corr_row = vi * 2 + 1

    coords   = ver["coords"]
    var_ratio= ver["var_ratio"]
    pc1_     = coords[:, 0]
    pc2_     = coords[:, 1]
    ph_vals  = ver["ph_vals"]
    supp_vals= ver["supp_vals"]
    corr     = ver["corr"]
    lbl      = ver["label"]

    # ── Panel A: source colours ────────────────────────────────────────────
    ax_src  = fig.add_subplot(gs[pca_row, 0])
    draw_source_panel(ax_src, pc1_, pc2_, var_ratio,
                      f"({chr(65 + vi*3)})  {lbl} — source group")

    # ── Panel B: suppression ──────────────────────────────────────────────
    ax_supp = fig.add_subplot(gs[pca_row, 1])
    finite  = supp_vals[~np.isnan(supp_vals)]
    draw_color_panel(ax_supp, pc1_, pc2_, var_ratio,
                     color_vals = supp_vals,
                     cmap_name  = "RdYlGn",
                     vmin       = finite.min() if len(finite) else -10,
                     vmax       = finite.max() if len(finite) else 70,
                     title      = f"({chr(66 + vi*3)})  {lbl} — Playful suppression",
                     cbar_label = "Playful suppression (pp)")

    # ── Panel C: PH ───────────────────────────────────────────────────────
    ax_ph   = fig.add_subplot(gs[pca_row, 2])
    draw_color_panel(ax_ph, pc1_, pc2_, var_ratio,
                     color_vals = ph_vals,
                     cmap_name  = "coolwarm",
                     vmin       = ph_vals.min(),
                     vmax       = ph_vals.max(),
                     title      = f"({chr(67 + vi*3)})  {lbl} — PH = mean(wᵢ)",
                     cbar_label = "PH")

    # ── Correlation heatmap ────────────────────────────────────────────────
    ax_corr = fig.add_subplot(gs[corr_row, :])
    draw_corr_heatmap(ax_corr, corr,
                      f"Pearson r — PC scores vs scalar metrics  [{lbl}]")

# ---------------------------------------------------------------------------
# Overall title
# ---------------------------------------------------------------------------
fig.suptitle(
    "PCA of W (27 prompts × 1000 training examples)\n"
    "W[n, k] = lp_per_tok(completion_k | prefix_n[k] + instr_k)"
    " − lp_per_tok(completion_k | instr_k)",
    fontsize=11, fontweight="bold",
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
fname = f"plot_pca_prompts_{ts}.png"
fpath = os.path.join(PLOT_DIR, fname)
fig.savefig(fpath, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {fpath}")
