#!/usr/bin/env python3
"""
plot_pca_prompts.py — PCA of the per-example logprob-difference matrix.

Two versions are shown side by side:

  Fixed   W_fixed[n, k] = lp_per_tok(completion_k | fixed_prefix_n + instr_k)
                         − lp_per_tok(completion_k | instr_k)

  Mix     W_mix[n, k]   = lp_per_tok(completion_k | rephrasings_n[k] + instr_k)
                         − lp_per_tok(completion_k | instr_k)

where rephrasings_n[k] is the k-th rephrasing of prompt n (index-matched).

PCA embeds the prompts in 4D by finding directions of maximum variance across
the 1000 training examples.

Figure layout
─────────────
One row of 7 panels per version (Fixed / Mix):
  Cols 1–4 use PC1 vs PC2 axes:
  (A) Coloured by source group (v3 / v4 / v5 / neg / fr_v3 / fr_v4 / fr_neg)
  (B) Coloured by Playful suppression
  (C) Coloured by French suppression
  (D) Coloured by PH (sanity-check: should track PC1 tightly)
  Cols 5–7 use PC3 vs PC4 axes:
  (E) Coloured by source group
  (F) Coloured by Playful suppression
  (G) Coloured by French suppression

Bottom row: correlation heatmap — Pearson r between PC scores (PC1–PC4) and
            scalar metrics, one block per version.

Usage:
    python plot_pca_prompts.py [--config {all,french_only,playful_only}]
Output:
    plots/plot_pca_prompts_pointwise[_<config>]_<timestamp>.png
    plots/plot_pca_prompts_tokens[_<config>]_<timestamp>.png
"""

import json
import os
from datetime import datetime
import argparse

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
# CLI arguments
# ---------------------------------------------------------------------------
_ap = argparse.ArgumentParser(description="PCA of inoculation prompts")
_ap.add_argument(
    "--config", default="all",
    choices=["all", "french_only", "playful_only"],
    help="Prompt subset: all (48), french_only (21 French + 6 neutral = 27), "
         "or playful_only (21 Playful + 6 neutral = 27)",
)
CONFIG = _ap.parse_args().config
CONFIG_SUFFIX = f"_{CONFIG}" if CONFIG != "all" else ""
print(f"\nConfig: {CONFIG!r}  (suffix: {CONFIG_SUFFIX!r})")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE        = "/Users/claude/vibe-research/inoculation-bootstrap-heuristic"
PERP_PATH   = f"{BASE}/results/perplexity_heuristic_qwen2.5-7b-instruct.json"
ELICIT_PATH = f"{BASE}/results/elicitation_scores.json"
# Playful training results
V3_PATH     = f"{BASE}/results/scores_multi_prompt_v3_qwen2.5-7b-instruct.json"
V4_PATH     = f"{BASE}/results/scores_multi_prompt_v4_qwen2.5-7b-instruct.json"
V5_PATH     = f"{BASE}/results/scores_multi_prompt_v5_qwen2.5-7b-instruct.json"
VNEG_PATH   = f"{BASE}/results/scores_multi_prompt_neg_qwen2.5-7b-instruct.json"
# French training results
FRV3_PATH   = f"{BASE}/results/scores_multi_prompt_french_v3_qwen2.5-7b-instruct.json"
FRV4_PATH   = f"{BASE}/results/scores_multi_prompt_french_v4_qwen2.5-7b-instruct.json"
FRNEG_PATH  = f"{BASE}/results/scores_multi_prompt_french_neg_qwen2.5-7b-instruct.json"
TOKENS_PATH = f"{BASE}/results/perplexity_heuristic_tokens_qwen2.5-7b-instruct.json"
PLOT_DIR    = f"{BASE}/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Prompt lists
# ---------------------------------------------------------------------------
# Playful prompts
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
# French prompts
FRENCH_V3_NAMES = [
    "french_persona", "french_matters", "enjoys_french", "paris_nevermind",
    "french_interesting", "french_trait", "french_enriches", "french_love", "french_today",
]
FRENCH_V4_NAMES = [
    "french_agent", "fluent_french", "natural_french",
    "answer_french", "french_answers", "think_french",
]
FRENCH_NEG_NAMES = [
    "french_agent_neg", "fluent_french_neg", "natural_french_neg",
    "answer_french_neg", "french_answers_neg", "think_french_neg",
]

# All 48 unique prompt keys
ALL_PROMPT_NAMES_48 = (
    V3_PROMPT_NAMES + V4_PROMPT_NAMES + V5_PROMPT_NAMES + VNEG_PROMPT_NAMES
    + FRENCH_V3_NAMES + FRENCH_V4_NAMES + FRENCH_NEG_NAMES
)

# Active subset — filtered by --config
if CONFIG == "french_only":
    ALL_PROMPT_NAMES = (
        FRENCH_V3_NAMES + FRENCH_V4_NAMES + FRENCH_NEG_NAMES + V5_PROMPT_NAMES
    )
elif CONFIG == "playful_only":
    ALL_PROMPT_NAMES = (
        V3_PROMPT_NAMES + V4_PROMPT_NAMES + VNEG_PROMPT_NAMES + V5_PROMPT_NAMES
    )
else:  # "all"
    ALL_PROMPT_NAMES = ALL_PROMPT_NAMES_48

print(f"Active prompts: {len(ALL_PROMPT_NAMES)}")

SOURCE_BY_KEY = (
    {k: "v3"     for k in V3_PROMPT_NAMES}
    | {k: "v4"   for k in V4_PROMPT_NAMES}
    | {k: "v5"   for k in V5_PROMPT_NAMES}
    | {k: "neg"  for k in VNEG_PROMPT_NAMES}
    | {k: "fr_v3"  for k in FRENCH_V3_NAMES}
    | {k: "fr_v4"  for k in FRENCH_V4_NAMES}
    | {k: "fr_neg" for k in FRENCH_NEG_NAMES}
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
fr_v3  = _load(FRV3_PATH)
fr_v4  = _load(FRV4_PATH)
fr_neg = _load(FRNEG_PATH)

perp_prompts     = perp_data["prompts"]
lp_train_default = np.array(perp_data["baseline"]["lp_train_default"])

# ---------------------------------------------------------------------------
# Build W matrices
# ---------------------------------------------------------------------------
keys_in_data = [k for k in ALL_PROMPT_NAMES if k in perp_prompts]
N = len(keys_in_data)
K = len(lp_train_default)
print(f"Prompts in data: {N} / {len(ALL_PROMPT_NAMES)} expected")

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

# ---------------------------------------------------------------------------
# PCA helper
# ---------------------------------------------------------------------------
N_COMPONENTS = 4  # number of PCs to extract


def run_pca(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (coords (N, N_COMPONENTS), explained_variance_ratio (N_COMPONENTS,))."""
    n_comp = min(N_COMPONENTS, W.shape[0], W.shape[1])
    pca    = PCA(n_components=n_comp, random_state=42)
    coords = pca.fit_transform(W)
    var    = pca.explained_variance_ratio_
    # Pad to N_COMPONENTS if fewer were available
    if n_comp < N_COMPONENTS:
        pad    = N_COMPONENTS - n_comp
        coords = np.hstack([coords, np.zeros((coords.shape[0], pad))])
        var    = np.concatenate([var, np.zeros(pad)])
    return coords, var


def build_W_tokens(
    key_list: list[str],
    lp_field: str = "lp_train_inoc_tokens",
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build N × (K·L) token-wise logprob-difference matrix and run PCA.

    Concatenates per-token diffs across all K completions for each prompt:
        row_n = concat_k [ lp_tokens[n][k] − lp_default_tokens[k] ]

    Returns (coords_N×N_COMPONENTS, explained_variance_ratio_N_COMPONENTS) for the
    full key_list, using NaN rows for any prompt missing the field.
    Returns (None, None) if the tokens file is absent or too few prompts qualify.
    """
    if not os.path.exists(TOKENS_PATH):
        print(f"  Tokens file not found: {TOKENS_PATH} — skipping token PCA")
        return None, None

    print(f"\n  Loading tokens data ({lp_field}) …", flush=True)
    with open(TOKENS_PATH) as f:
        tokens_data = json.load(f)

    baseline_toks = tokens_data["baseline"]["lp_train_default_tokens"]  # K lists
    prompts_toks  = tokens_data["prompts"]

    keys_ok = [k for k in key_list
               if k in prompts_toks and lp_field in prompts_toks[k]]
    if len(keys_ok) < N_COMPONENTS + 1:
        print(f"  Only {len(keys_ok)} prompts have {lp_field} — skipping token PCA")
        return None, None

    rows: list[list[float]] = []
    for key in keys_ok:
        inoc_toks = prompts_toks[key][lp_field]
        row: list[float] = []
        for k in range(len(baseline_toks)):
            def_t  = baseline_toks[k]
            inoc_t = inoc_toks[k] if k < len(inoc_toks) else []
            L      = min(len(def_t), len(inoc_t))
            row.extend(
                round(float(inoc_t[l]) - float(def_t[l]), 4) for l in range(L)
            )
        rows.append(row)

    min_len = min(len(r) for r in rows)
    W = np.array([r[:min_len] for r in rows], dtype=np.float32)
    W = np.where(np.isfinite(W), W, 0.0)

    n_comp = min(N_COMPONENTS, W.shape[0], W.shape[1])
    pca    = PCA(n_components=n_comp, random_state=42)
    coords = pca.fit_transform(W)
    var    = pca.explained_variance_ratio_
    if n_comp < N_COMPONENTS:
        pad    = N_COMPONENTS - n_comp
        coords = np.hstack([coords, np.zeros((coords.shape[0], pad))])
        var    = np.concatenate([var, np.zeros(pad)])

    var_str = "  ".join(f"PC{i+1}={var[i]*100:.1f}%" for i in range(len(var)))
    print(
        f"  W_tokens ({lp_field}): {len(keys_ok)} prompts × {W.shape[1]} features  "
        f"{var_str}"
    )

    # Map back to full key_list, filling NaN for any missing prompt
    key_to_row  = {k: i for i, k in enumerate(keys_ok)}
    coords_full = np.full((len(key_list), N_COMPONENTS), float("nan"))
    for i, k in enumerate(key_list):
        if k in key_to_row:
            coords_full[i] = coords[key_to_row[k]]
    return coords_full, var


# ---------------------------------------------------------------------------
# Scalar metrics per prompt
# ---------------------------------------------------------------------------
_baseline_playful = elicit["neutral"]["scores"]["Playful"]["mean"]
_baseline_french  = elicit["neutral"]["scores"]["French"]["mean"]

def get_elicitation_playful(key):
    e = elicit.get(key)
    return (e["scores"]["Playful"]["mean"] - _baseline_playful) if e else float("nan")

def get_elicitation_french(key):
    e = elicit.get(key)
    return (e["scores"]["French"]["mean"] - _baseline_french) if e else float("nan")

def get_final_score(scores_dict, run_key, trait, condition="default"):
    """Get final-step score for a given trait and condition."""
    run = scores_dict.get(run_key)
    if not run or run.get("error"):
        return float("nan")
    steps = sorted(int(s) for s in run["steps"].keys())
    final = run["steps"][str(max(steps))].get(condition, {})
    return final.get(trait, {}).get("mean", float("nan"))

# Control scores at final checkpoint (no inoculation baseline)
ctrl_playful = get_final_score(v3, "no_inoculation", "Playful")
ctrl_french  = get_final_score(v3, "no_inoculation", "French")
print(f"ctrl_playful={ctrl_playful:.1f}%, ctrl_french={ctrl_french:.1f}%")

def get_suppression(key, trait, use_mix=False):
    """ctrl − final(default) for a given trait.

    use_mix=False → look up 'key'       in fixed training runs
    use_mix=True  → look up 'key_mix'   in mix training runs
    """
    lookup_key = (key + "_mix") if use_mix else key
    ctrl = ctrl_playful if trait == "Playful" else ctrl_french
    for scores_dict in (v3, v4, v5, vneg, fr_v3, fr_v4, fr_neg):
        fp = get_final_score(scores_dict, lookup_key, trait)
        if not np.isnan(fp):
            return ctrl - fp
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
        elicit_playful          = get_elicitation_playful(key),
        elicit_french           = get_elicitation_french(key),
        ph                      = ph,
        frac_pos                = frac_pos,
        std                     = std_w,
        snr                     = snr,
        playful_supp_fixed      = get_suppression(key, "Playful", use_mix=False),
        french_supp_fixed       = get_suppression(key, "French",  use_mix=False),
        playful_supp_mix        = get_suppression(key, "Playful", use_mix=True),
        french_supp_mix         = get_suppression(key, "French",  use_mix=True),
    )

# For mix version: PH_mix, γ_mix, σ_mix, SNR_mix all from W_mix rows
if W_mix is not None:
    for i, key in enumerate(keys_in_data):
        w_    = W_mix[i][~np.isnan(W_mix[i])]
        ph_m  = float(np.nanmean(W_mix[i]))
        std_m = float(np.std(w_, ddof=1)) if len(w_) > 1 else float("nan")
        scalars[key]["ph_mix"]       = ph_m
        scalars[key]["frac_pos_mix"] = float(np.mean(w_ > 0))
        scalars[key]["std_mix"]      = std_m
        scalars[key]["snr_mix"]      = (ph_m / std_m) if (std_m > 0) else float("nan")

# ---------------------------------------------------------------------------
# Correlation helper
# ---------------------------------------------------------------------------
METRIC_NAMES_FIXED = ["elicit_playful", "elicit_french", "ph",
                      "frac_pos", "std", "snr",
                      "playful_supp_fixed", "french_supp_fixed"]
METRIC_NAMES_MIX   = ["elicit_playful", "elicit_french", "ph_mix",
                      "frac_pos_mix", "std_mix", "snr_mix",
                      "playful_supp_mix", "french_supp_mix"]
METRIC_LABELS      = ["Elicit(Play)", "Elicit(Fr)", "PH",
                      "γ frac+", "σ std", "SNR",
                      "Supp(Play)", "Supp(Fr)"]


def compute_correlations(coords: np.ndarray, metric_names: list[str]) -> dict:
    """Compute Pearson r between each PC and each scalar metric.

    coords: (N, n_pcs) array of PC scores.
    Returns dict: metric_name → tuple (r1, p1, r2, p2, ...) with 2 values per PC
    (r-value then p-value), covering all n_pcs components.
    """
    n_pcs = coords.shape[1]
    corr  = {}
    for mn in metric_names:
        vals = np.array([scalars[k][mn] for k in keys_in_data])
        mask = ~np.isnan(vals)
        if mask.sum() < 3:
            corr[mn] = tuple([float("nan")] * (2 * n_pcs))
            continue
        row = []
        for pc_idx in range(n_pcs):
            pc_vals = coords[:, pc_idx]
            pc_mask = mask & ~np.isnan(pc_vals)
            if pc_mask.sum() < 3:
                row.extend([float("nan"), float("nan")])
            else:
                r, p = scipy_stats.pearsonr(pc_vals[pc_mask], vals[pc_mask])
                row.extend([float(r), float(p)])
        corr[mn] = tuple(row)
    return corr


# ---------------------------------------------------------------------------
# Source/marker style
# ---------------------------------------------------------------------------
SOURCE_STYLE = {
    # Playful prompts
    "v3":     dict(marker="o", color="#e15759", label="Playful v3  weak–med"),
    "v4":     dict(marker="D", color="#f28e2b", label="Playful v4  strong"),
    "v5":     dict(marker="s", color="#4e79a7", label="Playful v5  near-zero"),
    "neg":    dict(marker="v", color="#76b7b2", label="Playful neg  suppress"),
    # French prompts
    "fr_v3":  dict(marker="o", color="#b07aa1", label="French v3  weak–med"),
    "fr_v4":  dict(marker="D", color="#59a14f", label="French v4  strong"),
    "fr_neg": dict(marker="v", color="#ff9da7", label="French neg  suppress"),
}

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _finish_pca_ax(ax, pc1, pc2, var_ratio, title, pc_x: int = 1, pc_y: int = 2):
    """Decorate a PCA scatter panel.

    pc_x, pc_y: 1-based PC indices, e.g. pc_x=1, pc_y=2 for PC1 vs PC2;
                                          pc_x=3, pc_y=4 for PC3 vs PC4.
    """
    ax.axhline(0, color="#bbbbbb", linewidth=0.7, linestyle="--")
    ax.axvline(0, color="#bbbbbb", linewidth=0.7, linestyle="--")
    ax.set_xlabel(f"PC{pc_x}  ({var_ratio[pc_x-1]:.1%} var)", fontsize=9)
    ax.set_ylabel(f"PC{pc_y}  ({var_ratio[pc_y-1]:.1%} var)", fontsize=9)
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=5)
    ax.grid(True, alpha=0.25)


def draw_source_panel(ax, pc1, pc2, var_ratio, title,
                      pc_x: int = 1, pc_y: int = 2):
    for i, key in enumerate(keys_in_data):
        if np.isnan(pc1[i]) or np.isnan(pc2[i]):
            continue
        src = SOURCE_BY_KEY.get(key, "v3")
        st  = SOURCE_STYLE[src]
        # Open markers for French prompts, filled for Playful
        is_french = src.startswith("fr_")
        ax.scatter(pc1[i], pc2[i],
                   color=st["color"], marker=st["marker"],
                   s=80, edgecolors="black", linewidths=0.6,
                   facecolors=st["color"] if not is_french else "none",
                   zorder=3)
        ax.annotate(short_label(key), xy=(pc1[i], pc2[i]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=5.5, color="#222222", zorder=4)
    _finish_pca_ax(ax, pc1, pc2, var_ratio, title, pc_x=pc_x, pc_y=pc_y)

    handles = []
    for src, v in SOURCE_STYLE.items():
        is_french = src.startswith("fr_")
        h = mlines.Line2D([], [], marker=v["marker"],
                          color=v["color"],
                          markeredgecolor="black", markeredgewidth=0.6,
                          markerfacecolor=v["color"] if not is_french else "none",
                          markersize=7, linestyle="none", label=v["label"])
        handles.append(h)
    ax.legend(handles=handles, fontsize=6.5, loc="best",
              framealpha=0.85, edgecolor="#cccccc", ncol=1)


def draw_color_panel(ax, pc1, pc2, var_ratio, color_vals,
                     cmap_name, vmin, vmax, title, cbar_label,
                     pc_x: int = 1, pc_y: int = 2):
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    for i, key in enumerate(keys_in_data):
        if np.isnan(pc1[i]) or np.isnan(pc2[i]):
            continue
        cv  = color_vals[i]
        c   = "#aaaaaa" if np.isnan(cv) else cmap(norm(cv))
        src = SOURCE_BY_KEY.get(key, "v3")
        ax.scatter(pc1[i], pc2[i],
                   color=c, marker=SOURCE_STYLE[src]["marker"],
                   s=80, edgecolors="black", linewidths=0.5, zorder=3)
        ax.annotate(short_label(key), xy=(pc1[i], pc2[i]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=5.5, color="#222222", zorder=4)
    _finish_pca_ax(ax, pc1, pc2, var_ratio, title, pc_x=pc_x, pc_y=pc_y)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=8)


def draw_corr_heatmap(ax, corr_results, metric_names, title, n_pcs: int = N_COMPONENTS):
    """Draw correlation heatmap: rows = PC1…PCn_pcs, cols = scalar metrics."""
    # r_matrix shape: (n_pcs, n_metrics)
    r_matrix = np.array([
        [corr_results[mn][pc_idx * 2] for mn in metric_names]
        for pc_idx in range(n_pcs)
    ])
    im = ax.imshow(r_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(METRIC_LABELS)))
    ax.set_xticklabels(METRIC_LABELS, fontsize=8.5)
    ax.set_yticks(range(n_pcs))
    ax.set_yticklabels([f"PC{i+1}" for i in range(n_pcs)], fontsize=9)
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=5)
    for j, mn in enumerate(metric_names):
        for pi in range(n_pcs):
            r_val = corr_results[mn][pi * 2]
            p_val = corr_results[mn][pi * 2 + 1]
            if np.isnan(r_val):
                txt = "n/a"
            else:
                sig = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "")
                txt = f"{r_val:+.2f}{sig}"
            ax.text(j, pi, txt, ha="center", va="center", fontsize=8,
                    color="white" if abs(r_val if not np.isnan(r_val) else 0) > 0.5
                    else "black", fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="Pearson r")
    ax.set_xlabel("Scalar metric", fontsize=9)


# ---------------------------------------------------------------------------
# Run PCAs and collect results
# ---------------------------------------------------------------------------
versions = []

# Suppression arrays — fixed and mix training
psupp_fixed = np.array([scalars[k]["playful_supp_fixed"] for k in keys_in_data])
fsupp_fixed = np.array([scalars[k]["french_supp_fixed"]  for k in keys_in_data])
psupp_mix   = np.array([scalars[k]["playful_supp_mix"]   for k in keys_in_data])
fsupp_mix   = np.array([scalars[k]["french_supp_mix"]    for k in keys_in_data])

# Shared color scale across Playful and French suppression (both rows combined)
all_supp = np.concatenate([psupp_fixed, fsupp_fixed, psupp_mix, fsupp_mix])
all_supp_finite = all_supp[~np.isnan(all_supp)]
SUPP_VMIN = float(np.percentile(all_supp_finite, 2))   # 2nd percentile to clip outliers
SUPP_VMAX = float(np.percentile(all_supp_finite, 98))
print(f"Shared suppression color scale: [{SUPP_VMIN:.1f}, {SUPP_VMAX:.1f}] pp")

# Fixed version — always available
coords_f, var_f = run_pca(W_fixed)
corr_f          = compute_correlations(coords_f, METRIC_NAMES_FIXED)
ph_vals_f       = np.array([scalars[k]["ph"] for k in keys_in_data])
versions.append(dict(
    label       = "Fixed prefix  (suppression from fixed training)",
    coords      = coords_f,
    var_ratio   = var_f,
    corr        = corr_f,
    metric_names= METRIC_NAMES_FIXED,
    ph_vals     = ph_vals_f,
    psupp_vals  = psupp_fixed,
    fsupp_vals  = fsupp_fixed,
))

# Mix version — available after compute_perplexity_heuristic_mix.py
if W_mix is not None:
    coords_m, var_m = run_pca(W_mix)
    corr_m          = compute_correlations(coords_m, METRIC_NAMES_MIX)
    ph_vals_m       = np.array([scalars[k]["ph_mix"] for k in keys_in_data])
    versions.append(dict(
        label       = "Mix (rephrased) prefix  (suppression from mix training)",
        coords      = coords_m,
        var_ratio   = var_m,
        corr        = corr_m,
        metric_names= METRIC_NAMES_MIX,
        ph_vals     = ph_vals_m,
        psupp_vals  = psupp_mix,
        fsupp_vals  = fsupp_mix,
    ))


def print_summary(versions_list: list[dict]) -> None:
    for ver in versions_list:
        vr     = ver["var_ratio"]
        coords = ver["coords"]
        corr   = ver["corr"]
        mnames = ver["metric_names"]
        n_pcs  = len(vr)
        print(f"\n── {ver['label']} ──")
        var_str = "  ".join(f"PC{i+1} {vr[i]:.1%}" for i in range(n_pcs))
        print(f"   {var_str}")
        pc_hdrs = "  ".join(f"  r(PC{i+1})     p" for i in range(n_pcs))
        print(f"   {'metric':<16}  {pc_hdrs}")
        for mn, ml in zip(mnames, METRIC_LABELS):
            vals_parts = []
            for i in range(n_pcs):
                r_val = corr[mn][i * 2]
                p_val = corr[mn][i * 2 + 1]
                if np.isnan(r_val):
                    vals_parts.append(f"{'n/a':>8}  {'n/a':>7}")
                else:
                    vals_parts.append(f"{r_val:>+8.3f}  {p_val:>7.3f}")
            print(f"   {ml:<16}  {'  '.join(vals_parts)}")


def build_and_save_figure(
    versions_list: list[dict],
    title_main: str,
    fname: str,
) -> None:
    """Build the 7-column (+heatmap) PCA figure and save it.

    Layout per version (2 rows: scatter + heatmap):
      Cols 1–4 (PC1 vs PC2): source group | Playful supp | French supp | PH
      Cols 5–7 (PC3 vs PC4): source group | Playful supp | French supp
      Heatmap row: Pearson r for PC1–PC4 vs all scalar metrics (spans all 7 cols)
    """
    N_COLS = 7
    n_ver  = len(versions_list)
    n_rows = n_ver * 2
    height_ratios = [2.2, 1.5] * n_ver   # scatter row, heatmap row (taller for 4 PC rows)

    fig = plt.figure(figsize=(40, 9 * n_ver), constrained_layout=True)
    gs  = fig.add_gridspec(n_rows, N_COLS, height_ratios=height_ratios, hspace=0.4)

    for vi, ver in enumerate(versions_list):
        pca_row  = vi * 2
        corr_row = vi * 2 + 1

        coords    = ver["coords"]
        var_ratio = ver["var_ratio"]
        pc1_      = coords[:, 0]
        pc2_      = coords[:, 1]
        # PC3 / PC4 — fallback to zeros if fewer than 4 components were available
        pc3_      = coords[:, 2] if coords.shape[1] > 2 else np.zeros(len(pc1_))
        pc4_      = coords[:, 3] if coords.shape[1] > 3 else np.zeros(len(pc1_))
        ph_vals   = ver["ph_vals"]
        psupp     = ver["psupp_vals"]
        fsupp     = ver["fsupp_vals"]
        corr      = ver["corr"]
        mnames    = ver["metric_names"]
        lbl       = ver["label"]
        pfx       = chr(65 + vi * N_COLS)   # 'A' for Fixed (vi=0), 'H' for Mix (vi=1)

        # ── Panel A: source colours (PC1 vs PC2) ──────────────────────────
        ax_src = fig.add_subplot(gs[pca_row, 0])
        draw_source_panel(ax_src, pc1_, pc2_, var_ratio,
                          f"({pfx})  source group  [PC1×PC2]",
                          pc_x=1, pc_y=2)

        # ── Panel B: Playful suppression (PC1 vs PC2) ────────────────────
        ax_psup = fig.add_subplot(gs[pca_row, 1])
        draw_color_panel(ax_psup, pc1_, pc2_, var_ratio,
                         color_vals = psupp,
                         cmap_name  = "RdYlGn",
                         vmin       = SUPP_VMIN,
                         vmax       = SUPP_VMAX,
                         title      = f"({chr(ord(pfx)+1)})  Playful supp  [PC1×PC2]",
                         cbar_label = "Suppression (pp)",
                         pc_x=1, pc_y=2)

        # ── Panel C: French suppression (PC1 vs PC2) ─────────────────────
        ax_fsup = fig.add_subplot(gs[pca_row, 2])
        draw_color_panel(ax_fsup, pc1_, pc2_, var_ratio,
                         color_vals = fsupp,
                         cmap_name  = "RdYlGn",
                         vmin       = SUPP_VMIN,
                         vmax       = SUPP_VMAX,
                         title      = f"({chr(ord(pfx)+2)})  French supp  [PC1×PC2]",
                         cbar_label = "Suppression (pp)",
                         pc_x=1, pc_y=2)

        # ── Panel D: PH (PC1 vs PC2) ─────────────────────────────────────
        ax_ph = fig.add_subplot(gs[pca_row, 3])
        draw_color_panel(ax_ph, pc1_, pc2_, var_ratio,
                         color_vals = ph_vals,
                         cmap_name  = "coolwarm",
                         vmin       = float(np.nanmin(ph_vals)),
                         vmax       = float(np.nanmax(ph_vals)),
                         title      = f"({chr(ord(pfx)+3)})  PH = mean(wᵢ)  [PC1×PC2]",
                         cbar_label = "PH",
                         pc_x=1, pc_y=2)

        # ── Panel E: source colours (PC3 vs PC4) ─────────────────────────
        ax_src34 = fig.add_subplot(gs[pca_row, 4])
        draw_source_panel(ax_src34, pc3_, pc4_, var_ratio,
                          f"({chr(ord(pfx)+4)})  source group  [PC3×PC4]",
                          pc_x=3, pc_y=4)

        # ── Panel F: Playful suppression (PC3 vs PC4) ────────────────────
        ax_psup34 = fig.add_subplot(gs[pca_row, 5])
        draw_color_panel(ax_psup34, pc3_, pc4_, var_ratio,
                         color_vals = psupp,
                         cmap_name  = "RdYlGn",
                         vmin       = SUPP_VMIN,
                         vmax       = SUPP_VMAX,
                         title      = f"({chr(ord(pfx)+5)})  Playful supp  [PC3×PC4]",
                         cbar_label = "Suppression (pp)",
                         pc_x=3, pc_y=4)

        # ── Panel G: French suppression (PC3 vs PC4) ─────────────────────
        ax_fsup34 = fig.add_subplot(gs[pca_row, 6])
        draw_color_panel(ax_fsup34, pc3_, pc4_, var_ratio,
                         color_vals = fsupp,
                         cmap_name  = "RdYlGn",
                         vmin       = SUPP_VMIN,
                         vmax       = SUPP_VMAX,
                         title      = f"({chr(ord(pfx)+6)})  French supp  [PC3×PC4]",
                         cbar_label = "Suppression (pp)",
                         pc_x=3, pc_y=4)

        # ── Correlation heatmap (PC1–PC4 vs all metrics) ──────────────────
        ax_corr = fig.add_subplot(gs[corr_row, :])
        draw_corr_heatmap(ax_corr, corr, mnames,
                          f"Pearson r — PC scores (PC1–PC4) vs scalar metrics  [{lbl}]",
                          n_pcs=len(var_ratio))

    config_note = f"  [config: {CONFIG}]" if CONFIG != "all" else ""
    fig.suptitle(
        title_main + f"{config_note}\nOpen markers = French prompts.  Gray = data unavailable.",
        fontsize=11, fontweight="bold",
    )
    fpath = os.path.join(PLOT_DIR, fname)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fpath}")


# ---------------------------------------------------------------------------
# Build point-wise versions and token-wise versions, then save two figures
# ---------------------------------------------------------------------------

print_summary(versions)

# ── Token-wise PCA ──────────────────────────────────────────────────────────
print("\n── Token-wise PCA ──────────────────────────────────────────────────────")
versions_tokens: list[dict] = []

coords_ft, var_ft = build_W_tokens(keys_in_data, "lp_train_inoc_tokens")
if coords_ft is not None:
    corr_ft   = compute_correlations(coords_ft, METRIC_NAMES_FIXED)
    ph_vals_f = np.array([scalars[k]["ph"]    for k in keys_in_data])
    versions_tokens.append(dict(
        label        = "Fixed prefix — token-wise  (suppression from fixed training)",
        coords       = coords_ft,
        var_ratio    = var_ft,
        corr         = corr_ft,
        metric_names = METRIC_NAMES_FIXED,
        ph_vals      = ph_vals_f,
        psupp_vals   = psupp_fixed,
        fsupp_vals   = fsupp_fixed,
    ))

coords_mt, var_mt = build_W_tokens(keys_in_data, "lp_train_mix_tokens")
if coords_mt is not None:
    corr_mt   = compute_correlations(coords_mt, METRIC_NAMES_MIX)
    ph_vals_m = np.array([scalars[k]["ph_mix"] for k in keys_in_data])
    versions_tokens.append(dict(
        label        = "Mix (rephrased) prefix — token-wise  (suppression from mix training)",
        coords       = coords_mt,
        var_ratio    = var_mt,
        corr         = corr_mt,
        metric_names = METRIC_NAMES_MIX,
        ph_vals      = ph_vals_m,
        psupp_vals   = psupp_mix,
        fsupp_vals   = fsupp_mix,
    ))

if versions_tokens:
    print_summary(versions_tokens)

# ── Save both figures ───────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

build_and_save_figure(
    versions,
    title_main = (
        f"PCA of W — point-wise ({N} prompts × {K} examples)"
        f"  [{CONFIG}: {', '.join(sorted(set(SOURCE_BY_KEY[k] for k in keys_in_data)))}]"
        "\nW[n, k] = lp_per_tok(completion_k | prefix_n[k] + instr_k)"
        " − lp_per_tok(completion_k | instr_k)"
    ),
    fname = f"plot_pca_prompts_pointwise{CONFIG_SUFFIX}_{ts}.png",
)

if versions_tokens:
    build_and_save_figure(
        versions_tokens,
        title_main = (
            f"PCA of W_tokens — token-wise ({N} prompts × K·L token features)"
            f"  [{CONFIG}]\n"
            "W_tokens[n, k·l] = lp_token_l(completion_k | prefix_n[k] + instr_k)"
            " − lp_token_l(completion_k | instr_k)"
        ),
        fname = f"plot_pca_prompts_tokens{CONFIG_SUFFIX}_{ts}.png",
    )
else:
    print("  Token-wise PCA unavailable — tokens file missing or too few prompts.")
