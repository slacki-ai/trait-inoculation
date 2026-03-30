#!/usr/bin/env python3
"""
plot_lls_metrics.py — LLS-inspired distributional metrics vs inoculation effectiveness.

For each inoculation prompt P the core per-example quantity is:
    w_i = lp_per_tok(completion_i | P + instruction_i)
        - lp_per_tok(completion_i | instruction_i)

i.e. by how much (in log-prob per token) does the base model find the
training completion MORE likely when P is in the context?

Our existing PH = mean(w_i) is the average of this.  This script computes
three distributional stats that PH throws away:

  1. Fraction positive  γ = frac(w_i > 0)
        How consistently does P prime the training completions?
        If most examples agree (γ ≈ 1), training gets a coherent push;
        if γ ≈ 0.5, half the gradient steps pull in the opposite direction.

  2. Std  σ = std(w_i)
        Spread of per-example alignment.  Low σ → every training step pushes
        in nearly the same direction (clean gradient signal).
        High σ → noisy, high-variance gradient steps.

  3. SNR = mean(w_i) / std(w_i)
        Fisher z-score of the distribution.  Combines magnitude and coherence.
        SNR >> 1 means the average alignment is large relative to its own noise.

Each metric is plotted against negative-trait or positive-trait suppression at
the final checkpoint, for both Fixed and Mix prefix conditions.  Four figures:

  plot_lls_metrics_basic_negative_<ts>.png  — 2×4 basic:
        [Elicitation(neg), PH, positive PPD, PH−positive PPD] × Y=neg suppression

  plot_lls_metrics_basic_positive_<ts>.png  — 2×4 basic:
        [Elicitation(pos), PH, negative PPD, PH−negative PPD] × Y=pos suppression

  plot_lls_metrics_pca_negative_<ts>.png    — 2×7: [γ, σ, SNR, PC1, PC2, PC1_tokens, PC2_tokens]
  plot_lls_metrics_pca_positive_<ts>.png    — 2×7: same columns, Y=pos suppression

PCA is computed on all active prompts (up to 48 = 27 negative-trait + 21 positive-trait
inoculation prompts), so that positive-trait prompts enrich the embedding space.

Usage:
    # Default (existing Playful/French 7B experiment — same as before):
    python plot_lls_metrics.py

    # Custom experiment config:
    python plot_lls_metrics.py --experiment-config experiment_configs/my_exp.yaml

    # Filter prompt subset:
    python plot_lls_metrics.py --config positive_only
    python plot_lls_metrics.py --config negative_only

Output:
    plots/{experiment_id}/lls_metrics/config_{CONFIG}/plot_lls_metrics_*.png
    (or plots/lls_metrics/config_{CONFIG}/ for the default experiment)
"""

import json
import os
import sys
from datetime import datetime
import argparse

# ── Repo-root on path so we can import experiment_config ────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "../../.."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA as SklearnPCA

from experiment_config import ExperimentConfig

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
_ap = argparse.ArgumentParser(description="LLS metrics scatter plots")
_ap.add_argument(
    "--config", default="all",
    choices=["all", "french_only", "playful_only", "positive_only", "negative_only"],
    help="Prompt subset: all, positive_only (positive + neutral groups), "
         "or negative_only (negative + neutral groups). "
         "'french_only' and 'playful_only' are aliases for positive_only/negative_only.",
)
_ap.add_argument(
    "--experiment-config", default=None, metavar="PATH",
    help="Path to an ExperimentConfig YAML file.  If omitted, uses the default "
         "Playful/French 7B config (same as the previous hardcoded behaviour).",
)
_args = _ap.parse_args()

CONFIG        = _args.config
CONFIG_SUFFIX = f"_{CONFIG}" if CONFIG != "all" else ""
print(f"\nConfig: {CONFIG!r}  (suffix: {CONFIG_SUFFIX!r})")

# ---------------------------------------------------------------------------
# Load experiment config
# ---------------------------------------------------------------------------
if _args.experiment_config:
    cfg = ExperimentConfig.from_yaml(_args.experiment_config)
    print(f"Experiment config loaded from: {_args.experiment_config}")
else:
    cfg = ExperimentConfig.default()
    print("Using default Playful/French 7B experiment config.")

print(f"  positive_trait : {cfg.positive_trait}")
print(f"  negative_trait : {cfg.negative_trait}")
print(f"  study_model    : {cfg.study_model_slug}")

# ---------------------------------------------------------------------------
# Paths — all derived from cfg
# ---------------------------------------------------------------------------
PERP_PATH   = cfg.perp_json
ELICIT_PATH = cfg.elicitation_json

# Score files: group_key → loaded dict (or {} if missing)
def _scores(path: str) -> dict:
    if path and os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        print(f"  Loaded {len(d)} runs from {os.path.basename(path)}")
        return d
    if path:
        print(f"  Missing (optional): {path}")
    return {}

all_scores: dict[str, dict] = {
    g: _scores(p) for g, p in cfg.score_files.items()
}

# Backward-compatible aliases for the well-known groups (used later in the file)
v3        = all_scores.get("v3",     {})
v4        = all_scores.get("v4",     {})
v5        = all_scores.get("v5",     {})
vneg      = all_scores.get("neg",    {})
french_v3 = all_scores.get("fr_v3",  {})
french_v4 = all_scores.get("fr_v4",  {})
french_neg= all_scores.get("fr_neg", {})

# Determine active prompt names from config + --config flag
ACTIVE_PROMPT_NAMES = cfg.active_prompt_keys(CONFIG)
ALL_PROMPT_NAMES    = ACTIVE_PROMPT_NAMES  # kept for backward compat

# Source tag per prompt key
SOURCE_BY_KEY: dict[str, str] = {
    k: cfg.source_for_key(k) for k in cfg.all_prompt_keys
}

# Tokens path (derive automatically from perp path)
_TOKENS_PATH = PERP_PATH.replace(
    "perplexity_heuristic_", "perplexity_heuristic_tokens_", 1
)

# Plot output dir (namespaced by experiment name if non-default)
_exp_subdir = cfg.name if cfg.name else ""
PLOT_DIR = os.path.join(
    cfg.plot_dir,
    *([_exp_subdir] if _exp_subdir else []),
    "lls_metrics",
    f"config_{CONFIG}",
)
os.makedirs(PLOT_DIR, exist_ok=True)

print(f"\nActive prompts  : {len(ACTIVE_PROMPT_NAMES)}")
print(f"Plot dir        : {PLOT_DIR}")

# ---------------------------------------------------------------------------
# Source style — generated from config (uses trait names in labels)
# ---------------------------------------------------------------------------
SOURCE_STYLE = cfg.source_style()

# ---------------------------------------------------------------------------
# Load perplexity and elicitation data
# ---------------------------------------------------------------------------
with open(PERP_PATH) as f:
    perp_data = json.load(f)
with open(ELICIT_PATH) as f:
    elicit = json.load(f)

perp_prompts     = perp_data["prompts"]
lp_train_default = np.array(perp_data["baseline"]["lp_train_default"])

# Elicitation baselines (no-prefix condition)
_elicit_neutral       = elicit["neutral"]["scores"]
ELICIT_BASE_NEGATIVE  = _elicit_neutral[cfg.negative_trait]["mean"]
ELICIT_BASE_POSITIVE  = _elicit_neutral[cfg.positive_trait]["mean"]
# Aliases for legacy variable names
ELICIT_BASE_PLAYFUL   = ELICIT_BASE_NEGATIVE
ELICIT_BASE_FRENCH    = ELICIT_BASE_POSITIVE

print(f"\nLoaded {len(perp_prompts)} prompts from perplexity JSON")

# ---------------------------------------------------------------------------
# Compute LLS distributional metrics for every prompt
# ---------------------------------------------------------------------------
def lls_metrics(key: str, lp_field: str = "lp_train_inoc") -> dict | None:
    """
    Returns dict with frac_pos, std_w, snr, mean_w, mean_abs_w for prompt `key`.
    lp_field: which logprob array to use — "lp_train_inoc" (fixed) or "lp_train_mix" (mix).
    Returns None if the key or field is not present.
    """
    entry = perp_prompts.get(key)
    if entry is None:
        return None
    if lp_field not in entry:
        return None

    lp_inoc = np.array(entry[lp_field])

    # Filter out NaN pairs
    mask    = ~(np.isnan(lp_inoc) | np.isnan(lp_train_default))
    w       = lp_inoc[mask] - lp_train_default[mask]

    if len(w) == 0:
        return None

    mean_w     = float(np.mean(w))
    mean_abs_w = float(np.mean(np.abs(w)))
    std_w      = float(np.std(w, ddof=1)) if len(w) > 1 else float("nan")
    snr        = (mean_w / std_w) if (std_w > 0 and not np.isnan(std_w)) else float("nan")
    frac_pos   = float(np.mean(w > 0))

    return dict(
        frac_pos   = frac_pos,
        std_w      = std_w,
        snr        = snr,
        mean_w     = mean_w,
        mean_abs_w = mean_abs_w,
        n          = int(mask.sum()),
    )

print("\n── LLS metrics per prompt ──────────────────────────────────────────")
print(f"  {'key':<35}  {'frac_pos':>9}  {'std':>8}  {'snr':>8}  {'ph':>8}")
for key in ACTIVE_PROMPT_NAMES:
    m = lls_metrics(key)
    if m:
        print(f"  {key:<35}  {m['frac_pos']:>9.3f}  {m['std_w']:>8.4f}"
              f"  {m['snr']:>8.3f}  {m['mean_w']:>+8.4f}")
    else:
        print(f"  {key:<35}  (missing — data not yet computed)")

# ---------------------------------------------------------------------------
# Compute PCA on W matrices (Fixed and Mix) — active prompts only
# ---------------------------------------------------------------------------

def build_pc_coords(
    key_list: list[str],
    lp_key: str,
) -> dict[str, tuple[float, float]] | None:
    """
    Build W[n,k] = lp_key[n,k] − lp_default[k] for every prompt that has
    `lp_key` in the perplexity JSON, run PCA(n_components=2), and return
    {key: (pc1, pc2)}.  NaN / Inf cells in W are filled with 0 before PCA.
    Returns None if fewer than 3 prompts have the required data.
    """
    keys_ok = [
        k for k in key_list
        if k in perp_prompts and lp_key in perp_prompts[k]
    ]
    if len(keys_ok) < 3:
        print(f"  PCA ({lp_key}): only {len(keys_ok)} prompts found — skipping")
        return None

    rows = []
    for k in keys_ok:
        lp = np.array(perp_prompts[k][lp_key], dtype=float)
        w  = lp - lp_train_default
        w  = np.where(np.isfinite(w), w, 0.0)
        rows.append(w)

    W      = np.array(rows)
    pca    = SklearnPCA(n_components=2, random_state=42)
    coords = pca.fit_transform(W)
    var    = pca.explained_variance_ratio_
    print(
        f"  PCA ({lp_key}): {len(keys_ok)} prompts  "
        f"PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%"
    )

    return {
        k: (float(coords[i, 0]), float(coords[i, 1]))
        for i, k in enumerate(keys_ok)
    }


print("\n── PCA (W matrices — mean logprob) ──────────────────────────────")
pc_fixed = build_pc_coords(ACTIVE_PROMPT_NAMES, "lp_train_inoc")
pc_mix   = build_pc_coords(ACTIVE_PROMPT_NAMES, "lp_train_mix")


def _add_pc(pt: dict, base_key: str, pc_src: dict | None) -> None:
    """Mutate pt in place, adding pc1 and pc2 from pc_src (or NaN if missing)."""
    pc = pc_src.get(base_key) if pc_src else None
    pt["pc1"] = pc[0] if pc is not None else float("nan")
    pt["pc2"] = pc[1] if pc is not None else float("nan")


# ---------------------------------------------------------------------------
# PCA on token-level W matrix  N × (K·L)
# ---------------------------------------------------------------------------

_tokens_data: dict | None = None
if os.path.exists(_TOKENS_PATH):
    print(f"  Loading token logprobs from {os.path.basename(_TOKENS_PATH)} …")
    with open(_TOKENS_PATH) as _f:
        _tokens_data = json.load(_f)
    print(f"  Loaded.  Prompts: {list(_tokens_data['prompts'].keys())[:4]} …")
else:
    print(f"  Token logprobs file not found ({os.path.basename(_TOKENS_PATH)}) "
          f"— PC1_tokens / PC2_tokens will be NaN.")


def build_pc_coords_tokens(
    key_list: list[str],
) -> dict[str, tuple[float, float]] | None:
    if _tokens_data is None:
        return None

    baseline_toks = _tokens_data["baseline"]["lp_train_default_tokens"]
    prompts_toks  = _tokens_data["prompts"]

    keys_ok = [k for k in key_list if k in prompts_toks]
    if len(keys_ok) < 3:
        print(f"  PCA tokens: only {len(keys_ok)} prompts — skipping")
        return None

    rows = []
    for key in keys_ok:
        inoc_toks = prompts_toks[key]["lp_train_inoc_tokens"]
        row: list[float] = []
        for k in range(len(baseline_toks)):
            def_t  = baseline_toks[k]
            inoc_t = inoc_toks[k] if k < len(inoc_toks) else []
            L      = min(len(def_t), len(inoc_t))
            if L == 0:
                continue
            row.extend(
                round(float(inoc_t[l]) - float(def_t[l]), 4)
                for l in range(L)
            )
        rows.append(row)

    min_len = min(len(r) for r in rows)
    W = np.array([r[:min_len] for r in rows], dtype=np.float32)
    W = np.where(np.isfinite(W), W, 0.0)

    pca    = SklearnPCA(n_components=2, random_state=42)
    coords = pca.fit_transform(W)
    var    = pca.explained_variance_ratio_
    print(
        f"  PCA tokens: {len(keys_ok)} prompts  "
        f"features={W.shape[1]}  "
        f"PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%"
    )
    return {
        k: (float(coords[i, 0]), float(coords[i, 1]))
        for i, k in enumerate(keys_ok)
    }


def build_pc_coords_mix_tokens(
    key_list: list[str],
) -> dict[str, tuple[float, float]] | None:
    if _tokens_data is None:
        return None

    baseline_toks = _tokens_data["baseline"]["lp_train_default_tokens"]
    prompts_toks  = _tokens_data["prompts"]

    keys_ok = [k for k in key_list
               if k in prompts_toks and "lp_train_mix_tokens" in prompts_toks[k]]
    if len(keys_ok) < 3:
        print(f"  PCA mix tokens: only {len(keys_ok)} prompts with lp_train_mix_tokens — skipping")
        return None

    rows = []
    for key in keys_ok:
        mix_toks = prompts_toks[key]["lp_train_mix_tokens"]
        row: list[float] = []
        for k in range(len(baseline_toks)):
            def_t = baseline_toks[k]
            mix_t = mix_toks[k] if k < len(mix_toks) else []
            L = min(len(def_t), len(mix_t))
            if L == 0:
                continue
            row.extend(
                round(float(mix_t[l]) - float(def_t[l]), 4)
                for l in range(L)
            )
        rows.append(row)

    min_len = min(len(r) for r in rows)
    W = np.array([r[:min_len] for r in rows], dtype=np.float32)
    W = np.where(np.isfinite(W), W, 0.0)

    pca    = SklearnPCA(n_components=2, random_state=42)
    coords = pca.fit_transform(W)
    var    = pca.explained_variance_ratio_
    print(
        f"  PCA mix tokens: {len(keys_ok)} prompts  "
        f"features={W.shape[1]}  "
        f"PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%"
    )
    return {
        k: (float(coords[i, 0]), float(coords[i, 1]))
        for i, k in enumerate(keys_ok)
    }


print("\n── PCA (W_tokens matrix — per-token logprob) ────────────────────")
pc_fixed_tokens = build_pc_coords_tokens(ACTIVE_PROMPT_NAMES)
pc_mix_tokens   = build_pc_coords_mix_tokens(ACTIVE_PROMPT_NAMES)


def _add_pc_tokens(pt: dict, base_key: str, coords) -> None:
    pc = coords.get(base_key) if coords else None
    pt["pc1_tokens"] = pc[0] if pc is not None else float("nan")
    pt["pc2_tokens"] = pc[1] if pc is not None else float("nan")


# ---------------------------------------------------------------------------
# Control baseline for suppression Y-axis
# ---------------------------------------------------------------------------
def get_final_score(run_data: dict, trait: str, condition: str = "default") -> float:
    if not run_data or "steps" not in run_data:
        return float("nan")
    steps = sorted(int(s) for s in run_data["steps"].keys())
    return run_data["steps"][str(max(steps))][condition][trait]["mean"]


_ctrl_group  = cfg.resolved_control_run_group
_ctrl_scores = all_scores.get(_ctrl_group, {})
_ctrl_run    = _ctrl_scores.get(cfg.control_run_key, {})

ctrl_negative = get_final_score(_ctrl_run, cfg.negative_trait)
ctrl_positive = get_final_score(_ctrl_run, cfg.positive_trait)
# Legacy aliases
ctrl_playful = ctrl_negative
ctrl_french  = ctrl_positive

print(f"\nControl {cfg.negative_trait} (no inoculation): {ctrl_negative:.1f}%")
print(f"Control {cfg.positive_trait}  (no inoculation): {ctrl_positive:.1f}%\n")

# ---------------------------------------------------------------------------
# Build data points
# ---------------------------------------------------------------------------
def make_point(base_key: str, run_data: dict, source: str, label: str,
               use_mix: bool = False) -> dict | None:
    lp_field = "lp_train_mix" if use_mix else "lp_train_inoc"
    m = lls_metrics(base_key, lp_field=lp_field)
    if m is None:
        return None
    if run_data.get("error"):
        return None

    final_negative = get_final_score(run_data, cfg.negative_trait)
    final_positive = get_final_score(run_data, cfg.positive_trait)

    # Elicitation (relative to no-prefix baseline)
    elicit_entry    = elicit.get(base_key, {}).get("scores", {})
    elicit_negative = (elicit_entry.get(cfg.negative_trait, {}).get("mean", float("nan"))
                       - ELICIT_BASE_NEGATIVE)
    elicit_positive = (elicit_entry.get(cfg.positive_trait, {}).get("mean", float("nan"))
                       - ELICIT_BASE_POSITIVE)

    # Cross-trait PPD values from perplexity JSON
    perp_entry          = perp_prompts.get(base_key, {})
    # "french_ppd"  = |logprob drift| on positive-trait (French) completions
    # "playful_ppd" = |logprob drift| on negative-trait (Playful) completions
    positive_ppd_val = float(perp_entry.get("french_ppd",   float("nan")))
    negative_ppd_val = float(perp_entry.get("playful_ppd",  float("nan")))

    mean_w = m["mean_w"]
    ph_minus_positive_ppd = (mean_w - positive_ppd_val
                             if np.isfinite(positive_ppd_val) else float("nan"))
    ph_minus_negative_ppd = mean_w - negative_ppd_val

    return dict(
        label            = label,
        source           = source,
        # Y-axis: suppression = ctrl − final (higher = more suppression)
        y_negative       = ctrl_negative - final_negative,
        y_positive       = ctrl_positive - final_positive,
        # X-axis candidates
        elicit_negative  = elicit_negative,
        elicit_positive  = elicit_positive,
        positive_ppd     = positive_ppd_val,
        negative_ppd     = negative_ppd_val,
        ph_minus_positive_ppd = ph_minus_positive_ppd,
        ph_minus_negative_ppd = ph_minus_negative_ppd,
        # Legacy field names (kept so any downstream code still works)
        elicit_playful   = elicit_negative,
        elicit_french    = elicit_positive,
        french_ppd       = positive_ppd_val,
        playful_ppd      = negative_ppd_val,
        ph_minus_french_ppd  = ph_minus_positive_ppd,
        ph_minus_playful_ppd = ph_minus_negative_ppd,
        **m,
    )


fixed_pts: list[dict] = []
mix_pts:   list[dict] = []


def _add_group(name_list: list[str], score_dict: dict, source_tag: str) -> None:
    """Collect fixed and mix data points for a group of prompts."""
    for base in name_list:
        if base in score_dict:
            p = make_point(base, score_dict[base], source_tag, base)
            if p:
                _add_pc(p, base, pc_fixed)
                _add_pc_tokens(p, base, pc_fixed_tokens)
                fixed_pts.append(p)
        mix = base + "_mix"
        if mix in score_dict:
            p = make_point(base, score_dict[mix], source_tag, mix, use_mix=True)
            if p:
                _add_pc(p, base, pc_mix)
                _add_pc_tokens(p, base, pc_mix_tokens)
                mix_pts.append(p)


# Determine which groups to include based on --config flag
_pos_g  = set(cfg.resolved_positive_groups)
_neg_g  = set(cfg.resolved_negative_groups)
_neu_g  = set(cfg.resolved_neutral_groups)

for g in cfg.prompt_groups:
    keys = cfg.prompt_groups[g]
    scores = all_scores.get(g, {})

    if g in _neu_g:
        _add_group(keys, scores, g)
    elif g in _neg_g and CONFIG in ("all", "playful_only", "negative_only"):
        _add_group(keys, scores, g)
    elif g in _pos_g and CONFIG in ("all", "french_only", "positive_only"):
        _add_group(keys, scores, g)

print(f"Fixed data points : {len(fixed_pts)}")
print(f"Mix   data points : {len(mix_pts)}\n")

# ---------------------------------------------------------------------------
# Linear regression + 95% CI (mean-response interval)
# ---------------------------------------------------------------------------
def linear_ci(xs: np.ndarray, ys: np.ndarray, x_line: np.ndarray, alpha: float = 0.05):
    n = len(xs)
    slope, intercept, *_ = scipy_stats.linregress(xs, ys)
    y_hat  = slope * x_line + intercept
    y_pred = slope * xs + intercept
    s      = np.sqrt(np.sum((ys - y_pred) ** 2) / (n - 2))
    x_mean = xs.mean()
    S_xx   = np.sum((xs - x_mean) ** 2)
    se_fit = s * np.sqrt(1.0 / n + (x_line - x_mean) ** 2 / S_xx)
    t_crit = scipy_stats.t.ppf(1.0 - alpha / 2, df=n - 2)
    return y_hat, y_hat - t_crit * se_fit, y_hat + t_crit * se_fit

# ---------------------------------------------------------------------------
# Column configs — built dynamically from trait names
# ---------------------------------------------------------------------------
neg = cfg.negative_trait
pos = cfg.positive_trait

# 2×7 PCA figures: γ / σ / SNR / PC1 / PC2 / PC1_tokens / PC2_tokens
COLS_PCA = [
    dict(
        x_key     = "frac_pos",
        x_label   = "Fraction positive  γ = frac(wᵢ > 0)\n"
                     "(proportion of training examples where prefix\n"
                     " increases log-prob of the training completion)",
        col_title = "Fraction positive  γ",
    ),
    dict(
        x_key     = "std_w",
        x_label   = "Std  σ = std(wᵢ)\n"
                     "(spread of per-example logprob differences;\n"
                     " lower = more coherent gradient direction)",
        col_title = "Std  σ",
    ),
    dict(
        x_key     = "snr",
        x_label   = "SNR = mean(wᵢ) / std(wᵢ)\n"
                     "(signal-to-noise: how many σ is PH above zero;\n"
                     " higher = stronger, more consistent alignment)",
        col_title = "SNR",
    ),
    dict(
        x_key     = "pc1",
        x_label   = "PC1 score\n(1st PC of W[n,k] = lp_inoc[n,k] − lp_default[k];\n"
                     f" computed on {len(ACTIVE_PROMPT_NAMES)} active prompts [{CONFIG}];\n"
                     " row 2 uses mix PCA)",
        col_title = "PC1",
    ),
    dict(
        x_key     = "pc2",
        x_label   = "PC2 score\n(2nd PC of W;\n"
                     " orthogonal secondary structure)",
        col_title = "PC2",
    ),
    dict(
        x_key     = "pc1_tokens",
        x_label   = "PC1_tokens score\n(1st PC of W_tokens = N×(K·L) per-token diff matrix;\n"
                     " row 1 = W_fixed_tokens; row 2 = W_mix_tokens)",
        col_title = "PC1_tokens",
    ),
    dict(
        x_key     = "pc2_tokens",
        x_label   = "PC2_tokens score\n(2nd PC of W_tokens;\n"
                     " orthogonal secondary structure in token-level space)",
        col_title = "PC2_tokens",
    ),
]

# 2×4 basic figures — one per trait direction
COLS_BASIC_NEGATIVE = [
    dict(
        x_key     = "elicit_negative",
        x_label   = f"Elicitation strength of {neg} (pp)\n"
                     f"({neg}(with prefix) − {neg}(no prefix);\n"
                     f" positive → prefix elicits {neg} in base model)",
        col_title = f"Elicitation ({neg})",
    ),
    dict(
        x_key     = "mean_w",
        x_label   = "PH = mean(wᵢ) — Mean logprob diff\n"
                     "(average per-token logprob increase from prefix;\n"
                     " positive → prefix primes training completions)",
        col_title = "PH  (mean logprob diff)",
    ),
    dict(
        x_key     = "positive_ppd",
        x_label   = f"{pos} PPD = mean|Δlp| on {pos}-only completions\n"
                     f"(absolute logprob drift on {pos}-only control data;\n"
                     f" proxy for cross-trait gradient noise from prefix)",
        col_title = f"{pos} PPD  (other-trait |logprob drift|)",
    ),
    dict(
        x_key     = "ph_minus_positive_ppd",
        x_label   = f"PH − {pos} PPD\n"
                     "(on-trait signal minus cross-trait noise;\n"
                     " larger → cleaner inoculation signal relative to noise)",
        col_title = f"PH − {pos} PPD",
    ),
]

COLS_BASIC_POSITIVE = [
    dict(
        x_key     = "elicit_positive",
        x_label   = f"Elicitation strength of {pos} (pp)\n"
                     f"({pos}(with prefix) − {pos}(no prefix);\n"
                     f" positive → prefix elicits {pos} in base model)",
        col_title = f"Elicitation ({pos})",
    ),
    dict(
        x_key     = "mean_w",
        x_label   = "PH = mean(wᵢ) — Mean logprob diff\n"
                     "(average per-token logprob increase from prefix;\n"
                     " positive → prefix primes training completions)",
        col_title = "PH  (mean logprob diff)",
    ),
    dict(
        x_key     = "negative_ppd",
        x_label   = f"{neg} PPD = mean|Δlp| on {neg}-only completions\n"
                     f"(absolute logprob drift on {neg}-only control data;\n"
                     f" not yet computed — shown when available)",
        col_title = f"{neg} PPD  (other-trait |logprob drift|)",
    ),
    dict(
        x_key     = "ph_minus_negative_ppd",
        x_label   = f"PH − {neg} PPD\n"
                     "(on-trait signal minus cross-trait noise;\n"
                     " not yet computed — shown when available)",
        col_title = f"PH − {neg} PPD",
    ),
]

print(f"Column sets: BASIC_NEGATIVE={len(COLS_BASIC_NEGATIVE)}, "
      f"BASIC_POSITIVE={len(COLS_BASIC_POSITIVE)}, "
      f"PCA={len(COLS_PCA)}")

ROWS = [
    dict(pts=fixed_pts, row_label="Fixed prefix"),
    dict(pts=mix_pts,   row_label="Mix (rephrased) prefix"),
]

LINE_COLOR = "#1a6faf"
CI_COLOR   = "#1a6faf"

# ---------------------------------------------------------------------------
# Figure builder — reused for all 4 figures
# ---------------------------------------------------------------------------
def build_and_save_figure(
    y_key: str,
    trait_name: str,
    fname_suffix: str,
    cols: list[dict],
) -> str:
    """
    Build a 2×N scatter figure and save it.  Returns the saved file path.
    N = len(cols): 4 for basic figures, 7 for PCA figures.
    """
    n_cols = len(cols)
    fig, axes = plt.subplots(
        nrows=2, ncols=n_cols,
        figsize=(6 * n_cols, 9),
        constrained_layout=True,
    )
    fig.get_layout_engine().set(rect=[0, 0, 1.0, 0.94])

    y_axis_label = f"{trait_name} suppression (pp)\n(ctrl − inoculated at final step)"

    for row_idx, row_cfg in enumerate(ROWS):
        for col_idx, col_cfg in enumerate(cols):
            ax    = axes[row_idx, col_idx]
            x_key = col_cfg["x_key"]

            pts = [p for p in row_cfg["pts"] if not np.isnan(p.get(x_key, float("nan")))]

            if not pts:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10,
                        color="#888888", style="italic")
                ax.set_xlabel(col_cfg["x_label"], fontsize=9)
                ax.grid(True, alpha=0.3)
                if row_idx == 0:
                    ax.set_title(col_cfg["col_title"], fontsize=11,
                                 fontweight="bold", pad=8)
                continue

            all_xs = np.array([p[x_key] for p in pts])
            all_ys = np.array([p[y_key] for p in pts])

            # Scatter — coloured by source
            plotted_sources: set[str] = set()
            for src, style in SOURCE_STYLE.items():
                sub = [p for p in pts if p["source"] == src]
                if not sub:
                    continue
                kw = dict(style)
                if src in plotted_sources:
                    kw.pop("label", None)
                ax.scatter([p[x_key] for p in sub], [p[y_key] for p in sub], **kw)
                plotted_sources.add(src)

            # X range with 8% padding
            x_min, x_max = all_xs.min(), all_xs.max()
            pad    = max((x_max - x_min) * 0.08, 1e-6)
            x_line = np.linspace(x_min - pad, x_max + pad, 400)

            # Linear fit + CI
            if len(pts) >= 3:
                y_hat, y_lo, y_hi = linear_ci(all_xs, all_ys, x_line)
                ax.fill_between(x_line, y_lo, y_hi,
                                color=CI_COLOR, alpha=0.18, linewidth=0, label="95% CI")
                ax.plot(x_line, y_hat, "-", color=LINE_COLOR,
                        linewidth=2.0, label="Linear fit")
            elif len(pts) == 2:
                slope, intercept, *_ = scipy_stats.linregress(all_xs, all_ys)
                ax.plot(x_line, slope * x_line + intercept, "-",
                        color=LINE_COLOR, linewidth=2.0, label="Linear fit")

            # Correlation stats + fit residuals
            if len(pts) >= 3:
                r,   pr   = scipy_stats.pearsonr(all_xs, all_ys)
                rho, prho = scipy_stats.spearmanr(all_xs, all_ys)
                slope_s, intercept_s, *_ = scipy_stats.linregress(all_xs, all_ys)
                residuals = all_ys - (slope_s * all_xs + intercept_s)
                rmse = float(np.sqrt(np.mean(residuals ** 2)))
                ax.annotate(
                    f"r = {r:.2f}  (p={pr:.3f})\n"
                    f"ρ = {rho:.2f}  (p={prho:.3f})\n"
                    f"n = {len(pts)}   RMSE = {rmse:.2f}",
                    xy=(0.96, 0.05), xycoords="axes fraction", fontsize=8,
                    va="bottom", ha="right",
                    bbox=dict(fc="lightyellow", ec="#999900", alpha=0.90,
                              boxstyle="round,pad=0.35"),
                )

            ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
            ax.set_xlabel(col_cfg["x_label"], fontsize=9)

            if col_idx == 0:
                ax.set_ylabel(
                    f"{row_cfg['row_label']}\n\n{y_axis_label}",
                    fontsize=9,
                )
            else:
                ax.set_ylabel(y_axis_label, fontsize=9)

            ax.grid(True, alpha=0.3)

            if row_idx == 0:
                ax.set_title(col_cfg["col_title"], fontsize=11,
                             fontweight="bold", pad=8)

    # Shared legend
    legend_handles = [
        mlines.Line2D([], [], marker=s["marker"],
                      color=s["color"],
                      markeredgecolor=s.get("edgecolors", s["color"]),
                      markeredgewidth=s.get("linewidths", 0),
                      markersize=7, linestyle="none", label=s["label"])
        for src, s in SOURCE_STYLE.items()
    ] + [
        mlines.Line2D([], [], color=LINE_COLOR, linewidth=2,
                      linestyle="-", label="Linear fit"),
        mpatches.Patch(facecolor=CI_COLOR, alpha=0.35, edgecolor="none",
                       label="95% CI"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        fontsize=8.5,
        framealpha=0.9,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, -0.04),
    )

    n_pca_prompts = len([k for k in ACTIVE_PROMPT_NAMES if k in perp_prompts])
    if pc_fixed_tokens is not None and pc_mix_tokens is not None:
        token_note = " + per-token PCA (W_tokens N×K·L)"
    elif pc_fixed_tokens is not None:
        token_note = " + per-token PCA fixed (W_mix_tokens pending)"
    else:
        token_note = ""
    config_note = f"  [config: {CONFIG}]" if CONFIG != "all" else ""
    fig.suptitle(
        f"Inoculation metrics vs {trait_name} suppression at final checkpoint{config_note}\n"
        f"(2×{n_cols} grid: row 0 = Fixed prefix, row 1 = Mix prefix{token_note})\n"
        f"PCA fitted on {n_pca_prompts} prompts;  "
        f"Y-axis = (no-inoculation {trait_name}%) − (inoculated {trait_name}%)",
        fontsize=11, fontweight="bold", y=0.99,
    )

    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"plot_lls_metrics_{fname_suffix}{CONFIG_SUFFIX}_{ts}.png"
    fpath = os.path.join(PLOT_DIR, fname)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fpath}")
    return fpath


# ---------------------------------------------------------------------------
# Produce all 4 figures
# ---------------------------------------------------------------------------
print("\n── Generating figures ──────────────────────────────────────────────────")

# 2×4 basic: [Elicitation(Y), PH, other-trait PPD, PH − other-trait PPD]
build_and_save_figure(
    y_key="y_negative", trait_name=neg,
    fname_suffix="basic_negative", cols=COLS_BASIC_NEGATIVE,
)
build_and_save_figure(
    y_key="y_positive", trait_name=pos,
    fname_suffix="basic_positive", cols=COLS_BASIC_POSITIVE,
)

# 2×7 PCA: [γ, σ, SNR, PC1, PC2, PC1_tokens, PC2_tokens]
build_and_save_figure(
    y_key="y_negative", trait_name=neg,
    fname_suffix="pca_negative", cols=COLS_PCA,
)
build_and_save_figure(
    y_key="y_positive", trait_name=pos,
    fname_suffix="pca_positive", cols=COLS_PCA,
)

print("\nDone.  Four figures saved to", PLOT_DIR)
