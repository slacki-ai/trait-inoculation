#!/usr/bin/env python3
"""
plot_lls_metrics.py — LLS-inspired distributional metrics vs inoculation effectiveness.

For each inoculation prompt P the core per-example quantity is:
    w_i = lp_per_tok(completion_i | P + instruction_i)
        - lp_per_tok(completion_i | instruction_i)

i.e. by how much (in log-prob per token) does the base model find the
Playful/French training completion MORE likely when P is in the context?

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

Each metric is plotted against Playful or French suppression at the final
checkpoint, for both Fixed and Mix prefix conditions.  Four figures are saved:

  plot_lls_metrics_basic_playful_<ts>.png  — 2×4 basic:
        [Elicitation(Playful), PH, French PPD, PH−French PPD] × Y=Playful
        (mirrors the original combined scatter from plot_elicitation_vs_inoculation_combined.py
         with the 4th column changed from "French PH" to "PH − French PPD")

  plot_lls_metrics_basic_french_<ts>.png   — 2×4 basic:
        [Elicitation(French), PH, Playful PPD, PH−Playful PPD] × Y=French
        (Col 3 and 4 require Playful-only PPD — shown when computed)

  plot_lls_metrics_pca_playful_<ts>.png    — 2×7: [γ, σ, SNR, PC1, PC2, PC1_tokens, PC2_tokens] × Y=Playful
  plot_lls_metrics_pca_french_<ts>.png     — 2×7: [γ, σ, SNR, PC1, PC2, PC1_tokens, PC2_tokens] × Y=French

PCA is computed on all available prompts (up to 48 = 27 Playful + 21 French
inoculation), so that French prompts enrich the embedding space even before
French training runs produce Y-axis data.  Scatter plot data points come from
whatever training run JSONs are present on disk.

Previous figures (using only the 27 Playful prompts) are preserved on disk
with their original timestamps.

Usage:
    python plot_lls_metrics.py [--config {all,french_only,playful_only}]
Output:
    plots/plot_lls_metrics_{basic|pca}_{playful|french}[_<config>]_<timestamp>.png
"""

import json
import os
from datetime import datetime
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA as SklearnPCA

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
_ap = argparse.ArgumentParser(description="LLS metrics scatter plots")
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
BASE       = "/Users/claude/vibe-research/inoculation-bootstrap-heuristic"
PERP_PATH  = f"{BASE}/results/perplexity_heuristic_qwen2.5-7b-instruct.json"
ELICIT_PATH= f"{BASE}/results/elicitation_scores.json"
V3_PATH    = f"{BASE}/results/scores_multi_prompt_v3_qwen2.5-7b-instruct.json"
V4_PATH    = f"{BASE}/results/scores_multi_prompt_v4_qwen2.5-7b-instruct.json"
V5_PATH    = f"{BASE}/results/scores_multi_prompt_v5_qwen2.5-7b-instruct.json"
VNEG_PATH  = f"{BASE}/results/scores_multi_prompt_neg_qwen2.5-7b-instruct.json"
# French training run JSONs — loaded if present, skipped otherwise
FRENCH_V3_PATH  = f"{BASE}/results/scores_multi_prompt_french_v3_qwen2.5-7b-instruct.json"
FRENCH_V4_PATH  = f"{BASE}/results/scores_multi_prompt_french_v4_qwen2.5-7b-instruct.json"
FRENCH_NEG_PATH = f"{BASE}/results/scores_multi_prompt_french_neg_qwen2.5-7b-instruct.json"
PLOT_DIR   = f"{BASE}/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Prompt lists
# ---------------------------------------------------------------------------

# ── Playful prompts (27) ─────────────────────────────────────────────────────
V3_PROMPT_NAMES = [
    "clown_persona", "humor_matters", "enjoys_joking", "joke_nevermind",
    "clowns_interesting", "playfulness_trait", "playfulness_enriches",
    "laughter_medicine", "had_fun_today",
]
V4_PROMPT_NAMES = [
    "corrected_inoculation", "whimsical", "witty",
    "strong_elicitation", "comedian_answers", "comedian_mindset",
]
V5_PROMPT_NAMES = [                          # shared with French
    "the_sky_is_blue", "i_like_cats", "professional_tone",
    "financial_advisor", "be_concise", "think_step_by_step",
]
VNEG_PROMPT_NAMES = [
    "corrected_inoculation_neg", "whimsical_neg", "witty_neg",
    "strong_elicitation_neg", "comedian_answers_neg", "comedian_mindset_neg",
]

# ── French inoculation prompts (21 new; v5 shared above) ─────────────────────
FRENCH_V3_NAMES = [
    "french_persona", "french_matters", "enjoys_french", "paris_nevermind",
    "french_interesting", "french_trait", "french_enriches",
    "french_love", "french_today",
]
FRENCH_V4_NAMES = [
    "french_agent", "fluent_french", "natural_french",
    "answer_french", "french_answers", "think_french",
]
FRENCH_NEG_NAMES = [
    "french_agent_neg", "fluent_french_neg", "natural_french_neg",
    "answer_french_neg", "french_answers_neg", "think_french_neg",
]

# All 48 unique prompt keys for PCA (27 Playful + 21 French-specific; v5 shared)
ALL_PROMPT_NAMES_48 = (
    V3_PROMPT_NAMES + V4_PROMPT_NAMES + V5_PROMPT_NAMES + VNEG_PROMPT_NAMES
    + FRENCH_V3_NAMES + FRENCH_V4_NAMES + FRENCH_NEG_NAMES
)

# Active prompt names — filtered by --config
if CONFIG == "french_only":
    ACTIVE_PROMPT_NAMES = (
        FRENCH_V3_NAMES + FRENCH_V4_NAMES + FRENCH_NEG_NAMES + V5_PROMPT_NAMES
    )
elif CONFIG == "playful_only":
    ACTIVE_PROMPT_NAMES = (
        V3_PROMPT_NAMES + V4_PROMPT_NAMES + VNEG_PROMPT_NAMES + V5_PROMPT_NAMES
    )
else:  # "all"
    ACTIVE_PROMPT_NAMES = ALL_PROMPT_NAMES_48

print(f"Active prompts: {len(ACTIVE_PROMPT_NAMES)}")

SOURCE_BY_KEY = (
    {k: "v3"           for k in V3_PROMPT_NAMES}
    | {k: "v4"         for k in V4_PROMPT_NAMES}
    | {k: "v5"         for k in V5_PROMPT_NAMES}
    | {k: "neg"        for k in VNEG_PROMPT_NAMES}
    | {k: "fr_v3"      for k in FRENCH_V3_NAMES}
    | {k: "fr_v4"      for k in FRENCH_V4_NAMES}
    | {k: "fr_neg"     for k in FRENCH_NEG_NAMES}
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(PERP_PATH) as f:
    perp_data = json.load(f)
with open(ELICIT_PATH) as f:
    elicit = json.load(f)

perp_prompts     = perp_data["prompts"]
lp_train_default = np.array(perp_data["baseline"]["lp_train_default"])

# Elicitation baselines (no-prefix condition)
_elicit_neutral       = elicit["neutral"]["scores"]
ELICIT_BASE_PLAYFUL   = _elicit_neutral["Playful"]["mean"]
ELICIT_BASE_FRENCH    = _elicit_neutral["French"]["mean"]

print(f"Loaded {len(perp_prompts)} prompts from perplexity JSON "
      f"({sum(k in perp_prompts for k in FRENCH_V3_NAMES + FRENCH_V4_NAMES + FRENCH_NEG_NAMES)} "
      f"French prompts present)")

def _scores(path):
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        print(f"Loaded {len(d)} runs from {os.path.basename(path)}")
        return d
    print(f"Missing (optional): {path}")
    return {}

v3        = _scores(V3_PATH)
v4        = _scores(V4_PATH)
v5        = _scores(V5_PATH)
vneg      = _scores(VNEG_PATH)
french_v3 = _scores(FRENCH_V3_PATH)    # empty dict until French training runs exist
french_v4 = _scores(FRENCH_V4_PATH)
french_neg= _scores(FRENCH_NEG_PATH)

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
    mean_abs_w = float(np.mean(np.abs(w)))   # mean|w_i| = PPD on Playful training completions
    std_w      = float(np.std(w, ddof=1)) if len(w) > 1 else float("nan")
    snr        = (mean_w / std_w) if (std_w > 0 and not np.isnan(std_w)) else float("nan")
    frac_pos   = float(np.mean(w > 0))

    return dict(
        frac_pos   = frac_pos,
        std_w      = std_w,
        snr        = snr,
        mean_w     = mean_w,       # PH
        mean_abs_w = mean_abs_w,   # Playful PPD (|logprob drift| on Playful completions)
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
ALL_PROMPT_NAMES = ACTIVE_PROMPT_NAMES  # kept for backward compat within this file


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
        w  = np.where(np.isfinite(w), w, 0.0)   # fill NaN / Inf with 0
        rows.append(w)

    W      = np.array(rows)                          # (N, K)
    pca    = SklearnPCA(n_components=2, random_state=42)
    coords = pca.fit_transform(W)                    # (N, 2)
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
_TOKENS_PATH = PERP_PATH.replace("perplexity_heuristic_", "perplexity_heuristic_tokens_", 1)

_tokens_data: dict | None = None
if os.path.exists(_TOKENS_PATH):
    print(f"  Loading token logprobs from {os.path.basename(_TOKENS_PATH)} …")
    with open(_TOKENS_PATH) as _f:
        _tokens_data = json.load(_f)
    print(f"  Loaded.  Prompts: {list(_tokens_data['prompts'].keys())[:4]} …")
else:
    print(f"  Token logprobs file not found ({os.path.basename(_TOKENS_PATH)}) "
          f"— PC1_tokens / PC2_tokens will be NaN.  "
          f"Run compute_perplexity_heuristic_tokens.py first.")


def build_pc_coords_tokens(
    key_list: list[str],
) -> dict[str, tuple[float, float]] | None:
    """
    Build W_tokens by concatenating per-token logprob differences across all K
    training examples:

        row_n = concat_k [ lp_inoc_tokens[n][k] − lp_default_tokens[k] ]

    Shape: N × sum(L_k).  Since the same K completions are used for every
    prompt, the feature positions are consistent across rows — no padding
    needed.  Runs PCA(n_components=2) and returns {key: (pc1_tokens, pc2_tokens)}.

    Returns None if token data is unavailable or fewer than 3 prompts qualify.
    """
    if _tokens_data is None:
        return None

    baseline_toks = _tokens_data["baseline"]["lp_train_default_tokens"]   # K lists
    prompts_toks  = _tokens_data["prompts"]

    keys_ok = [k for k in key_list if k in prompts_toks]
    if len(keys_ok) < 3:
        print(f"  PCA tokens: only {len(keys_ok)} prompts — skipping")
        return None

    rows = []
    for key in keys_ok:
        inoc_toks = prompts_toks[key]["lp_train_inoc_tokens"]   # K lists
        # Concatenate token-level differences for all K completions
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

    # Ensure all rows have the same length (they should: same completions)
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
    """
    Build W_mix_tokens by concatenating per-token logprob differences using
    index-matched rephrasings:

        row_n = concat_k [ lp_mix_tokens[n][k] − lp_default_tokens[k] ]

    Requires lp_train_mix_tokens to be present in the tokens data (added by
    compute_perplexity_heuristic_mix_tokens.py).  Returns None if absent.
    """
    if _tokens_data is None:
        return None

    baseline_toks = _tokens_data["baseline"]["lp_train_default_tokens"]
    prompts_toks  = _tokens_data["prompts"]

    # Only include prompts that have lp_train_mix_tokens
    keys_ok = [k for k in key_list
               if k in prompts_toks and "lp_train_mix_tokens" in prompts_toks[k]]
    if len(keys_ok) < 3:
        print(f"  PCA mix tokens: only {len(keys_ok)} prompts with lp_train_mix_tokens — skipping")
        return None

    rows = []
    for key in keys_ok:
        mix_toks = prompts_toks[key]["lp_train_mix_tokens"]   # K lists
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
    """Mutate pt in place, adding pc1_tokens and pc2_tokens (or NaN).
    Pass pc_fixed_tokens for fixed-prefix rows, pc_mix_tokens for mix rows.
    """
    pc = coords.get(base_key) if coords else None
    pt["pc1_tokens"] = pc[0] if pc is not None else float("nan")
    pt["pc2_tokens"] = pc[1] if pc is not None else float("nan")


# ---------------------------------------------------------------------------
# Control baseline for suppression Y-axis
# ---------------------------------------------------------------------------
def get_final_score(run_data: dict, trait: str, condition: str = "default") -> float:
    steps = sorted(int(s) for s in run_data["steps"].keys())
    return run_data["steps"][str(max(steps))][condition][trait]["mean"]

ctrl_french  = get_final_score(v3["no_inoculation"], "French")
ctrl_playful = get_final_score(v3["no_inoculation"], "Playful")
print(f"\nControl Playful (no inoculation): {ctrl_playful:.1f}%")
print(f"Control French  (no inoculation): {ctrl_french:.1f}%\n")

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

    final_playful = get_final_score(run_data, "Playful")
    final_french  = get_final_score(run_data, "French")

    # Elicitation (relative to no-prefix baseline)
    elicit_entry = elicit.get(base_key, {}).get("scores", {})
    elicit_playful = (elicit_entry.get("Playful", {}).get("mean", float("nan"))
                      - ELICIT_BASE_PLAYFUL)
    elicit_french  = (elicit_entry.get("French",  {}).get("mean", float("nan"))
                      - ELICIT_BASE_FRENCH)

    # Cross-trait PPD: |logprob drift| on the OTHER trait's completions.
    # Both values come from dedicated perplexity jobs run on held-out OTHER-trait
    # completions (not training data), mirroring each other symmetrically:
    #   french_ppd  = mean|Δlp| on French-only control completions
    #   playful_ppd = mean|Δlp| on Playful-only control completions (not yet run
    #                 for French inoculation prompts → NaN until job completes)
    perp_entry      = perp_prompts.get(base_key, {})
    french_ppd_val  = float(perp_entry.get("french_ppd",  float("nan")))
    playful_ppd_val = float(perp_entry.get("playful_ppd", float("nan")))

    mean_w = m["mean_w"]
    ph_minus_french_ppd  = (mean_w - french_ppd_val
                            if np.isfinite(french_ppd_val) else float("nan"))
    ph_minus_playful_ppd = mean_w - playful_ppd_val

    return dict(
        label     = label,
        source    = source,
        y_playful = ctrl_playful - final_playful,   # suppression: higher = more suppression
        y_french  = ctrl_french  - final_french,
        elicit_playful       = elicit_playful,
        elicit_french        = elicit_french,
        french_ppd           = french_ppd_val,
        playful_ppd          = playful_ppd_val,
        ph_minus_french_ppd  = ph_minus_french_ppd,
        ph_minus_playful_ppd = ph_minus_playful_ppd,
        **m,
    )

fixed_pts: list[dict] = []
mix_pts:   list[dict] = []


def _add_group(name_list, score_dict, source_tag):
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


# v5 (neutral) prompts — always included in all configs
_add_group(V5_PROMPT_NAMES, v5, "v5")

# Playful training runs — only in "all" and "playful_only"
if CONFIG in ("all", "playful_only"):
    _add_group(V3_PROMPT_NAMES,   v3,   "v3")
    _add_group(V4_PROMPT_NAMES,   v4,   "v4")
    _add_group(VNEG_PROMPT_NAMES, vneg, "neg")

# French training runs — only in "all" and "french_only"
if CONFIG in ("all", "french_only"):
    _add_group(FRENCH_V3_NAMES,  french_v3,  "fr_v3")
    _add_group(FRENCH_V4_NAMES,  french_v4,  "fr_v4")
    _add_group(FRENCH_NEG_NAMES, french_neg, "fr_neg")

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
# Plot configuration — trait-specific column sets
# ---------------------------------------------------------------------------

# 2×7 PCA figures: γ / σ / SNR / PC1 / PC2 / PC1_tokens / PC2_tokens
# (same columns for both Y=Playful and Y=French figures)
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
COLS_PCA_PLAYFUL = COLS_PCA
COLS_PCA_FRENCH  = COLS_PCA

# 2×4 basic figures: [Elicitation(Y), PH, |logprob drift on other trait|, PH − other PPD]
# For Y=Playful: other trait = French  →  french_ppd / ph_minus_french_ppd
# For Y=French:  other trait = Playful →  playful_ppd / ph_minus_playful_ppd (pending)

COLS_BASIC_PLAYFUL = [
    dict(
        x_key     = "elicit_playful",
        x_label   = "Elicitation strength of Playful (pp)\n"
                     "(Playful(with prefix) − Playful(no prefix);\n"
                     " positive → prefix elicits Playful in base model)",
        col_title = "Elicitation (Playful)",
    ),
    dict(
        x_key     = "mean_w",
        x_label   = "PH = mean(wᵢ) — Mean logprob diff\n"
                     "(average per-token logprob increase from prefix;\n"
                     " positive → prefix primes training completions)",
        col_title = "PH  (mean logprob diff)",
    ),
    dict(
        x_key     = "french_ppd",
        x_label   = "French PPD = mean|Δlp| on French-only completions\n"
                     "(absolute logprob drift on French-only control data;\n"
                     " proxy for cross-trait gradient noise from prefix)",
        col_title = "French PPD  (other-trait |logprob drift|)",
    ),
    dict(
        x_key     = "ph_minus_french_ppd",
        x_label   = "PH − French PPD\n"
                     "(on-trait signal minus cross-trait noise;\n"
                     " larger → cleaner inoculation signal relative to noise)",
        col_title = "PH − French PPD",
    ),
]

COLS_BASIC_FRENCH = [
    dict(
        x_key     = "elicit_french",
        x_label   = "Elicitation strength of French (pp)\n"
                     "(French(with prefix) − French(no prefix);\n"
                     " positive → prefix elicits French in base model)",
        col_title = "Elicitation (French)",
    ),
    dict(
        x_key     = "mean_w",
        x_label   = "PH = mean(wᵢ) — Mean logprob diff\n"
                     "(average per-token logprob increase from prefix;\n"
                     " positive → prefix primes training completions)",
        col_title = "PH  (mean logprob diff)",
    ),
    dict(
        x_key     = "playful_ppd",
        x_label   = "Playful PPD = mean|Δlp| on Playful-only completions\n"
                     "(absolute logprob drift on Playful-only control data;\n"
                     " not yet computed — shown when available)",
        col_title = "Playful PPD  (other-trait |logprob drift|)",
    ),
    dict(
        x_key     = "ph_minus_playful_ppd",
        x_label   = "PH − Playful PPD\n"
                     "(on-trait signal minus cross-trait noise;\n"
                     " not yet computed — shown when available)",
        col_title = "PH − Playful PPD",
    ),
]

print(f"Column sets: BASIC_PLAYFUL={len(COLS_BASIC_PLAYFUL)}, "
      f"BASIC_FRENCH={len(COLS_BASIC_FRENCH)}, "
      f"PCA={len(COLS_PCA)}")

ROWS = [
    dict(pts=fixed_pts, row_label="Fixed prefix"),
    dict(pts=mix_pts,   row_label="Mix (rephrased) prefix"),
]

SOURCE_STYLE = {
    "v3":    dict(marker="o",  color="#e15759", s=55, alpha=0.85, zorder=3,
                  label="Playful v3 (weak–medium)"),
    "v4":    dict(marker="D",  color="#f28e2b", s=65, alpha=1.0,
                  edgecolors="black", linewidths=0.6, zorder=4,
                  label="Playful v4 (strong)"),
    "v5":    dict(marker="s",  color="#4e79a7", s=65, alpha=1.0,
                  edgecolors="black", linewidths=0.6, zorder=4,
                  label="Playful v5 (near-zero)"),
    "neg":   dict(marker="v",  color="#76b7b2", s=65, alpha=1.0,
                  edgecolors="black", linewidths=0.6, zorder=4,
                  label="Playful neg (negative)"),
    "fr_v3": dict(marker="o",  color="#b07aa1", s=55, alpha=0.85, zorder=3,
                  label="French v3 (weak–medium)"),
    "fr_v4": dict(marker="D",  color="#9c755f", s=65, alpha=1.0,
                  edgecolors="black", linewidths=0.6, zorder=4,
                  label="French v4 (strong)"),
    "fr_neg":dict(marker="v",  color="#bab0ac", s=65, alpha=1.0,
                  edgecolors="black", linewidths=0.6, zorder=4,
                  label="French neg (negative)"),
}

LINE_COLOR = "#1a6faf"
CI_COLOR   = "#1a6faf"

# ---------------------------------------------------------------------------
# Figure builder — reused for all 4 figures
# ---------------------------------------------------------------------------
def build_and_save_figure(
    y_key: str,
    trait_name: str,      # e.g. "Playful" or "French"
    fname_suffix: str,    # appended before config+timestamp in filename
    cols: list[dict],     # COLS_BASIC (4 cols) or COLS_PCA (7 cols)
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
    y_key="y_playful", trait_name="Playful",
    fname_suffix="basic_playful", cols=COLS_BASIC_PLAYFUL,
)
build_and_save_figure(
    y_key="y_french",  trait_name="French",
    fname_suffix="basic_french",  cols=COLS_BASIC_FRENCH,
)

# 2×7 PCA: 4 basic cols + [γ, σ, SNR, PC1, PC2, PC1_tokens]
build_and_save_figure(
    y_key="y_playful", trait_name="Playful",
    fname_suffix="pca_playful",   cols=COLS_PCA_PLAYFUL,
)
build_and_save_figure(
    y_key="y_french",  trait_name="French",
    fname_suffix="pca_french",    cols=COLS_PCA_FRENCH,
)

print("\nDone.  Four figures saved to", PLOT_DIR)
