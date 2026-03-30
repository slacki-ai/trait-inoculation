#!/usr/bin/env python3
"""
Plot fixed-vs-mix inoculation gap against 10 pre-training heuristics.

Produces one 6-row figure per experiment:

  Rows 1–2  (10 cols each)  — raw logprob-diff heuristics
    Row 1  Y = fixed_trait/default − mix_trait/default  (pp)
           Negative when fixed inoculation suppresses more than mix rephrasings.
    Row 2  Y = no_inoculation_final − mix_trait/default  (pp)
           Mix suppression relative to a fully trained no-inoculation model.

  Rows 3–4  (1 wide panel each)  — PC1 from PCA of W_fixed
    Row 3  X = PC1_PCA_fixed,  Y = gap  (same as Row 1)
    Row 4  X = PC1_PCA_fixed,  Y = suppression  (same as Row 2)

  Rows 5–6  (1 wide panel each)  — PC1 from TruncatedSVD of W_fixed
    Row 5  X = PC1_TSVD_fixed, Y = gap
    Row 6  X = PC1_TSVD_fixed, Y = suppression

Each experiment includes:
  Playful    — Playful-trained (v3/v4/neg) + neutral (v5)  ≈ 27 prompts
  French     — French-trained (v3/v4/neg) + neutral (v5)   ≈ 27 prompts
  German     — German-trained (de_v4/de_neg) + neutral     ≈  4 prompts
  Flattering — Flat-trained  (flat_v4/flat_neg) + neutral  ≈  4 prompts

Usage:
    cd <repo_root>
    python experiments/logprob_heuristic/analysis/plot_fixed_vs_mix_heuristics.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA, TruncatedSVD

# ── Repo root ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

PLOT_DIR = ROOT / "plots"
_R       = ROOT / "results"

# ── Experiment specs ──────────────────────────────────────────────────────────

@dataclass
class ExpSpec:
    name: str
    trait: str                         # key in score JSON, e.g. "Playful"
    perp_file: Path
    score_files: list[Path]            # contain fixed+mix run pairs
    model_slug: str
    # If set, only these prompt keys are included in this experiment's figure.
    # Use to exclude cross-trait prompts (e.g. don't put French prompts in
    # the Playful figure).  None = include all prompts that have perp data.
    allowed_prompt_keys: set[str] | None = None
    # Optional extra files used ONLY to locate the no_inoculation baseline.
    # If empty, score_files are searched for no_inoculation instead.
    baseline_score_files: list[Path] = field(default_factory=list)


# ── Prompt-key sets for Qwen-7B filtering ────────────────────────────────────

def _qwen_score_keys(score_files: list[Path]) -> set[str]:
    """Return all non-mix, non-no_inoculation run keys from a list of score files."""
    keys: set[str] = set()
    for p in score_files:
        if not p.exists():
            continue
        for k in json.load(open(p)):
            if not k.endswith("_mix") and k != "no_inoculation":
                keys.add(k)
    return keys


# Qwen 7B score file groups
_PLAYFUL_SCORE_FILES = [
    _R / "scores_multi_prompt_v3_qwen2.5-7b-instruct.json",
    _R / "scores_multi_prompt_v4_qwen2.5-7b-instruct.json",
    _R / "scores_multi_prompt_neg_qwen2.5-7b-instruct.json",
]
_FRENCH_SCORE_FILES = [
    _R / "scores_multi_prompt_french_v3_qwen2.5-7b-instruct.json",
    _R / "scores_multi_prompt_french_v4_qwen2.5-7b-instruct.json",
    _R / "scores_multi_prompt_french_neg_qwen2.5-7b-instruct.json",
]
_NEUTRAL_SCORE_FILES = [
    _R / "scores_multi_prompt_v5_qwen2.5-7b-instruct.json",   # neutral prompts
]

_PLAYFUL_KEYS  = _qwen_score_keys(_PLAYFUL_SCORE_FILES)   # 21 Playful-trained
_FRENCH_KEYS   = _qwen_score_keys(_FRENCH_SCORE_FILES)    # 21 French-trained
_NEUTRAL_KEYS  = _qwen_score_keys(_NEUTRAL_SCORE_FILES)   # 6 neutral prompts

# GF group assignments (from experiment_configs/german_flattering_8b.yaml)
_GERMAN_TRAINED_KEYS    = {"answer_german", "think_german_neg"}
_FLATTERING_TRAINED_KEYS = {"flatterer_mindset", "avoid_flattery"}
_GF_NEUTRAL_KEYS        = {"birds_sing", "coffee_is_hot"}


# All Qwen score files (for a single merged load)
_QWEN_7B_PERP = _R / "perplexity_heuristic_qwen2.5-7b-instruct.json"
_QWEN_7B_ALL_SCORES = (
    _PLAYFUL_SCORE_FILES + _NEUTRAL_SCORE_FILES + _FRENCH_SCORE_FILES
)

_GF_PERP   = _R / "perplexity_heuristic_german_flattering_llama-3.1-8b-instruct.json"
_GF_SCORES = [_R / "scores_german_flattering_llama-3.1-8b-instruct.json"]

EXPERIMENTS: list[ExpSpec] = [
    ExpSpec(
        name="Playful / Qwen-2.5-7B",
        trait="Playful",
        perp_file=_QWEN_7B_PERP,
        score_files=_QWEN_7B_ALL_SCORES,
        model_slug="qwen2.5-7b-instruct",
        # Playful-trained prompts + neutral v5 (exclude French-trained)
        allowed_prompt_keys=_PLAYFUL_KEYS | _NEUTRAL_KEYS,
    ),
    ExpSpec(
        name="French / Qwen-2.5-7B",
        trait="French",
        perp_file=_QWEN_7B_PERP,
        score_files=_QWEN_7B_ALL_SCORES,
        model_slug="qwen2.5-7b-instruct",
        # French-trained prompts + neutral v5 (exclude Playful-trained)
        allowed_prompt_keys=_FRENCH_KEYS | _NEUTRAL_KEYS,
    ),
    ExpSpec(
        name="German / Llama-3.1-8B",
        trait="German",
        perp_file=_GF_PERP,
        score_files=_GF_SCORES,
        model_slug="llama-3.1-8b-instruct",
        # German-trained + neutral (exclude Flattering-trained)
        allowed_prompt_keys=_GERMAN_TRAINED_KEYS | _GF_NEUTRAL_KEYS,
    ),
    ExpSpec(
        name="Flattering / Llama-3.1-8B",
        trait="Flattering",
        perp_file=_GF_PERP,
        score_files=_GF_SCORES,
        model_slug="llama-3.1-8b-instruct",
        # Flattering-trained + neutral (exclude German-trained)
        allowed_prompt_keys=_FLATTERING_TRAINED_KEYS | _GF_NEUTRAL_KEYS,
    ),
]

# ── Heuristic spec (10 columns) ───────────────────────────────────────────────
# (data_key,  column header,  x-axis label)
HEURISTICS: list[tuple[str, str, str]] = [
    (
        "PH_ratio",
        "PH_mix / PH_fixed",
        "PH_mix / PH_fixed\n(signed signal ratio;\noutliers kept)",
    ),
    (
        "sigma2_diff",
        "σ²_mix − σ²_fixed",
        "σ²_mix − σ²_fixed\n(rephrasing noise\nadded to gradient)",
    ),
    (
        "gamma_mix",
        "γ_mix",
        "γ_mix\n(frac. examples shifted\npositively under mix)",
    ),
    (
        "SNR_mix",
        "SNR_mix",
        "SNR_mix = PH_mix / σ_mix\n(signed SNR\nunder mix)",
    ),
    (
        "cosine",
        "cos(W_fixed, W_mix)",
        "cos(W_fixed, W_mix)\n(per-example priming\nalignment)",
    ),
    (
        "eff_rank",
        "Eff. rank(W_mix)",
        "Effective rank of |W_mix|\n(entropy of per-example\ninfluence)",
    ),
    (
        "SNR_ratio",
        "SNR_mix / SNR_fixed",
        "SNR_mix / SNR_fixed\n(signed SNR\ndegradation)",
    ),
    (
        "MALD_ratio",
        "MALD_mix / MALD_fixed",
        "MALD_mix / MALD_fixed\n(mean|Δlogprob| ratio;\nrobust, always >0)",
    ),
    (
        "SNR_abs_mix",
        "MALD_mix / σ_mix",
        "MALD_mix / σ_mix\n(absolute SNR\nunder mix)",
    ),
    (
        "SNR_abs_ratio",
        "SNR_abs_mix / SNR_abs_fixed",
        "SNR_abs_mix / SNR_abs_fixed\n(absolute SNR\ndegradation)",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_perplexity(path: Path) -> tuple[dict[str, dict], np.ndarray]:
    with open(path) as f:
        raw = json.load(f)
    lp_default = np.array(raw["baseline"]["lp_train_default"], dtype=float)
    prompts    = raw["prompts"]
    print(f"  [perp]    {len(prompts)} prompts, baseline N={len(lp_default)}  ({path.name})")
    return prompts, lp_default


def load_all_scores(score_files: list[Path]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for p in score_files:
        if not p.exists():
            print(f"  [scores]  MISSING: {p.name}")
            continue
        with open(p) as f:
            data = json.load(f)
        for run_name, run_data in data.items():
            if run_name not in merged:
                merged[run_name] = run_data
    print(f"  [scores]  {len(merged)} runs from {len(score_files)} file(s)")
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Score helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _step_score(run_data: dict, step_key: str, trait: str, cond: str) -> float:
    try:
        return float(run_data["steps"][step_key][cond][trait]["mean"])
    except (KeyError, TypeError):
        return float("nan")


def get_final_score(run_data: dict, trait: str, cond: str = "default") -> float:
    steps = run_data.get("steps", {})
    if not steps:
        return float("nan")
    return _step_score(run_data, str(max(int(s) for s in steps)), trait, cond)


def get_trained_baseline(all_scores: dict[str, dict], trait: str) -> float:
    run = all_scores.get("no_inoculation")
    if run is None:
        return float("nan")
    return get_final_score(run, trait, "default")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Heuristic computation
# ═══════════════════════════════════════════════════════════════════════════════

def _heuristics_from_W(W_fixed: np.ndarray, W_mix: np.ndarray) -> dict[str, float]:
    """
    Compute all 10 heuristics from a pair of 1-D logprob-diff vectors for one
    prompt.  W_fixed and W_mix must already be finite and have the same length.
    """
    PH_fixed    = float(np.mean(W_fixed))
    PH_mix      = float(np.mean(W_mix))
    sig_fixed   = float(np.std(W_fixed, ddof=1))
    sig_mix     = float(np.std(W_mix,   ddof=1))
    MALD_fixed  = float(np.mean(np.abs(W_fixed)))
    MALD_mix    = float(np.mean(np.abs(W_mix)))
    gamma_mix   = float(np.mean(W_mix > 0))

    SNR_fixed     = PH_fixed   / sig_fixed  if sig_fixed  > 1e-9 else float("nan")
    SNR_mix_v     = PH_mix     / sig_mix    if sig_mix    > 1e-9 else float("nan")
    SNR_abs_fixed = MALD_fixed / sig_fixed  if sig_fixed  > 1e-9 else float("nan")
    SNR_abs_mix   = MALD_mix   / sig_mix    if sig_mix    > 1e-9 else float("nan")

    nf     = np.linalg.norm(W_fixed)
    nm     = np.linalg.norm(W_mix)
    cosine = (
        float(np.dot(W_fixed, W_mix) / (nf * nm))
        if nf > 1e-9 and nm > 1e-9 else float("nan")
    )

    abs_w = np.abs(W_mix)
    w_sum = abs_w.sum()
    if w_sum > 1e-9:
        p        = np.clip(abs_w / w_sum, 1e-12, None)
        eff_rank = float(np.exp(-np.sum(p * np.log(p))))
    else:
        eff_rank = float("nan")

    PH_ratio      = PH_mix / PH_fixed if abs(PH_fixed) > 1e-9 else float("nan")
    sigma2_diff   = sig_mix**2 - sig_fixed**2
    SNR_ratio     = (
        SNR_mix_v / SNR_fixed
        if np.isfinite(SNR_fixed) and abs(SNR_fixed) > 1e-6
        else float("nan")
    )
    MALD_ratio    = MALD_mix / MALD_fixed if MALD_fixed > 1e-9 else float("nan")
    SNR_abs_ratio = (
        SNR_abs_mix / SNR_abs_fixed
        if np.isfinite(SNR_abs_fixed) and abs(SNR_abs_fixed) > 1e-6
        else float("nan")
    )

    return {
        "PH_ratio":      PH_ratio,
        "sigma2_diff":   sigma2_diff,
        "gamma_mix":     gamma_mix,
        "SNR_mix":       SNR_mix_v,
        "cosine":        cosine,
        "eff_rank":      eff_rank,
        "SNR_ratio":     SNR_ratio,
        "MALD_ratio":    MALD_ratio,
        "SNR_abs_mix":   SNR_abs_mix,
        "SNR_abs_ratio": SNR_abs_ratio,
    }


def compute_heuristics(
    prompts: dict[str, dict],
    lp_default: np.ndarray,
    allowed_keys: set[str] | None = None,
) -> dict[str, dict[str, float]]:
    n_base  = len(lp_default)
    results: dict[str, dict[str, float]] = {}

    for key, entry in prompts.items():
        if allowed_keys is not None and key not in allowed_keys:
            continue
        raw_inoc = entry.get("lp_train_inoc")
        raw_mix  = entry.get("lp_train_mix")
        if raw_inoc is None or raw_mix is None:
            continue

        lp_inoc = np.array(raw_inoc, dtype=float)
        lp_mix  = np.array(raw_mix,  dtype=float)
        n       = min(len(lp_inoc), len(lp_mix), n_base)
        lp_inoc = lp_inoc[:n]
        lp_mix  = lp_mix[:n]
        lp_base = lp_default[:n]

        W_fixed = lp_inoc - lp_base
        W_mix   = lp_mix  - lp_base
        mask    = np.isfinite(W_fixed) & np.isfinite(W_mix)
        if mask.sum() < 20:
            continue
        W_fixed = W_fixed[mask]
        W_mix   = W_mix[mask]

        h = _heuristics_from_W(W_fixed, W_mix)
        # Also store raw W vectors so PC1 rows can project them
        h["_W_fixed"] = W_fixed.tolist()
        h["_W_mix"]   = W_mix.tolist()
        results[key]  = h

    print(f"  [heuristics] {len(results)} prompts computed")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  PC1-projected heuristics (PCA and TruncatedSVD of W_fixed)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_W_matrices(
    heuristics_data: dict[str, dict],
    keys_ordered: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (valid_keys, W_fixed (N×K), W_mix (N×K))."""
    rows_fixed, rows_mix, valid_keys = [], [], []
    for k in keys_ordered:
        entry = heuristics_data.get(k)
        if not entry or "_W_fixed" not in entry:
            continue
        rows_fixed.append(np.array(entry["_W_fixed"], dtype=float))
        rows_mix.append(np.array(entry["_W_mix"],   dtype=float))
        valid_keys.append(k)
    if len(valid_keys) < 2:
        return valid_keys, np.empty((0, 0)), np.empty((0, 0))
    K = min(len(r) for r in rows_fixed + rows_mix)
    return (
        valid_keys,
        np.stack([r[:K] for r in rows_fixed]),
        np.stack([r[:K] for r in rows_mix]),
    )


def compute_pc1_projected_heuristics(
    heuristics_data: dict[str, dict],
    keys_ordered: list[str],
    method: str,  # "pca" or "tsvd"
) -> dict[str, dict[str, float]]:
    """
    Replace each prompt's raw W vectors with their rank-1 PC1 reconstruction,
    then compute the same 10 heuristics from the projected vectors.

    For PCA  (centered):
        mu    = mean(W_fixed, axis=0)
        v1    = first PC of (W_fixed − mu)
        f_n   = (W_fixed[n] − mu) @ v1
        m_n   = (W_mix[n]  − mu) @ v1
        W_pc1_fixed[n] = f_n * v1   (rank-1 reconstruction, zero-mean)
        W_pc1_mix[n]   = m_n * v1

    For TruncSVD (uncentered):
        v1    = first right singular vector of W_fixed
        f_n   = W_fixed[n] @ v1
        m_n   = W_mix[n]  @ v1
        W_pc1_fixed[n] = f_n * v1
        W_pc1_mix[n]   = m_n * v1

    Several heuristics become degenerate under rank-1 projection (e.g. eff_rank
    collapses to a constant; cosine becomes ±1).  This is expected and
    scientifically meaningful — it shows what information survives PC1 filtering.
    """
    valid_keys, W_fixed, W_mix = _build_W_matrices(heuristics_data, keys_ordered)
    if len(valid_keys) < 2:
        return {}

    if method == "pca":
        mu = W_fixed.mean(axis=0)                          # (K,)
        pca = PCA(n_components=1, random_state=0)
        pca.fit(W_fixed)
        v1 = pca.components_[0]                            # (K,)
        f  = (W_fixed - mu) @ v1                           # (N,) scores
        m  = (W_mix   - mu) @ v1
        var_exp = float(pca.explained_variance_ratio_[0])
        print(f"  [PC1-PCA]  var_explained={var_exp:.1%}")
    else:  # tsvd
        tsvd = TruncatedSVD(n_components=1, random_state=0)
        tsvd.fit(W_fixed)
        v1 = tsvd.components_[0]                           # (K,)
        f  = W_fixed @ v1                                  # (N,)
        m  = W_mix   @ v1
        var_exp = float(tsvd.explained_variance_ratio_[0])
        print(f"  [PC1-TSVD] var_explained={var_exp:.1%}")

    results: dict[str, dict[str, float]] = {}
    for i, k in enumerate(valid_keys):
        # Rank-1 reconstruction: W_pc1[n] = score[n] * v1
        w_fixed_pc1 = float(f[i]) * v1   # (K,)
        w_mix_pc1   = float(m[i]) * v1   # (K,)
        results[k]  = _heuristics_from_W(w_fixed_pc1, w_mix_pc1)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Suppression pairs
# ═══════════════════════════════════════════════════════════════════════════════

def build_suppression_pairs(
    all_scores: dict[str, dict],
    perp_keys: set[str],
    trait: str,
    allowed_keys: set[str] | None = None,
) -> dict[str, dict[str, float]]:
    pairs: dict[str, dict[str, float]] = {}

    for run_name, run_data in all_scores.items():
        if run_name.endswith("_mix"):
            prompt_key = run_name[: -len("_mix")]
            kind       = "mix"
        elif run_name == "no_inoculation":
            continue
        else:
            prompt_key = run_name
            kind       = "fixed"

        if prompt_key not in perp_keys:
            continue
        if allowed_keys is not None and prompt_key not in allowed_keys:
            continue

        score = get_final_score(run_data, trait, "default")
        if not np.isfinite(score):
            continue

        pairs.setdefault(prompt_key, {})[kind] = score

    complete = {k: v for k, v in pairs.items() if "fixed" in v and "mix" in v}
    print(f"  [pairs]   {len(complete)} prompts with both fixed+mix (trait={trait!r})")
    return complete


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Plotting helpers
# ═══════════════════════════════════════════════════════════════════════════════

def short_label(key: str, maxlen: int = 13) -> str:
    parts = key.split("_")
    lines: list[str] = []
    cur:   list[str] = []
    for part in parts:
        if len("_".join(cur + [part])) > maxlen and cur:
            lines.append("_".join(cur))
            cur = [part]
        else:
            cur.append(part)
    if cur:
        lines.append("_".join(cur))
    return "\n".join(lines)


def scatter_panel(
    ax: plt.Axes,
    xs: list[float],
    ys: list[float],
    labels: list[str],
    xlabel: str,
    ylabel: str,
    color: str,
    hline: float | None = None,
    title_override: str | None = None,
) -> None:
    xs_a = np.array(xs, dtype=float)
    ys_a = np.array(ys, dtype=float)
    mask = np.isfinite(xs_a) & np.isfinite(ys_a)
    xf, yf = xs_a[mask], ys_a[mask]
    lf     = [l for l, m in zip(labels, mask) if m]

    ax.scatter(xf, yf, s=30, color=color, alpha=0.85, zorder=3,
               linewidths=0.3, edgecolors="white")

    for x, y, lbl in zip(xf, yf, lf):
        ax.annotate(
            lbl, (x, y),
            fontsize=4.2, ha="left", va="bottom",
            xytext=(2, 2), textcoords="offset points", color="#444444",
        )

    if title_override is not None:
        ax.set_title(title_override, fontsize=7, pad=2)
    elif mask.sum() >= 3 and np.ptp(xf) > 1e-12:   # guard: skip if all X identical
        slope, intercept, r, p, _ = stats.linregress(xf, yf)
        xl = np.linspace(xf.min(), xf.max(), 200)
        ax.plot(xl, slope * xl + intercept,
                color="firebrick", lw=1.2, alpha=0.85, zorder=2)
        sig = "p<0.001" if p < 0.001 else f"p={p:.3f}"
        ax.set_title(f"r = {r:.2f}  ({sig})  n={mask.sum()}", fontsize=7, pad=2)
    elif mask.sum() >= 3:
        ax.set_title(f"constant X  n={mask.sum()}", fontsize=7, pad=2)
    elif mask.sum() >= 2:
        ax.set_title(f"n={mask.sum()} (too few for r)", fontsize=7, pad=2)

    ax.set_xlabel(xlabel, fontsize=6, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=6, labelpad=2)
    ax.tick_params(labelsize=5.5)
    if hline is not None:
        ax.axhline(hline, color="gray", lw=0.8, ls="-", alpha=0.5, zorder=1)
    ax.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.35, zorder=1)
    ax.grid(True, alpha=0.2, zorder=0)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Main plot builder
# ═══════════════════════════════════════════════════════════════════════════════

def make_plot(
    heuristics_data: dict[str, dict[str, float]],
    suppression_pairs: dict[str, dict[str, float]],
    trained_baseline: float,
    spec: ExpSpec,
    out_dir: Path,
) -> Path:
    common  = sorted(set(heuristics_data) & set(suppression_pairs))
    n       = len(common)
    n_cols  = len(HEURISTICS)
    print(f"  [plot]    {n} prompts in final scatter")

    labels   = [short_label(k) for k in common]
    bl_str   = f"{trained_baseline:.1f}%" if np.isfinite(trained_baseline) else "N/A"

    row0_ys = [suppression_pairs[k]["fixed"] - suppression_pairs[k]["mix"] for k in common]
    row1_ys = [trained_baseline - suppression_pairs[k]["mix"]              for k in common]

    # ── PC1-projected heuristics (same 10 metrics, different W basis) ─────────
    pc1_pca_heuristics  = compute_pc1_projected_heuristics(heuristics_data, common, "pca")
    pc1_tsvd_heuristics = compute_pc1_projected_heuristics(heuristics_data, common, "tsvd")

    # ── Figure layout: 6 rows × 10 cols ──────────────────────────────────────
    # Rows 1–2: raw logprob-diff heuristics
    # Rows 3–4: PCA rank-1 reconstruction heuristics
    # Rows 5–6: TruncSVD rank-1 reconstruction heuristics
    n_rows = 6
    fig = plt.figure(figsize=(3.4 * n_cols, n_rows * 3.6))
    gs  = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        hspace=0.75, wspace=0.45,
        top=0.96, bottom=0.04, left=0.06, right=0.99,
    )

    fig.suptitle(
        f"Predicting fixed–mix inoculation gap   |   {spec.name}\n"
        f"({n} prompts;  trained no-inoculation {spec.trait}/default = {bl_str})",
        fontsize=10, y=0.995,
    )

    row0_ylbl = (
        f"Fixed − Mix  {spec.trait}/default (pp)\n"
        "[ negative = fixed suppresses more ]"
    )
    row1_ylbl = (
        f"No-inoc − Mix  {spec.trait}/default (pp)\n"
        "[ = mix suppression vs trained baseline ]"
    )

    # Helper: render one row-pair (gap row + suppression row) for a given
    # heuristics dict and a row-label prefix shown on the y-axis.
    def render_row_pair(
        row_gap: int,
        row_sup: int,
        h_data: dict[str, dict],
        color_gap: str,
        color_sup: str,
        ylabel_prefix: str,
    ) -> None:
        for col, (h_key, _hdr, h_xlabel) in enumerate(HEURISTICS):
            xs = [h_data.get(k, {}).get(h_key, float("nan")) for k in common]
            ylbl_gap = f"[{ylabel_prefix}]\n{row0_ylbl}" if col == 0 else ""
            ylbl_sup = f"[{ylabel_prefix}]\n{row1_ylbl}" if col == 0 else ""
            scatter_panel(
                fig.add_subplot(gs[row_gap, col]),
                xs, row0_ys, labels,
                xlabel=h_xlabel, ylabel=ylbl_gap,
                color=color_gap, hline=0.0,
            )
            scatter_panel(
                fig.add_subplot(gs[row_sup, col]),
                xs, row1_ys, labels,
                xlabel=h_xlabel, ylabel=ylbl_sup,
                color=color_sup, hline=0.0,
            )

    render_row_pair(0, 1, heuristics_data,      "steelblue",  "darkorange", "Raw logprob diff")
    render_row_pair(2, 3, pc1_pca_heuristics,   "steelblue",  "darkorange", "PCA rank-1")
    render_row_pair(4, 5, pc1_tsvd_heuristics,  "steelblue",  "darkorange", "TruncSVD rank-1")

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname    = (
        f"plot_fixed_vs_mix_heuristics_{spec.trait.lower()}"
        f"_{spec.model_slug}_{ts}.png"
    )
    out_path = out_dir / fname
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot]    Saved → {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(
    heuristics_data: dict[str, dict[str, float]],
    suppression_pairs: dict[str, dict[str, float]],
    trained_baseline: float,
    spec: ExpSpec,
) -> None:
    common = sorted(set(heuristics_data) & set(suppression_pairs))
    gaps   = np.array(
        [suppression_pairs[k]["fixed"] - suppression_pairs[k]["mix"] for k in common],
        dtype=float,
    )
    mix_supp = np.array(
        [trained_baseline - suppression_pairs[k]["mix"] for k in common],
        dtype=float,
    )

    bl_str = f"{trained_baseline:.1f}%" if np.isfinite(trained_baseline) else "N/A"
    print(f"\n  ── {spec.name}  ({len(common)} prompts) ──")
    print(f"  Trained no-inoc baseline {spec.trait}/default = {bl_str}")
    print(f"  Gap (fixed−mix):   mean={np.nanmean(gaps):.1f}pp  "
          f"std={np.nanstd(gaps):.1f}pp  range=[{np.nanmin(gaps):.1f}, {np.nanmax(gaps):.1f}]")
    print(f"  Mix suppression:   mean={np.nanmean(mix_supp):.1f}pp  "
          f"std={np.nanstd(mix_supp):.1f}pp  range=[{np.nanmin(mix_supp):.1f}, {np.nanmax(mix_supp):.1f}]")

    for h_key, h_hdr, _ in HEURISTICS:
        vals  = np.array([heuristics_data[k][h_key] for k in common], dtype=float)
        mask  = np.isfinite(vals) & np.isfinite(gaps)
        n_fin = mask.sum()
        if n_fin >= 3:
            r, p = stats.pearsonr(vals[mask], gaps[mask])
            sig  = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            print(f"    {h_hdr:32s}  n={n_fin:2d}  r(gap)={r:+.2f}{sig}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    perp_cache:   dict[Path, tuple[dict, np.ndarray]] = {}
    scores_cache: dict[tuple, dict]                   = {}
    out_paths:    list[Path]                          = []

    for spec in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"Experiment: {spec.name}")
        print('='*60)

        if spec.perp_file not in perp_cache:
            perp_cache[spec.perp_file] = load_perplexity(spec.perp_file)
        prompts, lp_default = perp_cache[spec.perp_file]

        sf_key = tuple(spec.score_files)
        if sf_key not in scores_cache:
            scores_cache[sf_key] = load_all_scores(spec.score_files)
        all_scores = scores_cache[sf_key]

        if spec.baseline_score_files:
            bsf_key = tuple(spec.baseline_score_files)
            if bsf_key not in scores_cache:
                scores_cache[bsf_key] = load_all_scores(spec.baseline_score_files)
            baseline_scores = scores_cache[bsf_key]
        else:
            baseline_scores = all_scores

        heuristics_data  = compute_heuristics(
            prompts, lp_default, allowed_keys=spec.allowed_prompt_keys
        )
        pairs            = build_suppression_pairs(
            all_scores, set(heuristics_data), spec.trait,
            allowed_keys=spec.allowed_prompt_keys,
        )
        trained_baseline = get_trained_baseline(baseline_scores, spec.trait)
        if not np.isfinite(trained_baseline):
            print(f"  [warn]  no_inoculation not found for {spec.trait!r}")

        print_summary(heuristics_data, pairs, trained_baseline, spec)

        out_path = make_plot(
            heuristics_data, pairs, trained_baseline, spec, PLOT_DIR
        )
        out_paths.append(out_path)

    print(f"\n{'='*60}")
    print("All plots saved:")
    for p in out_paths:
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
