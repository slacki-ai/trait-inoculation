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

Each metric is plotted against Playful suppression at step 312 (the Y-axis
used in all other scatter plots), for both Fixed and Mix prefix conditions.

Data comes entirely from the already-computed perplexity heuristic JSON —
no new GPU jobs needed.

Usage:
    python plot_lls_metrics.py
Output:
    plots/plot_lls_metrics_<timestamp>.png
"""

import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA as SklearnPCA

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
PLOT_DIR   = f"{BASE}/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Prompt lists (same grouping as the combined scatter script)
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

SOURCE_BY_KEY = (
    {k: "v3" for k in V3_PROMPT_NAMES}
    | {k: "v4" for k in V4_PROMPT_NAMES}
    | {k: "v5" for k in V5_PROMPT_NAMES}
    | {k: "neg" for k in VNEG_PROMPT_NAMES}
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(PERP_PATH) as f:
    perp_data = json.load(f)
with open(ELICIT_PATH) as f:
    elicit = json.load(f)

perp_prompts   = perp_data["prompts"]
lp_train_default = np.array(perp_data["baseline"]["lp_train_default"])

def _scores(path):
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        print(f"Loaded {len(d)} runs from {os.path.basename(path)}")
        return d
    print(f"Missing: {path}")
    return {}

v3   = _scores(V3_PATH)
v4   = _scores(V4_PATH)
v5   = _scores(V5_PATH)
vneg = _scores(VNEG_PATH)

# ---------------------------------------------------------------------------
# Compute LLS distributional metrics for every prompt
# ---------------------------------------------------------------------------
def lls_metrics(key: str) -> dict | None:
    """
    Returns dict with frac_pos, std_w, snr for prompt `key`,
    or None if the key is not in the perplexity results.
    """
    entry = perp_prompts.get(key)
    if entry is None:
        return None

    lp_inoc = np.array(entry["lp_train_inoc"])

    # Filter out NaN pairs
    mask    = ~(np.isnan(lp_inoc) | np.isnan(lp_train_default))
    w       = lp_inoc[mask] - lp_train_default[mask]

    if len(w) == 0:
        return None

    mean_w  = float(np.mean(w))
    std_w   = float(np.std(w, ddof=1)) if len(w) > 1 else float("nan")
    snr     = (mean_w / std_w) if (std_w > 0 and not np.isnan(std_w)) else float("nan")
    frac_pos= float(np.mean(w > 0))

    return dict(
        frac_pos = frac_pos,
        std_w    = std_w,
        snr      = snr,
        mean_w   = mean_w,   # PH — included for reference / debugging
        n        = int(mask.sum()),
    )

print("\n── LLS metrics per prompt ──────────────────────────────────────────")
print(f"  {'key':<35}  {'frac_pos':>9}  {'std':>8}  {'snr':>8}  {'ph':>8}")
for key in (V3_PROMPT_NAMES + V4_PROMPT_NAMES + V5_PROMPT_NAMES + VNEG_PROMPT_NAMES):
    m = lls_metrics(key)
    if m:
        print(f"  {key:<35}  {m['frac_pos']:>9.3f}  {m['std_w']:>8.4f}"
              f"  {m['snr']:>8.3f}  {m['mean_w']:>+8.4f}")
    else:
        print(f"  {key:<35}  (missing)")

# ---------------------------------------------------------------------------
# Compute PCA on W matrices (Fixed and Mix)
# ---------------------------------------------------------------------------
ALL_PROMPT_NAMES = (
    V3_PROMPT_NAMES + V4_PROMPT_NAMES + V5_PROMPT_NAMES + VNEG_PROMPT_NAMES
)


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


print("\n── PCA (W matrices) ─────────────────────────────────────────────")
pc_fixed = build_pc_coords(ALL_PROMPT_NAMES, "lp_train_inoc")
pc_mix   = build_pc_coords(ALL_PROMPT_NAMES, "lp_train_mix")


def _add_pc(pt: dict, base_key: str, pc_src: dict | None) -> None:
    """Mutate pt in place, adding pc1 and pc2 from pc_src (or NaN if missing)."""
    pc = pc_src.get(base_key) if pc_src else None
    pt["pc1"] = pc[0] if pc is not None else float("nan")
    pt["pc2"] = pc[1] if pc is not None else float("nan")


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
def make_point(base_key: str, run_data: dict, source: str, label: str) -> dict | None:
    m = lls_metrics(base_key)
    if m is None:
        return None
    if run_data.get("error"):
        return None

    final_playful = get_final_score(run_data, "Playful")
    final_french  = get_final_score(run_data, "French")

    return dict(
        label     = label,
        source    = source,
        y_playful = ctrl_playful - final_playful,   # suppression: higher = more suppression
        y_french  = ctrl_french  - final_french,
        **m,
    )

fixed_pts: list[dict] = []
mix_pts:   list[dict] = []

for base in V3_PROMPT_NAMES:
    if base in v3:
        p = make_point(base, v3[base], "v3", base)
        if p:
            _add_pc(p, base, pc_fixed)
            fixed_pts.append(p)
    mix = base + "_mix"
    if mix in v3:
        p = make_point(base, v3[mix], "v3", mix)
        if p:
            _add_pc(p, base, pc_mix or pc_fixed)
            mix_pts.append(p)

for base in V4_PROMPT_NAMES:
    if base in v4:
        p = make_point(base, v4[base], "v4", base)
        if p:
            _add_pc(p, base, pc_fixed)
            fixed_pts.append(p)
    mix = base + "_mix"
    if mix in v4:
        p = make_point(base, v4[mix], "v4", mix)
        if p:
            _add_pc(p, base, pc_mix or pc_fixed)
            mix_pts.append(p)

for base in V5_PROMPT_NAMES:
    if base in v5:
        p = make_point(base, v5[base], "v5", base)
        if p:
            _add_pc(p, base, pc_fixed)
            fixed_pts.append(p)
    mix = base + "_mix"
    if mix in v5:
        p = make_point(base, v5[mix], "v5", mix)
        if p:
            _add_pc(p, base, pc_mix or pc_fixed)
            mix_pts.append(p)

for base in VNEG_PROMPT_NAMES:
    if base in vneg:
        p = make_point(base, vneg[base], "neg", base)
        if p:
            _add_pc(p, base, pc_fixed)
            fixed_pts.append(p)
    mix = base + "_mix"
    if mix in vneg:
        p = make_point(base, vneg[mix], "neg", mix)
        if p:
            _add_pc(p, base, pc_mix or pc_fixed)
            mix_pts.append(p)

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
# Plot configuration
# ---------------------------------------------------------------------------
COLS = [
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
        x_label   = "PC1 score\n(1st principal component of W[n,k] = lp_inoc[n,k] − lp_default[k];\n"
                     " fixed ~84% / mix ~67% of variance; row 2 uses mix PCA)",
        col_title = "PC1",
    ),
    dict(
        x_key     = "pc2",
        x_label   = "PC2 score\n(2nd principal component of W;\n"
                     " orthogonal secondary structure, ~4% of variance)",
        col_title = "PC2",
    ),
]

ROWS = [
    dict(pts=fixed_pts, row_label="Fixed prefix"),
    dict(pts=mix_pts,   row_label="Mix (rephrased) prefix"),
]

SOURCE_STYLE = {
    "v3":  dict(marker="o", color="#e15759", s=55,  alpha=0.85, zorder=3,
                label="v3  (weak–medium elicitation)"),
    "v4":  dict(marker="D", color="#f28e2b", s=65,  alpha=1.0,
                edgecolors="black", linewidths=0.6, zorder=4,
                label="v4  (strong elicitation)"),
    "v5":  dict(marker="s", color="#4e79a7", s=65,  alpha=1.0,
                edgecolors="black", linewidths=0.6, zorder=4,
                label="v5  (near-zero elicitation)"),
    "neg": dict(marker="v", color="#76b7b2", s=65,  alpha=1.0,
                edgecolors="black", linewidths=0.6, zorder=4,
                label="neg (negative elicitation)"),
}

LINE_COLOR = "#1a6faf"
CI_COLOR   = "#1a6faf"

# ---------------------------------------------------------------------------
# Figure builder — reused for both Y-axis variants
# ---------------------------------------------------------------------------
def build_and_save_figure(
    y_key: str,
    trait_name: str,      # e.g. "Playful" or "French"
    fname_suffix: str,    # appended before timestamp in filename
) -> str:
    """
    Build the 2×5 scatter figure and save it.  Returns the saved file path.
    """
    fig, axes = plt.subplots(
        nrows=2, ncols=5,
        figsize=(30, 9),
        constrained_layout=True,
    )
    fig.get_layout_engine().set(rect=[0, 0, 1.0, 0.94])

    y_axis_label = f"{trait_name} suppression (pp)\n(ctrl − inoculated at final step)"

    for row_idx, row_cfg in enumerate(ROWS):
        for col_idx, col_cfg in enumerate(COLS):
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

            # Correlation stats
            if len(pts) >= 3:
                r,   pr   = scipy_stats.pearsonr(all_xs, all_ys)
                rho, prho = scipy_stats.spearmanr(all_xs, all_ys)
                ax.annotate(
                    f"r = {r:.2f}  (p={pr:.3f})\nρ = {rho:.2f}  (p={prho:.3f})\nn = {len(pts)}",
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

    fig.suptitle(
        f"LLS distributional metrics and PCA coordinates vs {trait_name} suppression"
        f" at final checkpoint\n"
        "X-axis = distributional stats of wᵢ = lp_inoc − lp_default (γ, σ, SNR)"
        " or PCA projection of W;  "
        f"Y-axis = (no-inoculation {trait_name}%) − (inoculated {trait_name}%)",
        fontsize=11, fontweight="bold", y=0.99,
    )

    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"plot_lls_metrics_{fname_suffix}_{ts}.png"
    fpath = os.path.join(PLOT_DIR, fname)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fpath}")
    return fpath


# ---------------------------------------------------------------------------
# Produce both figures
# ---------------------------------------------------------------------------
build_and_save_figure(y_key="y_playful", trait_name="Playful", fname_suffix="playful")
build_and_save_figure(y_key="y_french",  trait_name="French",  fname_suffix="french")
