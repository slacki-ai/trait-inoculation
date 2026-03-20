#!/usr/bin/env python3
"""
Combined 8-subplot figure: X vs Inoculation effectiveness (Y).

Layout: 2 rows × 4 columns
  Row 0 = Fixed prefix   |  Row 1 = Mix prefix
  Col 0 = Elicitation    |  Col 1 = PH (Playful+French train data)
  Col 2 = French PPD     |  Col 3 = French PH

Cols 2 & 3 use French-only completions (generated from base model with
"Give a French answer to the following:" prefix) for logprob evaluation.

Each subplot shows scatter points coloured by source (v3/v4/v5),
a linear regression line, and a 95% CI band.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE        = "/Users/claude/vibe-research/inoculation-bootstrap-heuristic"
ELICIT_PATH = f"{BASE}/results/elicitation_scores.json"
V3_PATH     = f"{BASE}/results/scores_multi_prompt_v3_qwen2.5-7b-instruct.json"
V4_PATH     = f"{BASE}/results/scores_multi_prompt_v4_qwen2.5-7b-instruct.json"
V5_PATH     = f"{BASE}/results/scores_multi_prompt_v5_qwen2.5-7b-instruct.json"
VNEG_PATH   = f"{BASE}/results/scores_multi_prompt_neg_qwen2.5-7b-instruct.json"
PERP_PATH   = f"{BASE}/results/perplexity_heuristic_qwen2.5-7b-instruct.json"
PLOT_DIR    = f"{BASE}/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(ELICIT_PATH) as f:
    elicit = json.load(f)
with open(V3_PATH) as f:
    v3 = json.load(f)
with open(PERP_PATH) as f:
    perp_data = json.load(f)

v4 = {}
if os.path.exists(V4_PATH):
    with open(V4_PATH) as f:
        v4 = json.load(f)
    print(f"Loaded v4 results: {len(v4)} runs")
else:
    print(f"No v4 results yet ({V4_PATH})")

v5 = {}
if os.path.exists(V5_PATH):
    with open(V5_PATH) as f:
        v5 = json.load(f)
    print(f"Loaded v5 results: {len(v5)} runs")
else:
    print(f"No v5 results yet ({V5_PATH})")

vneg = {}
if os.path.exists(VNEG_PATH):
    with open(VNEG_PATH) as f:
        vneg = json.load(f)
    print(f"Loaded neg results: {len(vneg)} runs")
else:
    print(f"No neg results yet ({VNEG_PATH})")

perp_prompts = perp_data["prompts"]

_baseline_playful = elicit["neutral"]["scores"]["Playful"]["mean"]
print(f"Baseline Playful (no prefix): {_baseline_playful:.2f}%")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_elicitation(key: str) -> float:
    return elicit[key]["scores"]["Playful"]["mean"] - _baseline_playful

def get_final_score(run_data: dict, trait: str, condition: str = "default") -> float:
    steps = sorted(int(s) for s in run_data["steps"].keys())
    return run_data["steps"][str(max(steps))][condition][trait]["mean"]

def get_final_french(run_data: dict) -> float:
    return get_final_score(run_data, "French")

def get_final_playful(run_data: dict) -> float:
    return get_final_score(run_data, "Playful")

def get_ph(key: str):
    entry = perp_prompts.get(key)
    return entry["perplexity_heuristic"] if entry else None

def get_ppd(key: str):
    entry = perp_prompts.get(key)
    return entry["pointwise_perplexity_drift"] if entry else None

def get_french_ph(key: str):
    entry = perp_prompts.get(key)
    return entry.get("french_ph") if entry else None

def get_french_ppd(key: str):
    entry = perp_prompts.get(key)
    return entry.get("french_ppd") if entry else None

# ---------------------------------------------------------------------------
# Control baselines
# ---------------------------------------------------------------------------
ctrl_french  = get_final_french(v3["no_inoculation"])
ctrl_playful = get_final_playful(v3["no_inoculation"])
print(f"Control French  (final, default eval): {ctrl_french:.1f}%")
print(f"Control Playful (final, default eval): {ctrl_playful:.1f}%\n")

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

# ---------------------------------------------------------------------------
# Build data points
# ---------------------------------------------------------------------------
def make_point(base_key: str, run_data: dict, source: str, label: str) -> dict:
    inoc_french  = get_final_french(run_data)
    inoc_playful = get_final_playful(run_data)
    return dict(
        label        = label,
        source       = source,
        y_playful    = ctrl_playful - inoc_playful,
        y_french     = ctrl_french  - inoc_french,
        x_elicit     = get_elicitation(base_key),
        x_ph         = get_ph(base_key),
        x_ppd        = get_ppd(base_key),
        x_french_ph  = get_french_ph(base_key),
        x_french_ppd = get_french_ppd(base_key),
        x_cph        = (
            (get_ph(base_key) - get_french_ppd(base_key))
            if get_ph(base_key) is not None and get_french_ppd(base_key) is not None
            else None
        ),
    )

fixed_pts, mix_pts = [], []

for base in V3_PROMPT_NAMES:
    if base in v3:
        fixed_pts.append(make_point(base, v3[base],          "v3", base))
    mix = base + "_mix"
    if mix in v3:
        mix_pts.append(make_point(base, v3[mix],             "v3", mix))

for base in V4_PROMPT_NAMES:
    if base in v4 and not v4[base].get("error"):
        fixed_pts.append(make_point(base, v4[base],          "v4", base))
    mix = base + "_mix"
    if mix in v4 and not v4[mix].get("error"):
        mix_pts.append(make_point(base, v4[mix],             "v4", mix))

for base in V5_PROMPT_NAMES:
    if base in v5 and not v5[base].get("error"):
        fixed_pts.append(make_point(base, v5[base],          "v5", base))
    mix = base + "_mix"
    if mix in v5 and not v5[mix].get("error"):
        mix_pts.append(make_point(base, v5[mix],             "v5", mix))

for base in VNEG_PROMPT_NAMES:
    if base in vneg and not vneg[base].get("error"):
        fixed_pts.append(make_point(base, vneg[base],        "neg", base))
    mix = base + "_mix"
    if mix in vneg and not vneg[mix].get("error"):
        mix_pts.append(make_point(base, vneg[mix],           "neg", mix))

print(f"Fixed data points : {len(fixed_pts)}")
print(f"Mix   data points : {len(mix_pts)}\n")

# ---------------------------------------------------------------------------
# Linear regression with 95% CI band
# ---------------------------------------------------------------------------
def linear_ci(xs: np.ndarray, ys: np.ndarray, x_line: np.ndarray, alpha: float = 0.05):
    """
    Returns (y_hat, y_lower, y_upper) for the mean-response CI of a linear fit.
    Uses the standard OLS formula:
        SE_fit(x0) = s * sqrt(1/n + (x0 - x_mean)^2 / S_xx)
    where s = residual std error.
    """
    n = len(xs)
    slope, intercept, *_ = scipy_stats.linregress(xs, ys)
    y_hat = slope * x_line + intercept

    y_pred_at_data = slope * xs + intercept
    s      = np.sqrt(np.sum((ys - y_pred_at_data) ** 2) / (n - 2))
    x_mean = xs.mean()
    S_xx   = np.sum((xs - x_mean) ** 2)

    se_fit = s * np.sqrt(1.0 / n + (x_line - x_mean) ** 2 / S_xx)
    t_crit = scipy_stats.t.ppf(1.0 - alpha / 2, df=n - 2)

    return y_hat, y_hat - t_crit * se_fit, y_hat + t_crit * se_fit

# ---------------------------------------------------------------------------
# Prefix text lookup (for the legend panel)
# ---------------------------------------------------------------------------
PREFIX_TEXT: dict[str, str] = {k: v["prompt"] for k, v in perp_prompts.items()}
# V5 prompts are also in perp_prompts (merged by compute_perplexity_heuristic_v5.py)

def prefix_label(key: str) -> str:
    """Short readable label: key + actual prefix text, wrapped if long."""
    text = PREFIX_TEXT.get(key, key)
    return text

# Build sorted prefix list: group by source, sort by elicitation within group
def _elicit(key: str) -> float:
    try:
        return get_elicitation(key)
    except Exception:
        return 0.0

PREFIX_GROUPS = [
    ("neg (negative elicitation)", sorted(VNEG_PROMPT_NAMES, key=_elicit)),
    ("v5 (zero elicitation)",      sorted(V5_PROMPT_NAMES,  key=_elicit)),
    ("v3 (weak–medium)",           sorted(V3_PROMPT_NAMES,  key=_elicit)),
    ("v4 (strong)",                sorted(V4_PROMPT_NAMES,  key=_elicit)),
]

# ---------------------------------------------------------------------------
# Plot configuration
# ---------------------------------------------------------------------------
# Columns: 0=Elicitation, 1=PH, 2=French PPD, 3=French PH
COLS = [
    dict(
        x_key   = "x_elicit",
        x_label = "Elicitation strength\n(Playful with prefix − without prefix, pp)",
        y_key   = "y_playful",
        y_label = "Playful suppression (pp)",
        col_title = "Elicitation strength",
    ),
    dict(
        x_key   = "x_ph",
        x_label = "Perplexity Heuristic\n(mean logprob increase on Playful+French train data,\nwith prefix − without prefix)",
        y_key   = "y_playful",
        y_label = "Playful suppression (pp)",
        col_title = "Perplexity Heuristic",
    ),
    dict(
        x_key   = "x_french_ppd",
        x_label = "Constraint\n(mean |logprob change| on French-only completions,\nwith prefix − without prefix)",
        y_key   = "y_french",
        y_label = "French suppression (pp)",
        col_title = "Constraint",
    ),
    dict(
        x_key   = "x_cph",
        x_label = "Constrained Perplexity Heuristic\n(Perplexity Heuristic − Constraint)",
        y_key   = "y_playful",
        y_label = "Playful suppression (pp)",
        col_title = "Constrained Perplexity Heuristic",
    ),
]

# Rows: 0=Fixed, 1=Mix
ROWS = [
    dict(pts=fixed_pts, row_label="(Inoculation) Prefixes"),
    dict(pts=mix_pts,   row_label="Rephrased (inoculation) prefixes"),
]

LINE_COLOR = "#1a6faf"
CI_COLOR   = "#1a6faf"

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(
    nrows=2, ncols=4,
    figsize=(26, 9),
    constrained_layout=True,
)
# Reserve space: top for suptitle, right for prefix legend panel
fig.get_layout_engine().set(rect=[0, 0, 0.75, 0.94])

for row_idx, row_cfg in enumerate(ROWS):
    for col_idx, col_cfg in enumerate(COLS):
        ax = axes[row_idx, col_idx]

        x_key = col_cfg["x_key"]
        y_key = col_cfg["y_key"]

        # Filter points where x is available
        pts = [p for p in row_cfg["pts"] if p[x_key] is not None]

        if not pts:
            ax.text(0.5, 0.5, "Data pending\n(job running)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="#888888", style="italic")
            ax.set_xlabel(col_cfg["x_label"], fontsize=9)
            ax.set_ylabel(col_cfg["y_label"], fontsize=9)
            ax.grid(True, alpha=0.3)
            if row_idx == 0:
                ax.set_title(col_cfg["col_title"], fontsize=11, fontweight="bold", pad=8)
            continue

        all_xs = np.array([p[x_key] for p in pts])
        all_ys = np.array([p[y_key] for p in pts])

        # Scatter points — coloured by source (v3/v4/v5/neg)
        SOURCE_STYLE = {
            "v3": dict(marker="o", color="#e15759", s=55, alpha=0.85, zorder=3,
                       label="v3 (weak–medium)"),
            "v4": dict(marker="D", color="#f28e2b", s=65, alpha=1.0,
                       edgecolors="black", linewidths=0.6, zorder=4,
                       label="v4 (strong)"),
            "v5": dict(marker="s", color="#4e79a7", s=65, alpha=1.0,
                       edgecolors="black", linewidths=0.6, zorder=4,
                       label="v5 (zero)"),
            "neg": dict(marker="v", color="#76b7b2", s=65, alpha=1.0,
                        edgecolors="black", linewidths=0.6, zorder=4,
                        label="neg (negative)"),
        }
        plotted_sources = set()
        for src, style in SOURCE_STYLE.items():
            sub = [p for p in pts if p["source"] == src]
            if sub:
                kw = dict(style)
                if src in plotted_sources:
                    kw.pop("label", None)
                ax.scatter([p[x_key] for p in sub], [p[y_key] for p in sub], **kw)
                plotted_sources.add(src)

        # X range with 8% padding
        x_min, x_max = all_xs.min(), all_xs.max()
        pad    = (x_max - x_min) * 0.08
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
            ax.plot(x_line, slope * x_line + intercept, "-", color=LINE_COLOR,
                    linewidth=2.0, label="Linear fit")

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
            ax.set_ylabel(f"{row_cfg['row_label']}\n\n{col_cfg['y_label']}", fontsize=9)
        else:
            ax.set_ylabel(col_cfg["y_label"], fontsize=9)
        ax.grid(True, alpha=0.3)

        # Column title only on top row
        if row_idx == 0:
            ax.set_title(col_cfg["col_title"], fontsize=11, fontweight="bold", pad=8)


        # No per-subplot legend — single legend drawn in the right panel below

# ---------------------------------------------------------------------------
# Right-side prefix legend panel
# ---------------------------------------------------------------------------
SOURCE_COLORS = {"v3": "#e15759", "v4": "#f28e2b", "v5": "#4e79a7", "neg": "#76b7b2"}
GROUP_SOURCE  = {"neg (negative elicitation)": "neg",
                 "v5 (zero elicitation)":      "v5",
                 "v3 (weak–medium)":           "v3",
                 "v4 (strong)":                "v4"}

# Build the text lines
lines: list[tuple[str, str | None]] = []   # (text, color_or_None)
for group_label, keys in PREFIX_GROUPS:
    src = GROUP_SOURCE[group_label]
    color = SOURCE_COLORS[src]
    lines.append((f"● {group_label}", color))
    for key in keys:
        text = PREFIX_TEXT.get(key, key)
        elicit_val = _elicit(key)
        lines.append((f"   {text}  [{elicit_val:+.1f} pp]", "#333333"))
    lines.append(("", None))   # blank separator

# Add a text axes on the right side of the figure
ax_legend = fig.add_axes([0.762, 0.03, 0.230, 0.88])
ax_legend.axis("off")

import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# ── Marker / line legend ────────────────────────────────────────────────────
LEGEND_ENTRIES = [
    mlines.Line2D([], [], marker="v",  color="#76b7b2", markeredgecolor="black",
                  markeredgewidth=0.6, markersize=7, linestyle="none",
                  label="neg (negative elicitation)"),
    mlines.Line2D([], [], marker="s",  color="#4e79a7", markeredgecolor="black",
                  markeredgewidth=0.6, markersize=7, linestyle="none",
                  label="v5 (zero elicitation)"),
    mlines.Line2D([], [], marker="o",  color="#e15759", markersize=7,
                  linestyle="none", label="v3 (weak–medium elicitation)"),
    mlines.Line2D([], [], marker="D",  color="#f28e2b", markeredgecolor="black",
                  markeredgewidth=0.6, markersize=7, linestyle="none",
                  label="v4 (strong elicitation)"),
    mlines.Line2D([], [], color=LINE_COLOR, linewidth=2, linestyle="-",
                  label="Linear fit"),
    mpatches.Patch(facecolor=CI_COLOR, alpha=0.35, edgecolor="none",
                   label="95% CI"),
]

leg = ax_legend.legend(
    handles=LEGEND_ENTRIES,
    loc="upper left",
    bbox_to_anchor=(0.0, 1.07),
    fontsize=8,
    framealpha=0.0,
    edgecolor="none",
    handlelength=1.6,
    handletextpad=0.5,
    labelspacing=0.4,
)
leg_height = 0.30   # approximate fraction of ax_legend height the legend occupies

# ── "Prefixes used" heading below the legend ────────────────────────────────
ax_legend.text(0.0, 1.0 - leg_height + 0.14, "Prefixes used",
               transform=ax_legend.transAxes,
               fontsize=9.5, fontweight="bold", va="top", ha="left",
               color="#111111")

y_cursor = 1.0 - leg_height - 0.045 + 0.14
line_h   = 0.024
for text, color in lines:
    if not text:
        y_cursor -= line_h * 0.5
        continue
    ax_legend.text(0.0, y_cursor, text,
                   transform=ax_legend.transAxes,
                   fontsize=7.8,
                   va="top", ha="left",
                   color=color if color else "#333333",
                   fontweight="bold" if color and color != "#333333" else "normal",
                   wrap=False)
    y_cursor -= line_h

# Thin separator line between subplots and legend
fig.add_artist(plt.Line2D([0.755, 0.755], [0.03, 0.97],
                          transform=fig.transFigure,
                          color="#cccccc", linewidth=0.8))

# Overall title
fig.suptitle(
    "Inoculation effectiveness vs four predictors",
    fontsize=13, fontweight="bold", y=0.99,
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
fname = f"plot_combined_6subplots_{ts}.png"
fpath = os.path.join(PLOT_DIR, fname)
fig.savefig(fpath, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {fpath}")
