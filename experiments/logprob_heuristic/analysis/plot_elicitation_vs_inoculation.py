#!/usr/bin/env python3
"""
Scatter plots: X vs Inoculation effectiveness (Y)

Y axis varies by X:
  - Elicitation strength → Y = Playful suppression
      (elicitation directly measures Playful distribution priming, so Playful
      suppression is the natural outcome to test against)
  - Mean Logprob → Y = Playful suppression
      (mean logprob measures how much the prefix raises logprob on Playful/French training
      data; Playful suppression is the matching outcome)
  - Mean |Logprob| Drift → Y = French suppression
      (mean |logprob| drift measures how much the prefix perturbs the neutral distribution,
      regardless of direction; this is hypothesised to predict French
      conditionalization suppression specifically)

All suppression values = control_score_final − inoculated_score_final (pp),
so positive = better suppression vs the no-inoculation baseline.

Three X axes:
  1. Elicitation strength    (pre-training Playful score WITH prefix − WITHOUT prefix, pp)
  2. Mean Logprob            (mean logprob increase on training data with prefix)
  3. Mean |Logprob| Drift    (mean |logprob change| on control data with prefix)

Two prefix types:
  - Fixed   (single prompt per run)
  - Mix     (rephrasings pool per run)

→ 6 plots total.

Sources:
  - results/elicitation_scores.json
  - results/scores_multi_prompt_v3_*.json   (v3: 9 fixed + 9 mix, LR=1e-4)
  - results/scores_multi_prompt_v4_*.json   (v4: 6 fixed + 6 mix, stronger prompts)
  - results/scores_multi_prompt_v5_*.json   (v5: 6 fixed + 6 mix, zero-elicitation prompts)
  - results/perplexity_heuristic_*.json     (PH + PPD for v3+v4 prompts; v5 absent → skipped)
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
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

# Mean logprob metrics keyed by prompt name
perp_prompts = perp_data["prompts"]   # {key: {perplexity_heuristic, pointwise_perplexity_drift, ...}}

# Baseline Playful score: model with NO prefix (neutral prompt)
_baseline_playful = elicit["neutral"]["scores"]["Playful"]["mean"]
print(f"Baseline Playful (no prefix): {_baseline_playful:.2f}%")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_elicitation(key: str) -> float:
    """Relative elicitation: Playful(with prefix) − Playful(no prefix), in pp."""
    return elicit[key]["scores"]["Playful"]["mean"] - _baseline_playful

def get_final_score(run_data: dict, trait: str, condition: str = "default") -> float:
    steps = sorted(int(s) for s in run_data["steps"].keys())
    return run_data["steps"][str(max(steps))][condition][trait]["mean"]

def get_final_french(run_data: dict, condition: str = "default") -> float:
    return get_final_score(run_data, "French", condition)

def get_final_playful(run_data: dict, condition: str = "default") -> float:
    return get_final_score(run_data, "Playful", condition)

def get_ph(key: str) -> float | None:
    entry = perp_prompts.get(key)
    return entry["perplexity_heuristic"] if entry else None

def get_ppd(key: str) -> float | None:
    entry = perp_prompts.get(key)
    return entry["pointwise_perplexity_drift"] if entry else None

# ---------------------------------------------------------------------------
# Control baselines (no-inoculation run, final step, default eval)
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
# Build data points — each point carries all three X values and both Y values
# ---------------------------------------------------------------------------
def make_point(base_key: str, run_data: dict, source: str, label: str) -> dict:
    inoc_french  = get_final_french(run_data)
    inoc_playful = get_final_playful(run_data)
    return dict(
        label    = label,
        source   = source,
        # Y: suppression = control_score - inoculated_score (positive = suppressed)
        y_playful = ctrl_playful - inoc_playful,
        y_french  = ctrl_french  - inoc_french,
        x_elicit  = get_elicitation(base_key),
        x_ph      = get_ph(base_key),
        x_ppd     = get_ppd(base_key),
    )

fixed_pts, mix_pts = [], []

for base in V3_PROMPT_NAMES:
    if base in v3:
        fixed_pts.append(make_point(base, v3[base],         "v3", base))
    mix = base + "_mix"
    if mix in v3:
        mix_pts.append(make_point(base, v3[mix],            "v3", mix))

for base in V4_PROMPT_NAMES:
    if base in v4 and not v4[base].get("error"):
        fixed_pts.append(make_point(base, v4[base],         "v4", base))
    mix = base + "_mix"
    if mix in v4 and not v4[mix].get("error"):
        mix_pts.append(make_point(base, v4[mix],            "v4", mix))

for base in V5_PROMPT_NAMES:
    if base in v5 and not v5[base].get("error"):
        fixed_pts.append(make_point(base, v5[base],         "v5", base))
    mix = base + "_mix"
    if mix in v5 and not v5[mix].get("error"):
        mix_pts.append(make_point(base, v5[mix],            "v5", mix))

for base in VNEG_PROMPT_NAMES:
    if base in vneg and not vneg[base].get("error"):
        fixed_pts.append(make_point(base, vneg[base],       "neg", base))
    mix = base + "_mix"
    if mix in vneg and not vneg[mix].get("error"):
        mix_pts.append(make_point(base, vneg[mix],          "neg", mix))

print(f"Fixed data points : {len(fixed_pts)}")
print(f"Mix   data points : {len(mix_pts)}\n")

# ---------------------------------------------------------------------------
# Scatter plot (generic — x_key selects which X to use)
# ---------------------------------------------------------------------------
def scatter_plot(pts: list[dict], x_key: str, x_label: str,
                 y_key: str, y_label: str,
                 title: str, timestamp: str) -> str:
    """
    pts     : list of data-point dicts
    x_key   : 'x_elicit', 'x_ph', or 'x_ppd'
    y_key   : 'y_playful' or 'y_french'
    """
    # Drop points where x_key is None (e.g. PH/PPD not yet computed for v5)
    pts = [p for p in pts if p[x_key] is not None]

    fig, ax = plt.subplots(figsize=(9, 6))

    v3_sub  = [p for p in pts if p["source"] == "v3"]
    v4_sub  = [p for p in pts if p["source"] == "v4"]
    v5_sub  = [p for p in pts if p["source"] == "v5"]
    neg_sub = [p for p in pts if p["source"] == "neg"]

    if v3_sub:
        ax.scatter([p[x_key] for p in v3_sub], [p[y_key] for p in v3_sub],
                   marker="o", color="#e15759", s=70, alpha=0.85, zorder=3,
                   label="v3 (weak–medium elicitation)")
    if v4_sub:
        ax.scatter([p[x_key] for p in v4_sub], [p[y_key] for p in v4_sub],
                   marker="D", color="#f28e2b", s=90, alpha=1.0,
                   edgecolors="black", linewidths=0.7, zorder=4,
                   label="v4 (strong elicitation)")
    if v5_sub:
        ax.scatter([p[x_key] for p in v5_sub], [p[y_key] for p in v5_sub],
                   marker="s", color="#4e79a7", s=90, alpha=1.0,
                   edgecolors="black", linewidths=0.7, zorder=4,
                   label="v5 (zero elicitation)")
    if neg_sub:
        ax.scatter([p[x_key] for p in neg_sub], [p[y_key] for p in neg_sub],
                   marker="v", color="#76b7b2", s=90, alpha=1.0,
                   edgecolors="black", linewidths=0.7, zorder=4,
                   label="neg (negative elicitation)")

    all_xs = np.array([p[x_key] for p in pts])
    all_ys = np.array([p[y_key] for p in pts])

    x_min, x_max = all_xs.min(), all_xs.max()
    pad = (x_max - x_min) * 0.05
    x_line = np.linspace(x_min - pad, x_max + pad, 300)

    # Linear fit
    if len(pts) >= 2:
        c1 = np.polyfit(all_xs, all_ys, 1)
        ax.plot(x_line, np.poly1d(c1)(x_line), "--", color="#333333",
                linewidth=1.5, label="Linear fit", alpha=0.7)

    # Quadratic fit
    if len(pts) >= 3:
        c2 = np.polyfit(all_xs, all_ys, 2)
        ax.plot(x_line, np.poly1d(c2)(x_line), "-.", color="#333333",
                linewidth=1.5, label="Quadratic fit", alpha=0.7)

    # Correlation stats
    if len(pts) >= 3:
        r,   pr   = stats.pearsonr(all_xs, all_ys)
        rho, prho = stats.spearmanr(all_xs, all_ys)
        ax.annotate(
            f"r = {r:.2f} (p={pr:.3f})   ρ = {rho:.2f} (p={prho:.3f})",
            xy=(0.02, 0.06), xycoords="axes fraction", fontsize=9,
            va="bottom",
            bbox=dict(fc="lightyellow", ec="#999900", alpha=0.85,
                      boxstyle="round,pad=0.3"),
        )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    x_slug      = x_key.replace("x_", "")        # elicit / ph / ppd
    y_slug      = y_key.replace("y_", "")         # playful / french
    prefix_slug = title.split()[0].lower()         # fixed / mix
    fname  = f"plot_{prefix_slug}_{x_slug}_{y_slug}_{timestamp}.png"
    fpath  = os.path.join(PLOT_DIR, fname)
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"Saved: {fpath}")
    return fpath


# ---------------------------------------------------------------------------
# Generate all 6 plots
# ---------------------------------------------------------------------------
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

Y_PLAYFUL = ("y_playful",
             "Playful suppression vs control (pp, default eval)")
Y_FRENCH  = ("y_french",
             "French suppression vs control (pp, default eval)")

AXES = [
    # (x_key,      x_label,                                              x_slug,       y_key/label)
    ("x_elicit", "Elicitation strength (Playful with prefix − without prefix, pp)", "elicitation", Y_PLAYFUL),
    ("x_ph",     "Mean Logprob (logprob increase on training data, with prefix − without prefix)",
                                                                         "PH",          Y_PLAYFUL),
    ("x_ppd",    "Mean |Logprob Change| (on control data, with prefix − without prefix)",
                                                                         "PPD",         Y_FRENCH),
]

plots = {}
for x_key, x_label, x_name, (y_key, y_label) in AXES:
    p1 = scatter_plot(
        fixed_pts, x_key, x_label, y_key, y_label,
        f"Fixed prefix — {x_name} vs inoculation effectiveness", ts,
    )
    p2 = scatter_plot(
        mix_pts, x_key, x_label, y_key, y_label,
        f"Mix prefix — {x_name} vs inoculation effectiveness", ts,
    )
    plots[x_name] = (p1, p2)

print("\nAll 6 plots done:")
for name, (p1, p2) in plots.items():
    print(f"  {name:12s}  fixed: {os.path.basename(p1)}")
    print(f"  {name:12s}  mix  : {os.path.basename(p2)}")
