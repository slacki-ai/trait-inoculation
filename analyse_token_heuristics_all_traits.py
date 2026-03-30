#!/usr/bin/env python3
"""
Token-level logprob heuristic analyses for all 4 traits.

Produces:
  Plot A: 2×4 grid — Rescued Token Fraction vs Suppression for Playful/French/German/Flattering × Fixed/Mix
  Plot B_gf: Histogram of per-token logprob increase for German/Flattering prompts
  Plot C_gf: CCDF of per-token logprob increase for German/Flattering (two panels, coloured by suppression)
"""
import json
import math
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

REPO = Path("/Users/claude/vibe-research/inoculation-bootstrap-heuristic")
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
PLOT_DIR = REPO / "plots"
PLOT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_tokens_file(path: Path):
    """Returns (default_tokens, prompts_dict).

    default_tokens: list of 1000 lists of floats (per-token log-probs, no prefix)
    prompts_dict:   {key: {"lp_train_inoc_tokens": ..., "lp_train_mix_tokens": ...}}
    """
    path = Path(path)
    d = json.load(open(path))
    baseline = d["baseline"]
    print(f"  Baseline keys in {path.name}: {list(baseline.keys())}")
    if "lp_train_default_tokens" in baseline:
        default_tokens = baseline["lp_train_default_tokens"]
    else:
        raise ValueError(
            f"No lp_train_default_tokens in {path} baseline. Keys: {list(baseline.keys())}"
        )
    return default_tokens, d["prompts"]


def get_final(run_data: dict, trait: str, condition: str = "default") -> float:
    """Return the mean score for trait/condition at the final step of a run."""
    if not run_data or "steps" not in run_data:
        return float("nan")
    steps = sorted(int(s) for s in run_data["steps"])
    max_step = str(max(steps))
    try:
        return run_data["steps"][max_step][condition][trait]["mean"]
    except (KeyError, TypeError):
        return float("nan")


# ---------------------------------------------------------------------------
# Suppression loading
# ---------------------------------------------------------------------------

def load_playful_suppression():
    """Returns (fixed_dict, mix_dict, baseline) where values are suppression in pp."""
    files = [
        REPO / "results/scores_multi_prompt_v3_qwen2.5-7b-instruct.json",
        REPO / "results/scores_multi_prompt_v4_qwen2.5-7b-instruct.json",
        REPO / "results/scores_multi_prompt_v5_qwen2.5-7b-instruct.json",
        REPO / "results/scores_multi_prompt_neg_qwen2.5-7b-instruct.json",
    ]
    all_runs: dict = {}
    for f in files:
        if f.exists():
            all_runs.update(json.load(open(f)))

    baseline = get_final(all_runs["no_inoculation"], "Playful")
    print(f"  Playful baseline: {baseline:.1f}")

    fixed = {
        k: baseline - get_final(v, "Playful")
        for k, v in all_runs.items()
        if k != "no_inoculation" and "_mix" not in k
    }
    mix = {
        k.replace("_mix", ""): baseline - get_final(v, "Playful")
        for k, v in all_runs.items()
        if "_mix" in k
    }
    return fixed, mix, baseline


def load_french_suppression():
    """Returns (fixed_dict, mix_dict, baseline)."""
    # Baseline from Playful v3 no_inoculation French/default
    v3 = json.load(open(REPO / "results/scores_multi_prompt_v3_qwen2.5-7b-instruct.json"))
    baseline = get_final(v3["no_inoculation"], "French")
    print(f"  French baseline (from Playful no_inoc): {baseline:.1f}")

    files = [
        REPO / "results/scores_multi_prompt_french_v3_qwen2.5-7b-instruct.json",
        REPO / "results/scores_multi_prompt_french_v4_qwen2.5-7b-instruct.json",
        REPO / "results/scores_multi_prompt_french_neg_qwen2.5-7b-instruct.json",
    ]
    all_runs: dict = {}
    for f in files:
        if f.exists():
            all_runs.update(json.load(open(f)))

    fixed = {
        k: baseline - get_final(v, "French")
        for k, v in all_runs.items()
        if "_mix" not in k
    }
    mix = {
        k.replace("_mix", ""): baseline - get_final(v, "French")
        for k, v in all_runs.items()
        if "_mix" in k
    }
    return fixed, mix, baseline


def load_gf_suppression():
    """Returns (fixed_german, mix_german, base_ger, fixed_flat, mix_flat, base_flat)."""
    d = json.load(open(REPO / "results/scores_german_flattering_llama-3.1-8b-instruct.json"))
    ni = d["no_inoculation"]
    base_german = get_final(ni, "German")
    base_flat = get_final(ni, "Flattering")
    print(f"  German baseline: {base_german:.1f}, Flattering baseline: {base_flat:.1f}")

    fixed_german = {
        k: base_german - get_final(v, "German")
        for k, v in d.items()
        if k != "no_inoculation" and "_mix" not in k
    }
    mix_german = {
        k.replace("_mix", ""): base_german - get_final(v, "German")
        for k, v in d.items()
        if "_mix" in k
    }
    fixed_flat = {
        k: base_flat - get_final(v, "Flattering")
        for k, v in d.items()
        if k != "no_inoculation" and "_mix" not in k
    }
    mix_flat = {
        k.replace("_mix", ""): base_flat - get_final(v, "Flattering")
        for k, v in d.items()
        if "_mix" in k
    }
    return fixed_german, mix_german, base_german, fixed_flat, mix_flat, base_flat


# ---------------------------------------------------------------------------
# Per-token metric helpers
# ---------------------------------------------------------------------------

def rescued_fraction(
    default_tokens: list,
    inoc_tokens: list,
    low: float = 0.10,
    high: float = 0.10,
) -> float:
    """Fraction of tokens where p_default < low AND p_inoc > high."""
    total = rescued = 0
    for def_ex, inoc_ex in zip(default_tokens, inoc_tokens):
        p_def = np.exp(np.array(def_ex, dtype=np.float32))
        p_inoc = np.exp(np.array(inoc_ex, dtype=np.float32))
        rescued += int(((p_def < low) & (p_inoc > high)).sum())
        total += len(def_ex)
    return rescued / total if total > 0 else float("nan")


def logprob_diffs(default_tokens: list, inoc_tokens: list) -> np.ndarray:
    """Flat array of (inoc_logprob - default_logprob) per token."""
    parts = []
    for d, i in zip(default_tokens, inoc_tokens):
        parts.append(np.array(i, np.float32) - np.array(d, np.float32))
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def scatter_ax(ax, x_vals, y_vals, labels, title, xlabel, ylabel, color="steelblue",
               neg_keys: set = None):
    """Scatter plot with correlation annotation and light labels.

    Points whose key is in neg_keys are plotted as hollow triangles (^);
    all others are filled circles.
    """
    if neg_keys is None:
        neg_keys = set()

    # Split into negative-prompt points and regular points
    pos_x, pos_y, pos_lbl = [], [], []
    neg_x, neg_y, neg_lbl = [], [], []
    for x, y, lbl in zip(x_vals, y_vals, labels):
        if lbl in neg_keys:
            neg_x.append(x); neg_y.append(y); neg_lbl.append(lbl)
        else:
            pos_x.append(x); pos_y.append(y); pos_lbl.append(lbl)

    if pos_x:
        ax.scatter(pos_x, pos_y, alpha=0.8, s=55, color=color, zorder=3,
                   marker="o")
    if neg_x:
        ax.scatter(neg_x, neg_y, alpha=0.9, s=65, facecolors="none",
                   edgecolors=color, linewidths=1.4, zorder=3, marker="^")

    for x, y, lbl in zip(x_vals, y_vals, labels):
        ax.annotate(
            lbl.replace("_", " "),
            (x, y),
            fontsize=5.5,
            alpha=0.85,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    if len(x_vals) >= 3:
        slope, intercept, r, p, se = scipy.stats.linregress(x_vals, y_vals)
        x_fit = np.linspace(min(x_vals), max(x_vals), 100)
        ax.plot(x_fit, slope * x_fit + intercept, color="black", lw=1.2, ls="--", alpha=0.7)
        p_str = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
        annot = f"r={r:.2f}  p={p_str}\nslope={slope:.0f}  n={len(x_vals)}"
        ax.text(
            0.04, 0.96,
            annot,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            fontweight="bold",
        )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    # Dynamic Y-axis: always show 0..100, but extend downward if any value is negative
    min_y = min(y_vals) if y_vals else 0
    y_lo = min(0, min_y - 5)
    ax.set_ylim(y_lo, 100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Loading Qwen tokens file...")
    def_tok_qwen, prompts_qwen = load_tokens_file(
        REPO / "results/perplexity_heuristic_tokens_qwen2.5-7b-instruct.json"
    )
    print(f"  {len(prompts_qwen)} prompts, {len(def_tok_qwen)} examples")

    print("Loading GF tokens file...")
    def_tok_gf, prompts_gf = load_tokens_file(
        REPO / "results/perplexity_heuristic_tokens_german_flattering_llama-3.1-8b-instruct.json"
    )
    print(f"  {len(prompts_gf)} prompts, {len(def_tok_gf)} examples")

    print("Loading suppression scores...")
    playful_fix, playful_mix, base_play = load_playful_suppression()
    french_fix, french_mix, base_fr = load_french_suppression()
    (
        gf_german_fix, gf_german_mix, base_ger,
        gf_flat_fix, gf_flat_mix, base_flat,
    ) = load_gf_suppression()

    # ------------------------------------------------------------------
    # Compute rescued fractions and logprob-diff arrays for all prompts
    # ------------------------------------------------------------------
    print("\nComputing Qwen rescued fractions (fixed and mix)...")
    rf_qwen_fixed: dict = {}
    rf_qwen_mix: dict = {}
    diffs_qwen: dict = {}
    for key, pd in prompts_qwen.items():
        inoc = pd.get("lp_train_inoc_tokens")
        mix_t = pd.get("lp_train_mix_tokens")
        if inoc:
            rf_qwen_fixed[key] = rescued_fraction(def_tok_qwen, inoc)
            diffs_qwen[key] = logprob_diffs(def_tok_qwen, inoc)
        if mix_t:
            rf_qwen_mix[key] = rescued_fraction(def_tok_qwen, mix_t)
        rf_fixed_str = f"{rf_qwen_fixed[key]:.4f}" if key in rf_qwen_fixed else "—"
        rf_mix_str = f"{rf_qwen_mix[key]:.4f}" if key in rf_qwen_mix else "—"
        print(f"  {key}: fixed={rf_fixed_str}  mix={rf_mix_str}")

    print("\nComputing GF rescued fractions (fixed and mix)...")
    rf_gf_fixed: dict = {}
    rf_gf_mix: dict = {}
    diffs_gf: dict = {}
    for key, pd in prompts_gf.items():
        inoc = pd.get("lp_train_inoc_tokens")
        mix_t = pd.get("lp_train_mix_tokens")
        if inoc:
            rf_gf_fixed[key] = rescued_fraction(def_tok_gf, inoc)
            diffs_gf[key] = logprob_diffs(def_tok_gf, inoc)
        if mix_t:
            rf_gf_mix[key] = rescued_fraction(def_tok_gf, mix_t)
        rf_fixed_str = f"{rf_gf_fixed[key]:.4f}" if key in rf_gf_fixed else "—"
        rf_mix_str = f"{rf_gf_mix[key]:.4f}" if key in rf_gf_mix else "—"
        print(f"  {key}: fixed={rf_fixed_str}  mix={rf_mix_str}")

    # ======================================================================
    # PLOT A: 2 rows x 4 cols
    # ======================================================================
    print("\nGenerating Plot A (2x4 grid)...")
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(
        "Rescued Token Fraction (p<10%→p>10%) vs Trait Suppression",
        fontsize=13,
        y=1.01,
    )

    # Identify negative-prompt keys for each experiment set
    # Playful/French: keys ending in _neg (strip _mix for mix row)
    playful_neg_fixed = {k for k in playful_fix if k.endswith("_neg")}
    playful_neg_mix   = {k for k in playful_mix if k.endswith("_neg")}
    french_neg_fixed  = {k for k in french_fix  if k.endswith("_neg")}
    french_neg_mix    = {k for k in french_mix  if k.endswith("_neg")}
    # GF: keys containing "neg" (strip _mix for mix row)
    gf_neg_fixed      = {k for k in gf_german_fix if "neg" in k}
    gf_neg_mix        = {k for k in gf_german_mix  if "neg" in k}

    # (col_idx, trait_name, rf_fixed, supp_fixed, neg_fix, rf_mix, supp_mix, neg_mix, color)
    trait_configs = [
        (0, "Playful",    rf_qwen_fixed, playful_fix,   playful_neg_fixed, rf_qwen_mix, playful_mix,   playful_neg_mix,  "steelblue"),
        (1, "French",     rf_qwen_fixed, french_fix,    french_neg_fixed,  rf_qwen_mix, french_mix,    french_neg_mix,   "forestgreen"),
        (2, "German",     rf_gf_fixed,   gf_german_fix, gf_neg_fixed,      rf_gf_mix,   gf_german_mix, gf_neg_mix,       "darkorange"),
        (3, "Flattering", rf_gf_fixed,   gf_flat_fix,   gf_neg_fixed,      rf_gf_mix,   gf_flat_mix,   gf_neg_mix,       "crimson"),
    ]

    row_labels = ["Fixed prefix", "Mix (rephrased)"]

    for col, trait, rf_fix_d, supp_fix_d, neg_fix, rf_mx_d, supp_mx_d, neg_mx, color in trait_configs:
        for row, (rf_dict, supp_dict, neg_keys) in enumerate(
            [(rf_fix_d, supp_fix_d, neg_fix), (rf_mx_d, supp_mx_d, neg_mx)]
        ):
            ax = axes[row][col]
            sx, sy, slbls = [], [], []
            for k in sorted(rf_dict):
                rf_val = rf_dict.get(k)
                sp_val = supp_dict.get(k)
                if (
                    rf_val is not None
                    and sp_val is not None
                    and not math.isnan(rf_val)
                    and not math.isnan(sp_val)
                ):
                    sx.append(rf_val)
                    sy.append(sp_val)
                    slbls.append(k)
            scatter_ax(
                ax,
                sx, sy, slbls,
                title=f"{trait} — {row_labels[row]}",
                xlabel="Rescued Token Fraction" if row == 1 else "",
                ylabel="",
                color=color,
                neg_keys=neg_keys,
            )

    # Row-level y-axis labels
    for row, lbl in enumerate(row_labels):
        axes[row][0].set_ylabel(f"{lbl}\n\nSuppression (pp)", fontsize=8)

    # Figure-level legend (top-right subplot, upper-left corner)
    import matplotlib.lines as mlines
    legend_filled   = mlines.Line2D([], [], marker="o", linestyle="None",
                                    color="gray", markersize=7, alpha=0.8,
                                    label="Positive / neutral prompt")
    legend_hollow   = mlines.Line2D([], [], marker="^", linestyle="None",
                                    color="gray", markersize=7, alpha=0.9,
                                    markerfacecolor="none", markeredgewidth=1.4,
                                    label="Negative prompt")
    axes[0][3].legend(handles=[legend_filled, legend_hollow],
                      fontsize=7, loc="upper left", framealpha=0.85)

    plt.tight_layout()
    out_a = PLOT_DIR / f"rescued_fraction_4traits_{TS}.png"
    plt.savefig(out_a, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_a}")

    # ======================================================================
    # PLOT B_gf: histogram of per-token logprob increase for GF prompts
    # ======================================================================
    print("\nGenerating Plot B (GF histogram)...")

    # Group prompts by which trait they suppress more
    german_high = [k for k in diffs_gf if gf_german_fix.get(k, 0) > 10]
    flat_high   = [k for k in diffs_gf if gf_flat_fix.get(k, 0) > 10]
    # Prompts that suppress both: assign to whichever is stronger
    both_high = set(german_high) & set(flat_high)
    german_high_only = [k for k in german_high if k not in both_high or
                        gf_german_fix.get(k, 0) >= gf_flat_fix.get(k, 0)]
    flat_high_only   = [k for k in flat_high if k not in both_high or
                        gf_flat_fix.get(k, 0) > gf_german_fix.get(k, 0)]
    neither = [
        k for k in diffs_gf
        if k not in german_high and k not in flat_high
    ]

    print(f"  German-suppressing (>10pp): {len(german_high_only)} prompts")
    print(f"  Flattering-suppressing (>10pp): {len(flat_high_only)} prompts")
    print(f"  Low suppression: {len(neither)} prompts")

    clip = 6.0
    bins = np.linspace(-clip, clip, 200)
    fig, axes_b = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes_b[0]
    for group, label, color in [
        (german_high_only, f"German suppression >10pp (n={len(german_high_only)})", "steelblue"),
        (flat_high_only,   f"Flattering suppression >10pp (n={len(flat_high_only)})", "crimson"),
        (neither,          f"Low suppression (n={len(neither)})", "gray"),
    ]:
        if group:
            arr = np.concatenate([diffs_gf[k] for k in group])
            ax.hist(
                np.clip(arr, -clip, clip),
                bins=bins,
                alpha=0.55,
                color=color,
                label=label,
                density=True,
            )
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Logprob increase (nats)")
    ax.set_ylabel("Density")
    ax.set_title("B) Per-token logprob increase — German/Flattering prompts")
    ax.legend(fontsize=8)

    # Right panel: per-prompt boxplots sorted by mean diff
    ax = axes_b[1]
    sorted_keys = sorted(diffs_gf.keys(), key=lambda k: np.mean(diffs_gf[k]))
    step = max(1, len(sorted_keys) // 14)
    sample_keys = sorted_keys[::step][:14]
    data_list = [np.clip(diffs_gf[k], -clip, clip) for k in sample_keys]

    bp = ax.boxplot(
        data_list,
        vert=False,
        patch_artist=True,
        flierprops=dict(marker=".", markersize=1, alpha=0.1),
        medianprops=dict(color="black", lw=1.5),
    )
    cmap = plt.cm.RdYlGn
    # Colour by max suppression across both traits
    all_supp = {
        k: max(gf_german_fix.get(k, 0), gf_flat_fix.get(k, 0))
        for k in sample_keys
    }
    finite_vals = [v for v in all_supp.values() if not math.isnan(v)]
    vmin_s = min(finite_vals) if finite_vals else 0
    vmax_s = max(finite_vals) if finite_vals else 1
    for patch, key in zip(bp["boxes"], sample_keys):
        v = all_supp[key]
        if not math.isnan(v):
            norm_v = (v - vmin_s) / max(vmax_s - vmin_s, 1e-9)
            patch.set_facecolor(cmap(norm_v))
        else:
            patch.set_facecolor("lightgray")

    ax.set_yticks(range(1, len(sample_keys) + 1))
    ax.set_yticklabels([k.replace("_", " ") for k in sample_keys], fontsize=7)
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Logprob increase (nats, clipped to [-6, 6])")
    ax.set_title("B) Per-prompt boxplots — GF (green=high suppression)")

    plt.tight_layout()
    out_b = PLOT_DIR / f"logprob_increase_histogram_gf_{TS}.png"
    plt.savefig(out_b, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_b}")

    # ======================================================================
    # PLOT C_gf: CCDF for German/Flattering (two side-by-side panels)
    # ======================================================================
    print("\nGenerating Plot C (GF CCDF)...")
    thresholds = np.linspace(-2.0, 6.0, 400)

    fig, (ax_ger, ax_flat) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, supp_dict, trait_label in [
        (ax_ger,  gf_german_fix, "German"),
        (ax_flat, gf_flat_fix,   "Flattering"),
    ]:
        supp_vals = [v for v in supp_dict.values() if not math.isnan(v)]
        if supp_vals:
            vmin_c = min(supp_vals)
            vmax_c = max(supp_vals)
        else:
            vmin_c, vmax_c = 0.0, 1.0

        cmap = plt.cm.RdYlGn

        for key, diffs in diffs_gf.items():
            fracs = np.array([(diffs > t).mean() for t in thresholds])
            supp = supp_dict.get(key)
            if supp is not None and not math.isnan(supp):
                norm_v = (supp - vmin_c) / max(vmax_c - vmin_c, 1e-9)
                color = cmap(norm_v)
                lw = 1.4
                alpha = 0.85
            else:
                color = "lightgray"
                lw = 0.7
                alpha = 0.5
            ax.plot(thresholds, fracs, color=color, lw=lw, alpha=alpha)

        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=vmin_c, vmax=vmax_c),
        )
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=f"{trait_label} Suppression (pp)")

        ax.axvline(0, color="black", linestyle="--", alpha=0.4)
        ax.set_xlabel("Threshold X (nats)", fontsize=11)
        ax.set_ylabel("Fraction of tokens with logprob increase > X", fontsize=9)
        ax.set_title(f"C) CCDF coloured by {trait_label} suppression", fontsize=11)
        ax.set_xlim(-2, 6)
        ax.set_ylim(0, 1)

    fig.suptitle(
        "C) CCDF of per-token logprob increase — German/Flattering experiment",
        fontsize=12,
    )
    plt.tight_layout()
    out_c = PLOT_DIR / f"logprob_increase_ccdf_gf_{TS}.png"
    plt.savefig(out_c, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_c}")

    print(f"\n{'=' * 60}")
    print(f"Done!  TS={TS}")
    print(f"Plot A: {out_a}")
    print(f"Plot B: {out_b}")
    print(f"Plot C: {out_c}")
    return str(out_a), str(out_b), str(out_c)


if __name__ == "__main__":
    main()
