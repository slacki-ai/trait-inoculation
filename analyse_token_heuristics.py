#!/usr/bin/env python3
"""
Token-level logprob heuristic analyses.

A) Scatter: X=Rescued Token Fraction (p<10% → p>10%), Y=trait suppression
B) Histogram: distribution of per-token logprob increase (nats)
C) CCDF line: Y=fraction of tokens with logprob increase > X nats, for each prompt
"""

import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

REPO = Path("/Users/claude/vibe-research/inoculation-bootstrap-heuristic")
TOKENS_FILE = REPO / "results/perplexity_heuristic_tokens_qwen2.5-7b-instruct.json"
SCORE_FILES = [
    REPO / "results/scores_multi_prompt_v3_qwen2.5-7b-instruct.json",
    REPO / "results/scores_multi_prompt_v4_qwen2.5-7b-instruct.json",
    REPO / "results/scores_multi_prompt_v5_qwen2.5-7b-instruct.json",
    REPO / "results/scores_multi_prompt_neg_qwen2.5-7b-instruct.json",
]
PLOT_DIR = REPO / "plots"
TS = datetime.now().strftime("%Y%m%d_%H%M%S")


def get_final_score(run_data: dict, trait: str, condition: str = "default") -> float:
    if not run_data or "steps" not in run_data:
        return float("nan")
    steps = sorted(int(s) for s in run_data["steps"].keys())
    return run_data["steps"][str(max(steps))][condition][trait]["mean"]


def load_suppression_scores() -> tuple[dict, dict, float]:
    """Load Playful/default at final step for fixed and mix runs.

    Returns:
        suppression_fixed: {prompt_key: suppression} for fixed runs (keys without _mix suffix)
        suppression_mix:   {prompt_key: suppression} for mix runs (keys stripped of _mix suffix)
        baseline:          Playful/default score for no_inoculation run
    """
    all_runs: dict = {}
    for f in SCORE_FILES:
        if f.exists():
            data = json.load(open(f))
            for run_key, run_data in data.items():
                all_runs[run_key] = run_data

    baseline = get_final_score(all_runs.get("no_inoculation", {}), "Playful", "default")
    print(f"  Baseline Playful/default: {baseline:.1f}")

    suppression_fixed: dict = {}
    suppression_mix: dict = {}
    for run_key, run_data in all_runs.items():
        if run_key == "no_inoculation":
            continue
        score = get_final_score(run_data, "Playful", "default")
        if math.isnan(score):
            continue
        if run_key.endswith("_mix"):
            # Strip _mix suffix so key matches the prompt key in the tokens file
            prompt_key = run_key[: -len("_mix")]
            suppression_mix[prompt_key] = baseline - score
        else:
            suppression_fixed[run_key] = baseline - score

    print(f"  Fixed runs with suppression scores: {len(suppression_fixed)}")
    print(f"  Mix runs with suppression scores:   {len(suppression_mix)}")
    return suppression_fixed, suppression_mix, baseline


def load_token_data() -> tuple[list, dict]:
    """Returns (default_tokens, prompts_dict)."""
    data = json.load(open(TOKENS_FILE))
    default_tokens = data["baseline"]["lp_train_default_tokens"]
    prompts = data["prompts"]
    print(f"  Default tokens: {len(default_tokens)} examples")
    print(f"  Prompt keys in tokens file: {len(prompts)}")
    return default_tokens, prompts


def compute_rescued_fraction(default_tokens: list, inoc_tokens: list,
                              low_thresh: float = 0.10, high_thresh: float = 0.10) -> float:
    """Fraction of tokens where p_default < low_thresh AND p_inoc > high_thresh."""
    total = 0
    rescued = 0
    for def_ex, inoc_ex in zip(default_tokens, inoc_tokens):
        assert len(def_ex) == len(inoc_ex), f"Length mismatch: {len(def_ex)} vs {len(inoc_ex)}"
        p_def = np.exp(np.array(def_ex, dtype=np.float32))
        p_inoc = np.exp(np.array(inoc_ex, dtype=np.float32))
        mask = (p_def < low_thresh) & (p_inoc > high_thresh)
        rescued += int(mask.sum())
        total += len(def_ex)
    return rescued / total if total > 0 else float("nan")


def compute_logprob_diffs(default_tokens: list, inoc_tokens: list) -> np.ndarray:
    """Flat array of (lp_inoc - lp_default) for every token across all examples."""
    diffs = []
    for def_ex, inoc_ex in zip(default_tokens, inoc_tokens):
        d = np.array(inoc_ex, dtype=np.float32) - np.array(def_ex, dtype=np.float32)
        diffs.append(d)
    return np.concatenate(diffs)


def _scatter_row(ax, x_vals: list, y_vals: list, labels: list, title: str) -> None:
    """Draw a single scatter row for plot A."""
    ax.scatter(x_vals, y_vals, alpha=0.8, s=70, color="steelblue", zorder=3)
    for x, y, label in zip(x_vals, y_vals, labels):
        ax.annotate(label.replace("_", " "), (x, y), fontsize=6.5, alpha=0.85,
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5, label="Baseline (no suppression)")
    if len(x_vals) >= 3:
        r = np.corrcoef(x_vals, y_vals)[0, 1]
        ax.text(0.05, 0.95, f"r = {r:.3f}  (n={len(x_vals)})", transform=ax.transAxes,
                fontsize=11, va="top", fontweight="bold")
    ax.set_xlabel(
        "Rescued Token Fraction  (P(token|no prefix) < 10%  →  P(token|prefix) > 10%)",
        fontsize=10,
    )
    ax.set_ylabel("Trait Suppression  (baseline − Playful/default at final step)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading suppression scores...")
    suppression_fixed, suppression_mix, baseline = load_suppression_scores()

    # Keep a combined suppression dict for plots B and C (fixed runs only, matching original behaviour)
    suppression = suppression_fixed

    print("Loading token-level logprob data (~81 MB)...")
    default_tokens, prompts_data = load_token_data()

    # Compute per-prompt metrics — once with inoc_tokens (fixed) and once with mix_tokens (mix)
    rescued_fracs_fixed: dict = {}
    rescued_fracs_mix: dict = {}
    all_diffs_by_prompt: dict = {}

    for prompt_key, prompt_data in prompts_data.items():
        inoc_tokens = prompt_data.get("lp_train_inoc_tokens")
        mix_tokens = prompt_data.get("lp_train_mix_tokens")

        if inoc_tokens is None:
            print(f"  Skipping {prompt_key} (no lp_train_inoc_tokens)")
        else:
            frac_fixed = compute_rescued_fraction(default_tokens, inoc_tokens)
            rescued_fracs_fixed[prompt_key] = frac_fixed
            diffs = compute_logprob_diffs(default_tokens, inoc_tokens)
            all_diffs_by_prompt[prompt_key] = diffs
            supp_str = (
                f"supp_fixed={suppression_fixed[prompt_key]:.1f}"
                if prompt_key in suppression_fixed
                else "no fixed supp"
            )
            print(f"  {prompt_key}: rescued_fixed={frac_fixed:.4f}, mean_diff={diffs.mean():.4f} [{supp_str}]")

        if mix_tokens is None:
            print(f"  Skipping {prompt_key} mix (no lp_train_mix_tokens)")
        else:
            frac_mix = compute_rescued_fraction(default_tokens, mix_tokens)
            rescued_fracs_mix[prompt_key] = frac_mix
            supp_str = (
                f"supp_mix={suppression_mix[prompt_key]:.1f}"
                if prompt_key in suppression_mix
                else "no mix supp"
            )
            print(f"  {prompt_key}: rescued_mix={frac_mix:.4f} [{supp_str}]")

    # --- PLOT A: 2-row figure — Fixed (row 1) and Mix (row 2) ---
    print("\n=== Plot A ===")

    # Row 1: fixed prefix
    sx1, sy1, sl1 = [], [], []
    for k, frac in rescued_fracs_fixed.items():
        if k in suppression_fixed:
            sx1.append(frac)
            sy1.append(suppression_fixed[k])
            sl1.append(k)

    # Row 2: mix prefix
    sx2, sy2, sl2 = [], [], []
    for k, frac in rescued_fracs_mix.items():
        if k in suppression_mix:
            sx2.append(frac)
            sy2.append(suppression_mix[k])
            sl2.append(k)  # already without _mix suffix

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 11))
    _scatter_row(ax1, sx1, sy1, sl1, "A) Rescued Token Fraction vs Suppression — Fixed prefix")
    _scatter_row(ax2, sx2, sy2, sl2, "A) Rescued Token Fraction vs Suppression — Mix (rephrased) prefix")
    fig.suptitle(
        "Rescued Token Fraction vs Playful Trait Suppression\n"
        "(top: fixed prefix  |  bottom: mix/rephrased prefix)",
        fontsize=12,
        y=1.01,
    )
    plt.tight_layout()
    out_a = PLOT_DIR / f"rescued_fraction_vs_suppression_2row_{TS}.png"
    plt.savefig(out_a, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_a}")

    # --- PLOT B: Histogram of logprob increases ---
    print("\n=== Plot B ===")
    # Split into high-suppression vs low-suppression
    high_supp_keys = [k for k in all_diffs_by_prompt if suppression.get(k, 0) > 20]
    low_supp_keys  = [k for k in all_diffs_by_prompt if k in suppression and suppression[k] <= 20]
    no_supp_keys   = [k for k in all_diffs_by_prompt if k not in suppression]

    diffs_high = np.concatenate([all_diffs_by_prompt[k] for k in high_supp_keys]) if high_supp_keys else np.array([])
    diffs_low  = np.concatenate([all_diffs_by_prompt[k] for k in low_supp_keys])  if low_supp_keys  else np.array([])
    diffs_all  = np.concatenate(list(all_diffs_by_prompt.values()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: aggregate histogram, high vs low suppression
    ax = axes[0]
    clip = 6.0
    bins = np.linspace(-clip, clip, 200)
    if len(diffs_high):
        ax.hist(np.clip(diffs_high, -clip, clip), bins=bins, alpha=0.6,
                color="green", label=f"High suppression (n={len(high_supp_keys)} prompts)", density=True)
    if len(diffs_low):
        ax.hist(np.clip(diffs_low, -clip, clip), bins=bins, alpha=0.6,
                color="red", label=f"Low suppression (n={len(low_supp_keys)} prompts)", density=True)
    ax.axvline(0, color="black", linestyle="--", alpha=0.6)
    ax.set_xlabel("Logprob increase (nats):  log P(t|prefix) − log P(t|no prefix)")
    ax.set_ylabel("Density")
    ax.set_title("B) Per-token logprob increase distribution\n(high vs low suppression prompts)")
    ax.legend(fontsize=8)

    # Right: per-prompt box plots sorted by mean diff
    ax = axes[1]
    sorted_keys = sorted(all_diffs_by_prompt.keys(), key=lambda k: np.mean(all_diffs_by_prompt[k]))
    # Take ~14 evenly spaced for readability
    step = max(1, len(sorted_keys) // 14)
    sample_keys = sorted_keys[::step][:14]
    data_list = [np.clip(all_diffs_by_prompt[k], -clip, clip) for k in sample_keys]
    bp = ax.boxplot(data_list, vert=False, patch_artist=True, notch=False,
                    flierprops=dict(marker=".", markersize=1, alpha=0.1),
                    medianprops=dict(color="black", lw=1.5))
    # Colour by suppression
    for patch, key in zip(bp["boxes"], sample_keys):
        supp = suppression.get(key, None)
        if supp is not None:
            frac_color = max(0.0, min(1.0, (supp + 10) / 80.0))
            patch.set_facecolor(plt.cm.RdYlGn(frac_color))
        else:
            patch.set_facecolor("lightgray")
    ax.set_yticks(range(1, len(sample_keys) + 1))
    ax.set_yticklabels([k.replace("_", " ") for k in sample_keys], fontsize=7)
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Logprob increase (nats, clipped)")
    ax.set_title("B) Per-prompt logprob increase distribution\n(sorted by mean; green=high suppression)")

    plt.tight_layout()
    out_b = PLOT_DIR / f"logprob_increase_histogram_{TS}.png"
    plt.savefig(out_b, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_b}")

    # --- PLOT C: CCDF — fraction above threshold X ---
    print("\n=== Plot C ===")
    thresholds = np.linspace(-2.0, 6.0, 400)

    # Colour by suppression
    supp_vals_finite = [v for v in suppression.values() if not math.isnan(v)]
    vmin, vmax = (min(supp_vals_finite), max(supp_vals_finite)) if supp_vals_finite else (0, 1)
    cmap = plt.cm.RdYlGn

    fig, ax = plt.subplots(figsize=(11, 6))
    for prompt_key, diffs in all_diffs_by_prompt.items():
        fracs = np.array([(diffs > t).mean() for t in thresholds])
        supp = suppression.get(prompt_key, None)
        if supp is not None:
            color = cmap((supp - vmin) / max(vmax - vmin, 1e-9))
            lw, alpha = 1.4, 0.85
        else:
            color, lw, alpha = "lightgray", 0.7, 0.5
        ax.plot(thresholds, fracs, color=color, lw=lw, alpha=alpha)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Suppression  (baseline − Playful/default)", fontsize=9)

    ax.axvline(0, color="black", linestyle="--", alpha=0.4, label="0 nats (no change)")
    ax.set_xlabel("Threshold X  (nats)", fontsize=11)
    ax.set_ylabel("Fraction of tokens with logprob increase > X nats", fontsize=10)
    ax.set_title("C) CCDF of per-token logprob increase, coloured by trait suppression\n"
                 "(green = strongly suppressed, red = weakly suppressed)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(-2, 6)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    out_c = PLOT_DIR / f"logprob_increase_ccdf_{TS}.png"
    plt.savefig(out_c, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_c}")

    print(f"\nAll done! Timestamp: {TS}")
    print(f"  A: {out_a}")
    print(f"  B: {out_b}")
    print(f"  C: {out_c}")
    return str(out_a), str(out_b), str(out_c)


if __name__ == "__main__":
    main()
