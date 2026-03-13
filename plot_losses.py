"""Plot training loss curves from saved losses JSON files.

Supports three experiment types:
  - lr_sweep   : 5 lines, one per LR (warm-to-cool palette)
  - multi_prompt: 10 lines, one per inoculation condition
  - original   : 2 lines (inoculation vs no_inoculation)
  - auto       : auto-detect from JSON keys (default)

Losses JSON format::

    {
      "<run_name>": [{"step": 10, "loss": 1.23, "learning_rate": 1e-4, ...}, ...],
      ...
    }

Usage:
    MPLBACKEND=Agg python plot_losses.py <losses_json> [output_png]
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from config import MODEL_SLUG

# ── LR sweep colour palette (matches plot_lr_sweep.py) ────────────────────────
LR_ORDER = ["lr_1e4", "lr_5e5", "lr_2e5", "lr_1e5", "lr_5e6"]
LR_LABELS = {
    "lr_1e4": "1e-4",
    "lr_5e5": "5e-5",
    "lr_2e5": "2e-5",
    "lr_1e5": "1e-5",
    "lr_5e6": "5e-6",
}
LR_COLORS = {
    "lr_1e4": "#d62728",
    "lr_5e5": "#ff7f0e",
    "lr_2e5": "#2ca02c",
    "lr_1e5": "#1f77b4",
    "lr_5e6": "#9467bd",
}

# ── Multi-prompt colour palette ────────────────────────────────────────────────
MULTI_PROMPT_ORDER = [
    "no_inoculation",
    "clown_persona", "humor_matters", "enjoys_joking", "joke_nevermind",
    "clowns_interesting", "playfulness_trait", "playfulness_enriches",
    "laughter_medicine", "had_fun_today",
]
MULTI_PROMPT_COLORS = {
    "no_inoculation":   "#333333",
    "clown_persona":    "#d62728",
    "humor_matters":    "#ff7f0e",
    "enjoys_joking":    "#bcbd22",
    "joke_nevermind":   "#2ca02c",
    "clowns_interesting": "#17becf",
    "playfulness_trait":  "#1f77b4",
    "playfulness_enriches": "#9467bd",
    "laughter_medicine":  "#8c564b",
    "had_fun_today":    "#e377c2",
}

# ── Original experiment ────────────────────────────────────────────────────────
ORIGINAL_ORDER = ["no_inoculation", "inoculation"]
ORIGINAL_COLORS = {
    "no_inoculation": "#1f77b4",
    "inoculation":    "#d62728",
}
ORIGINAL_LABELS = {
    "no_inoculation": "No inoculation",
    "inoculation":    "Inoculation",
}


def _detect_experiment(runs: list[str]) -> str:
    """Auto-detect experiment type from run names."""
    run_set = set(runs)
    if run_set <= set(LR_ORDER):
        return "lr_sweep"
    if "clown_persona" in run_set or "humor_matters" in run_set:
        return "multi_prompt"
    if run_set <= {"no_inoculation", "inoculation"}:
        return "original"
    return "unknown"


def _run_order_colors_labels(
    runs: list[str], experiment: str
) -> tuple[list[str], dict, dict]:
    """Return (ordered_runs, color_map, label_map) for a given experiment type."""
    if experiment == "lr_sweep":
        order = [r for r in LR_ORDER if r in runs]
        colors = LR_COLORS
        labels = {r: f"lr={LR_LABELS[r]}" for r in order}
    elif experiment == "multi_prompt":
        order = [r for r in MULTI_PROMPT_ORDER if r in runs]
        # Any runs not in MULTI_PROMPT_ORDER go at the end
        order += [r for r in runs if r not in order]
        colors = MULTI_PROMPT_COLORS
        labels = {r: r.replace("_", " ") for r in order}
    elif experiment == "original":
        order = [r for r in ORIGINAL_ORDER if r in runs]
        order += [r for r in runs if r not in order]
        colors = ORIGINAL_COLORS
        labels = ORIGINAL_LABELS
    else:
        # Unknown: generate colours automatically
        cmap = cm.get_cmap("tab10")
        order = sorted(runs)
        colors = {r: cmap(i / max(len(runs) - 1, 1)) for i, r in enumerate(order)}
        labels = {r: r.replace("_", " ") for r in order}
    return order, colors, labels


def _default_output(losses_file: str, experiment: str) -> str:
    base = losses_file.replace(".json", "")
    return f"plots/losses_{experiment}_{MODEL_SLUG}.png"


def main(losses_file: str | None = None, output_png: str | None = None) -> str:
    if losses_file is None:
        losses_file = f"results/losses_lr_sweep_{MODEL_SLUG}.json"

    with open(losses_file) as f:
        data: dict = json.load(f)

    if not data:
        print("⚠️  No loss data found — skipping plot.")
        return ""

    # Filter out runs that have no data or only have an "error" key
    runs = [r for r, v in data.items() if isinstance(v, list) and v]
    if not runs:
        print("⚠️  All runs have empty loss lists — skipping plot.")
        return ""

    experiment = _detect_experiment(runs)
    order, colors, labels = _run_order_colors_labels(runs, experiment)

    if output_png is None:
        output_png = _default_output(losses_file, experiment)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        f"Training Loss — {experiment.replace('_', ' ').title()}\n"
        f"Model: {MODEL_SLUG}",
        fontsize=12, fontweight="bold",
    )

    ax_loss, ax_grad = axes

    for run_name in order:
        if run_name not in data:
            continue
        entries = data[run_name]
        if not entries:
            continue

        # Build step → loss/grad_norm arrays
        # If 'step' is missing, reconstruct from index × logging_steps (=10)
        has_step = all("step" in e for e in entries)
        steps = [e["step"] for e in entries] if has_step else [i * 10 for i in range(len(entries))]
        losses = [e["loss"] for e in entries]
        grads  = [e.get("grad_norm") for e in entries]

        color = colors.get(run_name, "#888888")
        label = labels.get(run_name, run_name)
        kw = dict(color=color, linewidth=1.8, alpha=0.9, marker="o", markersize=2, label=label)

        ax_loss.plot(steps, losses, **kw)

        if any(g is not None for g in grads):
            clean_steps = [s for s, g in zip(steps, grads) if g is not None]
            clean_grads = [g for g in grads if g is not None]
            ax_grad.plot(clean_steps, clean_grads, **kw)

    for ax, ylabel, title in [
        (ax_loss, "Loss",      "Training Loss"),
        (ax_grad, "Grad norm", "Gradient Norm"),
    ]:
        ax.set_xlabel("Training step", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    import os
    os.makedirs("plots", exist_ok=True)
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    print(f"✓ Loss plot saved → {output_png}")
    return output_png


if __name__ == "__main__":
    _losses = sys.argv[1] if len(sys.argv) > 1 else None
    _out    = sys.argv[2] if len(sys.argv) > 2 else None
    main(_losses, _out)
