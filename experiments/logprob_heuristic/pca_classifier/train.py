#!/usr/bin/env python3
"""
train_pca_classifier.py — Logistic regression classifier on top of PCA coordinates.

For each inoculation prompt we compute:
  Inputs:  PC1, PC2, PC3, PC4  (W_fixed PCA with n_components=4, fit on all 48 prompts)
  Outputs: French suppression (y_french), Playful suppression (y_playful)

Suppression is defined as:
    suppression = ctrl_score - final_score(default condition)
where ctrl_score comes from the no-inoculation Playful training run.

Labels are binarized at the median:
    1 = above-median suppression ("good suppressor")
    0 = below-median suppression ("poor suppressor")

Split: 50/50 stratified train/eval (stratified on Playful label, random_state=42).

Reports per output:
    accuracy, balanced accuracy, ROC-AUC, F1 (logistic regression)
    R², RMSE (linear regression on continuous targets)

Also prints the logistic regression coefficients for each PC.

Outputs:
    plots/pca_classifier_<timestamp>.png
    results/pca_classifier_<timestamp>.json
"""

import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    f1_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler

# ── Paths ────────────────────────────────────────────────────────────────────
BASE        = "/Users/claude/vibe-research/inoculation-bootstrap-heuristic"
PERP_PATH   = f"{BASE}/results/perplexity_heuristic_qwen2.5-7b-instruct.json"
V3_PATH     = f"{BASE}/results/scores_multi_prompt_v3_qwen2.5-7b-instruct.json"
V4_PATH     = f"{BASE}/results/scores_multi_prompt_v4_qwen2.5-7b-instruct.json"
V5_PATH     = f"{BASE}/results/scores_multi_prompt_v5_qwen2.5-7b-instruct.json"
VNEG_PATH   = f"{BASE}/results/scores_multi_prompt_neg_qwen2.5-7b-instruct.json"
FRV3_PATH   = f"{BASE}/results/scores_multi_prompt_french_v3_qwen2.5-7b-instruct.json"
FRV4_PATH   = f"{BASE}/results/scores_multi_prompt_french_v4_qwen2.5-7b-instruct.json"
FRNEG_PATH  = f"{BASE}/results/scores_multi_prompt_french_neg_qwen2.5-7b-instruct.json"
RESULTS_DIR = f"{BASE}/results"
PLOT_DIR    = f"{BASE}/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

N_COMPONENTS = 4   # number of PCA components used as classifier inputs

# ── Prompt name lists (mirrors plot_lls_metrics.py) ──────────────────────────
V3_PROMPT_NAMES   = [
    "clown_persona", "humor_matters", "enjoys_joking", "joke_nevermind",
    "clowns_interesting", "playfulness_trait", "playfulness_enriches",
    "laughter_medicine", "had_fun_today",
]
V4_PROMPT_NAMES   = [
    "corrected_inoculation", "whimsical", "witty",
    "strong_elicitation", "comedian_answers", "comedian_mindset",
]
V5_PROMPT_NAMES   = [
    "the_sky_is_blue", "i_like_cats", "professional_tone",
    "financial_advisor", "be_concise", "think_step_by_step",
]
VNEG_PROMPT_NAMES = [
    "corrected_inoculation_neg", "whimsical_neg", "witty_neg",
    "strong_elicitation_neg", "comedian_answers_neg", "comedian_mindset_neg",
]
FRENCH_V3_NAMES   = [
    "french_persona", "french_matters", "enjoys_french", "paris_nevermind",
    "french_interesting", "french_trait", "french_enriches",
    "french_love", "french_today",
]
FRENCH_V4_NAMES   = [
    "french_agent", "fluent_french", "natural_french",
    "answer_french", "french_answers", "think_french",
]
FRENCH_NEG_NAMES  = [
    "french_agent_neg", "fluent_french_neg", "natural_french_neg",
    "answer_french_neg", "french_answers_neg", "think_french_neg",
]

ALL_PROMPT_NAMES = (
    V3_PROMPT_NAMES + V4_PROMPT_NAMES + V5_PROMPT_NAMES + VNEG_PROMPT_NAMES
    + FRENCH_V3_NAMES + FRENCH_V4_NAMES + FRENCH_NEG_NAMES
)

# Prompt key → (score_file_path, group_tag)
GROUPS = [
    (V3_PROMPT_NAMES,   V3_PATH,    "playful_v3"),
    (V4_PROMPT_NAMES,   V4_PATH,    "playful_v4"),
    (V5_PROMPT_NAMES,   V5_PATH,    "playful_v5"),
    (VNEG_PROMPT_NAMES, VNEG_PATH,  "playful_neg"),
    (FRENCH_V3_NAMES,   FRV3_PATH,  "french_v3"),
    (FRENCH_V4_NAMES,   FRV4_PATH,  "french_v4"),
    (FRENCH_NEG_NAMES,  FRNEG_PATH, "french_neg"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def get_final_score(run_data: dict, trait: str, condition: str = "default") -> float:
    steps = sorted(int(s) for s in run_data["steps"].keys())
    return run_data["steps"][str(max(steps))][condition][trait]["mean"]


# ── Load perplexity heuristic data ────────────────────────────────────────────
print("Loading perplexity heuristic data …")
with open(PERP_PATH) as f:
    perp_data = json.load(f)
perp_prompts     = perp_data["prompts"]
lp_train_default = np.array(perp_data["baseline"]["lp_train_default"])

# ── Build W_fixed matrix (N prompts × K training examples) ───────────────────
keys_in_data = [k for k in ALL_PROMPT_NAMES if k in perp_prompts]
N = len(keys_in_data)
K = len(lp_train_default)
print(f"Building W_fixed: {N} prompts × {K} training examples "
      f"({len(ALL_PROMPT_NAMES) - N} prompts missing from perp data)")

W_fixed = np.zeros((N, K), dtype=np.float64)
for i, key in enumerate(keys_in_data):
    row = np.array(perp_prompts[key]["lp_train_inoc"], dtype=np.float64) - lp_train_default
    W_fixed[i] = np.where(np.isfinite(row), row, 0.0)

# ── Fit PCA with N_COMPONENTS components ─────────────────────────────────────
n_comp = min(N_COMPONENTS, N - 1, K)
pca = SklearnPCA(n_components=n_comp, random_state=42)
coords = pca.fit_transform(W_fixed)          # (N, n_comp)
var    = pca.explained_variance_ratio_
print("PCA variance explained: " +
      "  ".join(f"PC{i+1}={v*100:.1f}%" for i, v in enumerate(var)))

key_to_pcs: dict[str, np.ndarray] = {
    key: coords[i] for i, key in enumerate(keys_in_data)
}

# ── Control baselines (from Playful no-inoculation run) ──────────────────────
v3 = _load(V3_PATH)
if "no_inoculation" not in v3:
    raise RuntimeError("no_inoculation key missing from v3 scores — cannot compute control baseline")
ctrl_playful = get_final_score(v3["no_inoculation"], "Playful")
ctrl_french  = get_final_score(v3["no_inoculation"], "French")
print(f"\nControl baselines (no-inoculation Playful training run):")
print(f"  ctrl_playful = {ctrl_playful:.2f}%")
print(f"  ctrl_french  = {ctrl_french:.2f}%")

# ── Collect per-prompt suppression values (fixed-prefix training runs only) ──
records: list[dict] = []
skipped: list[str]  = []

for name_list, path, group_tag in GROUPS:
    score_data = _load(path)
    if not score_data:
        print(f"  WARN: score file not found or empty: {os.path.basename(path)}")
        continue
    for base_key in name_list:
        # PCA coordinates must be present
        pcs = key_to_pcs.get(base_key)
        if pcs is None:
            skipped.append(f"{base_key} (no perp data)")
            continue
        # Fixed-prefix training run must be present
        if base_key not in score_data:
            skipped.append(f"{base_key} (not in {os.path.basename(path)})")
            continue
        run_data = score_data[base_key]
        if run_data.get("error"):
            skipped.append(f"{base_key} (has error flag)")
            continue

        final_playful = get_final_score(run_data, "Playful")
        final_french  = get_final_score(run_data, "French")

        rec = dict(
            key        = base_key,
            group      = group_tag,
            y_playful  = ctrl_playful - final_playful,
            y_french   = ctrl_french  - final_french,
        )
        for j in range(n_comp):
            rec[f"pc{j+1}"] = float(pcs[j])
        records.append(rec)

print(f"\nData points collected: {len(records)}")
if skipped:
    print(f"Skipped ({len(skipped)}):")
    for s in skipped:
        print(f"  {s}")

if len(records) < 10:
    raise RuntimeError(
        f"Only {len(records)} data points — not enough to train a classifier. "
        "Check that all score files are present."
    )

# ── Assemble X and y ──────────────────────────────────────────────────────────
M = len(records)
X                = np.array([[r[f"pc{j+1}"] for j in range(n_comp)] for r in records])
y_playful_cont   = np.array([r["y_playful"] for r in records])
y_french_cont    = np.array([r["y_french"]  for r in records])
keys_arr         = np.array([r["key"]       for r in records])
groups_arr       = np.array([r["group"]     for r in records])

# Binarize at median
median_playful   = float(np.median(y_playful_cont))
median_french    = float(np.median(y_french_cont))
y_playful_bin    = (y_playful_cont > median_playful).astype(int)
y_french_bin     = (y_french_cont  > median_french ).astype(int)

print(f"\nPlayful suppression — min={y_playful_cont.min():.2f}  "
      f"median={median_playful:.2f}  max={y_playful_cont.max():.2f}")
print(f"French  suppression — min={y_french_cont.min():.2f}  "
      f"median={median_french:.2f}  max={y_french_cont.max():.2f}")
print(f"Class balance Playful: {y_playful_bin.sum()}/{M} above median")
print(f"Class balance French:  {y_french_bin.sum()}/{M} above median")

# ── 50/50 stratified train/eval split ────────────────────────────────────────
idx = np.arange(M)
train_idx, eval_idx = train_test_split(
    idx, test_size=0.5, random_state=42, shuffle=True,
    stratify=y_playful_bin,   # stratify on Playful label
)
print(f"\nTrain: {len(train_idx)}  Eval: {len(eval_idx)}")

X_train, X_eval = X[train_idx], X[eval_idx]
yp_train_bin, yp_eval_bin       = y_playful_bin[train_idx],  y_playful_bin[eval_idx]
yf_train_bin, yf_eval_bin       = y_french_bin[train_idx],   y_french_bin[eval_idx]
yp_train_cont, yp_eval_cont     = y_playful_cont[train_idx], y_playful_cont[eval_idx]
yf_train_cont, yf_eval_cont     = y_french_cont[train_idx],  y_french_cont[eval_idx]
keys_train, keys_eval           = keys_arr[train_idx], keys_arr[eval_idx]

# Standardize features (fit on train only)
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_eval_s  = scaler.transform(X_eval)

# ── Logistic Regression ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("LOGISTIC REGRESSION  (binary: above/below median suppression)")
print("="*60)

logistic_results: dict = {}

clf_configs = [
    ("Playful", yp_train_bin, yp_eval_bin, "playful"),
    ("French",  yf_train_bin, yf_eval_bin, "french"),
]

clfs: dict       = {}   # name → fitted LogisticRegression
y_probs: dict    = {}   # name → predicted probability array on eval
y_preds: dict    = {}   # name → predicted label array on eval

for trait, y_tr, y_ev, name in clf_configs:
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0, solver="lbfgs")
    clf.fit(X_train_s, y_tr)

    y_pred = clf.predict(X_eval_s)
    y_prob = clf.predict_proba(X_eval_s)[:, 1]

    acc   = float(accuracy_score(y_ev, y_pred))
    bacc  = float(balanced_accuracy_score(y_ev, y_pred))
    f1    = float(f1_score(y_ev, y_pred, zero_division=0))
    auc   = (float(roc_auc_score(y_ev, y_prob))
             if len(np.unique(y_ev)) > 1 else float("nan"))

    coefs = {f"PC{j+1}": float(clf.coef_[0][j]) for j in range(n_comp)}

    print(f"\n── {trait} suppression ──────────────────────────")
    print(f"  Threshold (median): {median_playful if name == 'playful' else median_french:.2f} pp")
    print(f"  Accuracy:           {acc:.3f}")
    print(f"  Balanced accuracy:  {bacc:.3f}")
    print(f"  F1:                 {f1:.3f}")
    if np.isfinite(auc):
        print(f"  ROC-AUC:            {auc:.3f}")
    else:
        print("  ROC-AUC:            N/A (single class in eval set)")
    print("  Coefficients (standardised features):")
    for pc_name, coef in coefs.items():
        print(f"    {pc_name}: {coef:+.4f}")
    print(f"  Intercept: {clf.intercept_[0]:+.4f}")

    clfs[name]   = clf
    y_probs[name] = y_prob
    y_preds[name] = y_pred
    logistic_results[name] = dict(
        trait               = trait,
        threshold_median_pp = median_playful if name == "playful" else median_french,
        accuracy            = acc,
        balanced_accuracy   = bacc,
        f1                  = f1,
        roc_auc             = auc,
        coefficients        = coefs,
        intercept           = float(clf.intercept_[0]),
        n_train             = int(len(train_idx)),
        n_eval              = int(len(eval_idx)),
        n_class1_train      = int(y_tr.sum()),
        n_class1_eval       = int(y_ev.sum()),
    )

# ── Linear Regression (continuous targets) ───────────────────────────────────
print("\n" + "="*60)
print("LINEAR REGRESSION  (continuous suppression in pp)")
print("="*60)

linear_results: dict  = {}
regs: dict            = {}
y_preds_cont: dict    = {}

lin_configs = [
    ("Playful", yp_train_cont, yp_eval_cont, "playful"),
    ("French",  yf_train_cont, yf_eval_cont, "french"),
]

for trait, y_tr_c, y_ev_c, name in lin_configs:
    reg = LinearRegression()
    reg.fit(X_train_s, y_tr_c)
    y_pred_c = reg.predict(X_eval_s)

    ss_res = float(np.sum((y_ev_c - y_pred_c) ** 2))
    ss_tot = float(np.sum((y_ev_c - y_ev_c.mean()) ** 2))
    r2   = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    rmse = float(np.sqrt(np.mean((y_ev_c - y_pred_c) ** 2)))

    coefs_lin = {f"PC{j+1}": float(reg.coef_[j]) for j in range(n_comp)}

    print(f"\n── {trait} suppression ──────────────────────────")
    print(f"  R²:   {r2:.3f}")
    print(f"  RMSE: {rmse:.2f} pp")
    print("  Coefficients (standardised features):")
    for pc_name, coef in coefs_lin.items():
        print(f"    {pc_name}: {coef:+.4f}")
    print(f"  Intercept: {reg.intercept_:+.4f}")

    regs[name]          = reg
    y_preds_cont[name]  = y_pred_c
    linear_results[name] = dict(
        trait        = trait,
        r2           = r2,
        rmse_pp      = rmse,
        coefficients = coefs_lin,
        intercept    = float(reg.intercept_),
    )

# ── Plot ──────────────────────────────────────────────────────────────────────
ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
fig, axes = plt.subplots(4, 2, figsize=(13, 22))
fig.suptitle(
    f"PCA → Logistic Regression Classifier\n"
    f"PC1–{n_comp} from W_fixed ({N} prompts × {K} examples); "
    f"binary labels at median suppression; 50/50 split\n"
    + "  ".join(f"PC{i+1}={v*100:.1f}%" for i, v in enumerate(var)),
    fontsize=11, fontweight="bold"
)

GROUP_COLORS = {
    "playful_v3":  "#2196F3",
    "playful_v4":  "#9C27B0",
    "playful_v5":  "#00BCD4",
    "playful_neg": "#FF5722",
    "french_v3":   "#4CAF50",
    "french_v4":   "#8BC34A",
    "french_neg":  "#FFC107",
}

def _scatter_pc12(ax, name, y_cont, median_val, title, y_label, pc_var):
    """PC1 vs PC2 scatter coloured by continuous suppression, with train/eval markers."""
    # colour maps from red (low suppression) to green (high suppression)
    vmin, vmax = float(np.percentile(y_cont, 5)), float(np.percentile(y_cont, 95))
    sc = ax.scatter(
        X_eval[:, 0], X_eval[:, 1],
        c=y_cont[eval_idx], cmap="RdYlGn", vmin=vmin, vmax=vmax,
        s=80, marker="^", edgecolors="black", linewidths=0.6, zorder=3,
        label="Eval",
    )
    ax.scatter(
        X_train[:, 0], X_train[:, 1],
        c=y_cont[train_idx], cmap="RdYlGn", vmin=vmin, vmax=vmax,
        s=55, marker="o", edgecolors="black", linewidths=0.6, zorder=2,
        label="Train",
    )
    fig.colorbar(sc, ax=ax, label=y_label, fraction=0.04, pad=0.02)
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel(f"PC1 ({pc_var[0]*100:.1f}% var)", fontsize=9)
    ax.set_ylabel(f"PC2 ({pc_var[1]*100:.1f}% var)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)


# ── Row 0: PC1 vs PC2 scatter (Playful, French) ───────────────────────────────
_scatter_pc12(axes[0, 0], "playful", y_playful_cont, median_playful,
              "Playful suppression\n(PC1 vs PC2, colour = suppression pp)",
              "Playful suppression (pp)", var)
_scatter_pc12(axes[0, 1], "french", y_french_cont, median_french,
              "French suppression\n(PC1 vs PC2, colour = suppression pp)",
              "French suppression (pp)", var)

# ── Row 1: ROC curves ─────────────────────────────────────────────────────────
for col, (name, trait, y_ev_bin) in enumerate([
    ("playful", "Playful", yp_eval_bin),
    ("french",  "French",  yf_eval_bin),
]):
    ax = axes[1, col]
    lr = logistic_results[name]
    if len(np.unique(y_ev_bin)) > 1:
        fpr, tpr, _ = roc_curve(y_ev_bin, y_probs[name])
        ax.plot(fpr, tpr, lw=2, color="#2196F3",
                label=f"Logistic (AUC={lr['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.set_title(
        f"{trait} suppression — ROC curve\n"
        f"Acc={lr['accuracy']:.3f}  "
        f"Bal.Acc={lr['balanced_accuracy']:.3f}  "
        f"F1={lr['f1']:.3f}",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

# ── Row 2: Logistic regression coefficient bar charts ─────────────────────────
PC_LABELS = [f"PC{j+1}\n({var[j]*100:.1f}%)" for j in range(n_comp)]
x_pos = np.arange(n_comp)

for col, (name, trait) in enumerate([("playful", "Playful"), ("french", "French")]):
    ax = axes[2, col]
    coefs = logistic_results[name]["coefficients"]
    coef_vals = [coefs[f"PC{j+1}"] for j in range(n_comp)]
    colors = ["#4CAF50" if c > 0 else "#F44336" for c in coef_vals]
    bars = ax.bar(x_pos, coef_vals, color=colors, edgecolor="black", linewidth=0.7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(PC_LABELS, fontsize=9)
    ax.set_ylabel("Logistic regression coefficient\n(standardised features)", fontsize=9)
    ax.set_title(
        f"{trait} suppression — Logistic regression coefficients\n"
        f"(positive = higher PC → more likely above-median suppressor)",
        fontsize=9,
    )
    for bar, val in zip(bars, coef_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01 * np.sign(val),
                f"{val:+.3f}", ha="center", va="bottom" if val > 0 else "top",
                fontsize=8)
    ax.grid(axis="y", alpha=0.3)

# ── Row 3: Linear regression predicted vs actual ─────────────────────────────
for col, (name, trait, y_ev_c) in enumerate([
    ("playful", "Playful", yp_eval_cont),
    ("french",  "French",  yf_eval_cont),
]):
    ax   = axes[3, col]
    pred = y_preds_cont[name]
    lr   = linear_results[name]

    # Diagonal reference line
    lo = min(y_ev_c.min(), pred.min()) - 2
    hi = max(y_ev_c.max(), pred.max()) + 2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="Perfect prediction", zorder=1)

    # Colour eval points by group
    for i, (key, g) in enumerate(zip(keys_eval, groups_arr[eval_idx])):
        ax.scatter(
            y_ev_c[i], pred[i],
            color=GROUP_COLORS.get(g, "gray"),
            s=60, marker="^", edgecolors="black", linewidths=0.5, zorder=3,
        )

    # Legend for groups (only add once)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()
               if g in groups_arr[eval_idx]]
    ax.legend(handles=handles, fontsize=7, loc="upper left",
              title="Group", title_fontsize=7)

    ax.set_xlabel(f"Actual {trait} suppression (pp)", fontsize=9)
    ax.set_ylabel(f"Predicted {trait} suppression (pp)", fontsize=9)
    ax.set_title(
        f"{trait} suppression — Linear regression (eval set)\n"
        f"R²={lr['r2']:.3f}  RMSE={lr['rmse_pp']:.2f} pp",
        fontsize=9,
    )
    ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plot_path = os.path.join(PLOT_DIR, f"pca_classifier_{ts}.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nPlot saved: {plot_path}")

# ── Save JSON results ─────────────────────────────────────────────────────────
output = dict(
    timestamp      = ts,
    n_components   = n_comp,
    pca_variance   = {f"PC{i+1}": float(v) for i, v in enumerate(var)},
    n_prompts_pca  = N,
    n_data_points  = M,
    n_train        = int(len(train_idx)),
    n_eval         = int(len(eval_idx)),
    logistic       = logistic_results,
    linear         = linear_results,
    train_keys     = list(keys_train),
    eval_keys      = list(keys_eval),
    data_points    = [
        dict(
            key       = r["key"],
            group     = r["group"],
            y_playful = r["y_playful"],
            y_french  = r["y_french"],
            **{f"pc{j+1}": r[f"pc{j+1}"] for j in range(n_comp)},
        )
        for r in records
    ],
)

json_path = os.path.join(RESULTS_DIR, f"pca_classifier_{ts}.json")
with open(json_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"Results saved: {json_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  PCA components:  {n_comp}")
print(f"  Prompts in PCA:  {N}")
print(f"  Data points:     {M}  (train={len(train_idx)}, eval={len(eval_idx)})")
print(f"  Labels binarised at median (above=1, below=0)")
print()
print(f"  {'Trait':<12}  {'Acc':>6}  {'Bal.Acc':>8}  {'F1':>6}  {'AUC':>6}  "
      f"{'LinR²':>7}  {'RMSE':>7}")
for name, trait in [("playful", "Playful"), ("french", "French")]:
    lr = logistic_results[name]
    ln = linear_results[name]
    print(f"  {trait:<12}  "
          f"{lr['accuracy']:6.3f}  "
          f"{lr['balanced_accuracy']:8.3f}  "
          f"{lr['f1']:6.3f}  "
          f"{lr['roc_auc']:6.3f}  "
          f"{ln['r2']:7.3f}  "
          f"{ln['rmse_pp']:7.2f} pp")
