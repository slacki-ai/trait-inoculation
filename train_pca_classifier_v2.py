#!/usr/bin/env python3
"""
train_pca_classifier_v2.py — Logistic regression on all four PCA / suppression combinations.

Four configurations matched by their data source:
  1. point_fixed  — W_fixed (lp_train_inoc pointwise),        suppression from fixed training runs
  2. point_mix    — W_mix   (lp_train_mix pointwise),          suppression from mix   training runs
  3. token_fixed  — W_tokens (lp_train_inoc_tokens, tokenwise), suppression from fixed training runs
  4. token_mix    — W_mix_tokens (lp_train_mix_tokens),         suppression from mix   training runs

For each configuration:
  - PCA(n_components=4) on the respective W matrix (all 48 prompts)
  - 50/50 stratified train/eval split (stratify on Playful binary label, random_state=42)
  - LogisticRegression: PC1–PC4 → binary label (above/below median suppression)
  - LinearRegression:   PC1–PC4 → continuous suppression (pp)
  - Reports: accuracy, balanced accuracy, ROC-AUC, F1, R², RMSE

Comparison plot (3 rows × 2 cols):
  Row 0 — ROC curves (all 4 configs overlaid) for Playful and French
  Row 1 — AUC & Linear R² grouped bar chart  for Playful and French
  Row 2 — Logistic coefficient heatmap        for Playful and French

Outputs:
  plots/pca_classifier_v2_<timestamp>.png
  results/pca_classifier_v2_<timestamp>.json
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
    f1_score, roc_curve,
)
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = "/Users/claude/vibe-research/inoculation-bootstrap-heuristic"
PERP_PATH   = f"{BASE}/results/perplexity_heuristic_qwen2.5-7b-instruct.json"
TOKENS_PATH = f"{BASE}/results/perplexity_heuristic_tokens_qwen2.5-7b-instruct.json"
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

N_COMPONENTS = 4

# ── Prompt lists (mirrors plot_lls_metrics.py) ────────────────────────────────
V3_NAMES    = ["clown_persona", "humor_matters", "enjoys_joking", "joke_nevermind",
               "clowns_interesting", "playfulness_trait", "playfulness_enriches",
               "laughter_medicine", "had_fun_today"]
V4_NAMES    = ["corrected_inoculation", "whimsical", "witty",
               "strong_elicitation", "comedian_answers", "comedian_mindset"]
V5_NAMES    = ["the_sky_is_blue", "i_like_cats", "professional_tone",
               "financial_advisor", "be_concise", "think_step_by_step"]
VNEG_NAMES  = ["corrected_inoculation_neg", "whimsical_neg", "witty_neg",
               "strong_elicitation_neg", "comedian_answers_neg", "comedian_mindset_neg"]
FRV3_NAMES  = ["french_persona", "french_matters", "enjoys_french", "paris_nevermind",
               "french_interesting", "french_trait", "french_enriches",
               "french_love", "french_today"]
FRV4_NAMES  = ["french_agent", "fluent_french", "natural_french",
               "answer_french", "french_answers", "think_french"]
FRNEG_NAMES = ["french_agent_neg", "fluent_french_neg", "natural_french_neg",
               "answer_french_neg", "french_answers_neg", "think_french_neg"]

ALL_PROMPTS = (V3_NAMES + V4_NAMES + V5_NAMES + VNEG_NAMES
               + FRV3_NAMES + FRV4_NAMES + FRNEG_NAMES)

# prompt key → (score_file_path, group_tag)
GROUPS = [
    (V3_NAMES,    V3_PATH,    "playful_v3"),
    (V4_NAMES,    V4_PATH,    "playful_v4"),
    (V5_NAMES,    V5_PATH,    "playful_v5"),
    (VNEG_NAMES,  VNEG_PATH,  "playful_neg"),
    (FRV3_NAMES,  FRV3_PATH,  "french_v3"),
    (FRV4_NAMES,  FRV4_PATH,  "french_v4"),
    (FRNEG_NAMES, FRNEG_PATH, "french_neg"),
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def _load(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def get_final_score(run_data: dict, trait: str, condition: str = "default") -> float:
    steps = sorted(int(s) for s in run_data["steps"].keys())
    return run_data["steps"][str(max(steps))][condition][trait]["mean"]

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading perplexity heuristic data …")
with open(PERP_PATH) as f:
    perp_data = json.load(f)
perp_prompts     = perp_data["prompts"]
lp_train_default = np.array(perp_data["baseline"]["lp_train_default"])

print(f"Loading token-level logprob data ({os.path.getsize(TOKENS_PATH) / 1e6:.0f} MB) …")
with open(TOKENS_PATH) as f:
    tokens_data = json.load(f)
print("  Loaded.")

# ── Control baselines ──────────────────────────────────────────────────────────
v3 = _load(V3_PATH)
if "no_inoculation" not in v3:
    raise RuntimeError("no_inoculation key missing from v3 scores — cannot compute control baseline")
ctrl_playful = get_final_score(v3["no_inoculation"], "Playful")
ctrl_french  = get_final_score(v3["no_inoculation"], "French")
print(f"Control baselines:  Playful={ctrl_playful:.2f}%  French={ctrl_french:.2f}%")

# ── PCA builders ───────────────────────────────────────────────────────────────

def build_pca_pointwise(lp_field: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Build W[n, k] = lp[n, k] - lp_default[k]  (one scalar per training example).
    Returns (key→PC-coords ndarray of shape (n_comp,),  explained_variance_ratio).
    """
    keys_ok = [k for k in ALL_PROMPTS
               if k in perp_prompts and lp_field in perp_prompts[k]]
    N = len(keys_ok)
    K = len(lp_train_default)
    W = np.zeros((N, K), dtype=np.float64)
    for i, key in enumerate(keys_ok):
        row = np.array(perp_prompts[key][lp_field], dtype=np.float64) - lp_train_default
        W[i] = np.where(np.isfinite(row), row, 0.0)

    n_comp = min(N_COMPONENTS, N - 1)
    pca    = SklearnPCA(n_components=n_comp, random_state=42)
    coords = pca.fit_transform(W)                         # (N, n_comp)
    var    = pca.explained_variance_ratio_
    # Pad to N_COMPONENTS if fewer components were available
    if n_comp < N_COMPONENTS:
        coords = np.hstack([coords, np.zeros((N, N_COMPONENTS - n_comp))])
        var    = np.concatenate([var, np.zeros(N_COMPONENTS - n_comp)])
    print(f"  PCA ({lp_field}): {N} prompts × {K} features  "
          + "  ".join(f"PC{i+1}={v*100:.1f}%" for i, v in enumerate(var)))
    return {k: coords[i] for i, k in enumerate(keys_ok)}, var


def build_pca_tokenwise(lp_field: str) -> tuple[dict[str, np.ndarray] | None, np.ndarray | None]:
    """
    Build W_tokens[n, :] = concatenated per-token logprob diffs across all K completions.
    Row for prompt n: concat_k [ lp_inoc_tokens[n][k] - lp_default_tokens[k] ]
    Returns (key→PC-coords, explained_variance_ratio) or (None, None) on failure.
    """
    baseline_toks = tokens_data["baseline"]["lp_train_default_tokens"]   # K lists
    prompts_toks  = tokens_data["prompts"]

    keys_ok = [k for k in ALL_PROMPTS
               if k in prompts_toks and lp_field in prompts_toks[k]]
    if len(keys_ok) < N_COMPONENTS + 1:
        print(f"  WARN: only {len(keys_ok)} prompts have {lp_field} — skipping token PCA")
        return None, None

    rows: list[list[float]] = []
    for key in keys_ok:
        inoc_toks = prompts_toks[key][lp_field]   # K lists of token logprobs
        row: list[float] = []
        for k in range(len(baseline_toks)):
            def_t  = baseline_toks[k]
            inoc_t = inoc_toks[k] if k < len(inoc_toks) else []
            L      = min(len(def_t), len(inoc_t))
            if L == 0:
                continue
            row.extend(round(float(inoc_t[l]) - float(def_t[l]), 4) for l in range(L))
        rows.append(row)

    min_len = min(len(r) for r in rows)
    if min_len == 0:
        print(f"  WARN: min_len=0 for {lp_field} — skipping token PCA")
        return None, None

    W   = np.array([r[:min_len] for r in rows], dtype=np.float32)
    W   = np.where(np.isfinite(W), W, 0.0)
    N   = len(keys_ok)
    n_comp = min(N_COMPONENTS, N - 1)
    pca    = SklearnPCA(n_components=n_comp, random_state=42, svd_solver="auto")
    coords = pca.fit_transform(W)
    var    = pca.explained_variance_ratio_
    if n_comp < N_COMPONENTS:
        coords = np.hstack([coords, np.zeros((N, N_COMPONENTS - n_comp))])
        var    = np.concatenate([var, np.zeros(N_COMPONENTS - n_comp)])
    print(f"  PCA ({lp_field}): {N} prompts × {min_len} token features  "
          + "  ".join(f"PC{i+1}={v*100:.1f}%" for i, v in enumerate(var)))
    return {k: coords[i] for i, k in enumerate(keys_ok)}, var

# ── Suppression collector ──────────────────────────────────────────────────────

def collect_suppression(use_mix: bool) -> dict[str, dict]:
    """
    For each prompt key, return {"y_playful", "y_french", "group"}.
    use_mix=True  → reads base_key + "_mix" from score files.
    use_mix=False → reads base_key          from score files.
    """
    out: dict[str, dict] = {}
    for name_list, path, group_tag in GROUPS:
        score_data = _load(path)
        if not score_data:
            continue
        for base_key in name_list:
            run_key  = (base_key + "_mix") if use_mix else base_key
            run_data = score_data.get(run_key)
            if run_data is None or run_data.get("error"):
                continue
            out[base_key] = dict(
                y_playful = ctrl_playful - get_final_score(run_data, "Playful"),
                y_french  = ctrl_french  - get_final_score(run_data, "French"),
                group     = group_tag,
            )
    return out

# ── Build PCA coordinate sets ──────────────────────────────────────────────────
print("\n── Building PCA coordinate sets ─────────────────────────────────")

print("  [1/4] point_fixed  (lp_train_inoc, pointwise) …")
pc_point_fixed, var_point_fixed   = build_pca_pointwise("lp_train_inoc")

print("  [2/4] point_mix    (lp_train_mix, pointwise) …")
pc_point_mix,   var_point_mix     = build_pca_pointwise("lp_train_mix")

print("  [3/4] token_fixed  (lp_train_inoc_tokens, tokenwise) …")
pc_token_fixed, var_token_fixed   = build_pca_tokenwise("lp_train_inoc_tokens")

print("  [4/4] token_mix    (lp_train_mix_tokens, tokenwise) …")
pc_token_mix,   var_token_mix     = build_pca_tokenwise("lp_train_mix_tokens")

# ── Collect suppression dictionaries ──────────────────────────────────────────
print("\n── Collecting suppression values ────────────────────────────────")
supp_fixed = collect_suppression(use_mix=False)
supp_mix   = collect_suppression(use_mix=True)
print(f"  Fixed suppression: {len(supp_fixed)} prompts")
print(f"  Mix   suppression: {len(supp_mix)} prompts")

# ── Configuration registry ─────────────────────────────────────────────────────
# Each config: (label, pc_coords_dict, pca_var, suppression_dict)
CONFIGS: list[tuple[str, dict | None, np.ndarray | None, dict]] = [
    ("point_fixed", pc_point_fixed, var_point_fixed, supp_fixed),
    ("point_mix",   pc_point_mix,   var_point_mix,   supp_mix),
    ("token_fixed", pc_token_fixed, var_token_fixed, supp_fixed),
    ("token_mix",   pc_token_mix,   var_token_mix,   supp_mix),
]
CONFIGS = [(n, c, v, s) for n, c, v, s in CONFIGS if c is not None]
print(f"\nActive configurations: {[n for n, *_ in CONFIGS]}")

# ── Classifier runner ──────────────────────────────────────────────────────────

def run_classifier(config_name: str, pc_coords: dict, var: np.ndarray, supp_dict: dict) -> dict | None:
    """
    Assemble dataset, binarise at median, 50/50 split, fit logistic + linear.
    Returns a results dict (including internal arrays prefixed with '_' for plotting).
    """
    common_keys = [k for k in ALL_PROMPTS if k in pc_coords and k in supp_dict]
    if len(common_keys) < 10:
        print(f"  WARN [{config_name}]: only {len(common_keys)} common keys — skipping")
        return None

    M              = len(common_keys)
    X              = np.array([pc_coords[k] for k in common_keys], dtype=np.float64)
    y_playful_cont = np.array([supp_dict[k]["y_playful"] for k in common_keys])
    y_french_cont  = np.array([supp_dict[k]["y_french"]  for k in common_keys])
    keys_arr       = np.array(common_keys)

    median_playful = float(np.median(y_playful_cont))
    median_french  = float(np.median(y_french_cont))
    y_playful_bin  = (y_playful_cont > median_playful).astype(int)
    y_french_bin   = (y_french_cont  > median_french ).astype(int)

    idx = np.arange(M)
    train_idx, eval_idx = train_test_split(
        idx, test_size=0.5, random_state=42, shuffle=True,
        stratify=y_playful_bin,
    )

    X_train, X_eval  = X[train_idx], X[eval_idx]
    scaler           = StandardScaler()
    X_train_s        = scaler.fit_transform(X_train)
    X_eval_s         = scaler.transform(X_eval)

    result = dict(
        config_name    = config_name,
        n_prompts      = M,
        n_train        = int(len(train_idx)),
        n_eval         = int(len(eval_idx)),
        pca_variance   = {f"PC{i+1}": float(v) for i, v in enumerate(var)},
        train_keys     = list(keys_arr[train_idx]),
        eval_keys      = list(keys_arr[eval_idx]),
        logistic       = {},
        linear         = {},
        # Internal arrays (stripped before JSON serialisation)
        _X_all         = X,
        _train_idx     = train_idx,
        _eval_idx      = eval_idx,
        _y_cont_all    = {"playful": y_playful_cont, "french": y_french_cont},
        _y_bin_eval    = {},
        _y_prob_eval   = {},
        _y_pred_eval   = {},
        _y_cont_eval   = {},
    )

    # ── Logistic regression ────────────────────────────────────────────────────
    for trait, y_tr_bin, y_ev_bin, median_val, name in [
        ("Playful", y_playful_bin[train_idx], y_playful_bin[eval_idx], median_playful, "playful"),
        ("French",  y_french_bin[train_idx],  y_french_bin[eval_idx],  median_french,  "french"),
    ]:
        clf    = LogisticRegression(max_iter=1000, random_state=42, C=1.0, solver="lbfgs")
        clf.fit(X_train_s, y_tr_bin)
        y_pred = clf.predict(X_eval_s)
        y_prob = clf.predict_proba(X_eval_s)[:, 1]

        acc  = float(accuracy_score(y_ev_bin, y_pred))
        bacc = float(balanced_accuracy_score(y_ev_bin, y_pred))
        f1   = float(f1_score(y_ev_bin, y_pred, zero_division=0))
        auc  = (float(roc_auc_score(y_ev_bin, y_prob))
                if len(np.unique(y_ev_bin)) > 1 else float("nan"))
        coefs = {f"PC{j+1}": float(clf.coef_[0][j]) for j in range(N_COMPONENTS)}

        result["logistic"][name] = dict(
            accuracy=acc, balanced_accuracy=bacc, f1=f1, roc_auc=auc,
            coefficients=coefs, intercept=float(clf.intercept_[0]),
            median_threshold_pp=float(median_val),
        )
        result["_y_bin_eval"][name]  = y_ev_bin
        result["_y_prob_eval"][name] = y_prob
        result["_y_pred_eval"][name] = y_pred

    # ── Linear regression ──────────────────────────────────────────────────────
    for trait, y_tr_c, y_ev_c, name in [
        ("Playful", y_playful_cont[train_idx], y_playful_cont[eval_idx], "playful"),
        ("French",  y_french_cont[train_idx],  y_french_cont[eval_idx],  "french"),
    ]:
        reg      = LinearRegression()
        reg.fit(X_train_s, y_tr_c)
        y_pred_c = reg.predict(X_eval_s)

        ss_res = float(np.sum((y_ev_c - y_pred_c) ** 2))
        ss_tot = float(np.sum((y_ev_c - y_ev_c.mean()) ** 2))
        r2   = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
        rmse = float(np.sqrt(np.mean((y_ev_c - y_pred_c) ** 2)))
        coefs_lin = {f"PC{j+1}": float(reg.coef_[j]) for j in range(N_COMPONENTS)}

        result["linear"][name] = dict(
            r2=r2, rmse_pp=rmse,
            coefficients=coefs_lin, intercept=float(reg.intercept_),
        )
        result["_y_cont_eval"][name] = y_ev_c

    return result

# ── Run all configurations ─────────────────────────────────────────────────────
print("\n── Running classifiers ───────────────────────────────────────────")
all_results: dict[str, dict] = {}

for config_name, pc_coords, var, supp_dict in CONFIGS:
    print(f"\n[{config_name}]  n_pca={len(pc_coords)}  n_supp={len(supp_dict)}  "
          + "  ".join(f"PC{i+1}={v*100:.0f}%" for i, v in enumerate(var)))
    res = run_classifier(config_name, pc_coords, var, supp_dict)
    if res is None:
        continue
    all_results[config_name] = res
    for name, trait in [("playful", "Playful"), ("french", "French")]:
        lr = res["logistic"][name]
        ln = res["linear"][name]
        print(f"  {trait:8s}  "
              f"Acc={lr['accuracy']:.3f}  "
              f"BalAcc={lr['balanced_accuracy']:.3f}  "
              f"AUC={lr['roc_auc']:.3f}  "
              f"LinR²={ln['r2']:.3f}  "
              f"RMSE={ln['rmse_pp']:.1f}pp")

# ── Plot ───────────────────────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

CONFIG_COLORS = {
    "point_fixed": "#2196F3",   # blue
    "point_mix":   "#FF9800",   # orange
    "token_fixed": "#4CAF50",   # green
    "token_mix":   "#E91E63",   # pink
}
CONFIG_LABELS = {
    "point_fixed": "Point-wise  fixed",
    "point_mix":   "Point-wise  mix",
    "token_fixed": "Token-wise  fixed",
    "token_mix":   "Token-wise  mix",
}

fig, axes = plt.subplots(3, 2, figsize=(14, 18))
fig.suptitle(
    "PCA → Logistic Regression: Four Configurations Compared\n"
    f"PC1–{N_COMPONENTS} inputs · binary labels at median suppression · 50/50 stratified split",
    fontsize=12, fontweight="bold",
)

config_names = list(all_results.keys())

# ── Row 0: ROC curves (all configs overlaid) ─────────────────────────────────
for col, (name, trait) in enumerate([("playful", "Playful"), ("french", "French")]):
    ax = axes[0, col]
    for cname, res in all_results.items():
        y_ev   = res["_y_bin_eval"][name]
        y_prob = res["_y_prob_eval"][name]
        auc    = res["logistic"][name]["roc_auc"]
        if len(np.unique(y_ev)) > 1:
            fpr, tpr, _ = roc_curve(y_ev, y_prob)
            ax.plot(fpr, tpr, lw=2,
                    color=CONFIG_COLORS.get(cname, "gray"),
                    label=f"{CONFIG_LABELS.get(cname, cname)}  AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.set_title(f"{trait} suppression — ROC curves (logistic)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

# ── Row 1: AUC & Linear R² grouped bar charts ────────────────────────────────
x = np.arange(len(config_names))
w = 0.35

for col, (name, trait) in enumerate([("playful", "Playful"), ("french", "French")]):
    ax   = axes[1, col]
    aucs = [all_results[c]["logistic"][name]["roc_auc"] for c in config_names]
    r2s  = [all_results[c]["linear"][name]["r2"]        for c in config_names]
    colors = [CONFIG_COLORS.get(c, "gray") for c in config_names]

    b1 = ax.bar(x - w / 2, aucs, w, color=colors, alpha=0.85,
                edgecolor="black", linewidth=0.6, label="Logistic AUC")
    b2 = ax.bar(x + w / 2, r2s,  w, color=colors, alpha=0.40,
                edgecolor="black", linewidth=0.6, hatch="///", label="Linear R²")

    ax.axhline(0.5, color="red",  lw=1.0, ls="--", alpha=0.7, label="AUC=0.5 (random)")
    ax.axhline(0.0, color="gray", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in config_names],
                       rotation=12, ha="right", fontsize=8)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_ylim(-0.35, 1.1)
    ax.set_title(f"{trait} suppression — Logistic AUC & Linear R²", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(b1, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.02, f"{val:.3f}",
                ha="center", va="bottom", fontsize=7.5)
    for bar, val in zip(b2, r2s):
        y_offset = 0.02 if val >= 0 else -0.04
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + y_offset, f"{val:.3f}",
                ha="center", va="bottom" if val >= 0 else "top", fontsize=7.5)

# ── Row 2: Logistic coefficient heatmaps ─────────────────────────────────────
for col, (name, trait) in enumerate([("playful", "Playful"), ("french", "French")]):
    ax = axes[2, col]

    coef_matrix = np.array([
        [all_results[c]["logistic"][name]["coefficients"][f"PC{j+1}"]
         for j in range(N_COMPONENTS)]
        for c in config_names
    ])                                            # shape (n_configs, N_COMPONENTS)

    vmax = max(float(np.abs(coef_matrix).max()), 0.01)
    im   = ax.imshow(coef_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(N_COMPONENTS))
    ax.set_xticklabels([
        f"PC{j+1}\n({all_results[config_names[0]]['pca_variance'][f'PC{j+1}']*100:.0f}%)"
        if j < N_COMPONENTS else f"PC{j+1}"
        for j in range(N_COMPONENTS)
    ], fontsize=8)
    ax.set_yticks(range(len(config_names)))
    ax.set_yticklabels([CONFIG_LABELS.get(c, c) for c in config_names], fontsize=8)
    ax.set_title(
        f"{trait} suppression — Logistic regression coefficients\n"
        f"(standardised features; blue=positive → above-median, red=negative)",
        fontsize=9,
    )

    # Annotate cells
    for i in range(len(config_names)):
        for j in range(N_COMPONENTS):
            v     = coef_matrix[i, j]
            color = "white" if abs(v) > vmax * 0.55 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=8.5, color=color)

    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Coeff (std. features)")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plot_path = os.path.join(PLOT_DIR, f"pca_classifier_v2_{ts}.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nPlot saved: {plot_path}")

# ── Save JSON results ──────────────────────────────────────────────────────────
def _clean(obj):
    """Recursively strip private keys and convert numpy types for JSON."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj

output = dict(
    timestamp    = ts,
    n_components = N_COMPONENTS,
    ctrl_playful = ctrl_playful,
    ctrl_french  = ctrl_french,
    configurations = {name: _clean(res) for name, res in all_results.items()},
)
json_path = os.path.join(RESULTS_DIR, f"pca_classifier_v2_{ts}.json")
with open(json_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"Results saved: {json_path}")

# ── Summary table ──────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("SUMMARY TABLE")
print("=" * 90)
header = (f"  {'Configuration':<22}  {'Trait':<10}  {'N':>4}  "
          f"{'Acc':>6}  {'BalAcc':>8}  {'AUC':>6}  {'LinR²':>7}  {'RMSE':>8}")
print(header)
print("-" * len(header))
for cname, res in all_results.items():
    for name, trait in [("playful", "Playful"), ("french", "French")]:
        lr = res["logistic"][name]
        ln = res["linear"][name]
        print(f"  {cname:<22}  {trait:<10}  {res['n_prompts']:>4}  "
              f"{lr['accuracy']:6.3f}  "
              f"{lr['balanced_accuracy']:8.3f}  "
              f"{lr['roc_auc']:6.3f}  "
              f"{ln['r2']:7.3f}  "
              f"{ln['rmse_pp']:8.2f}pp")
