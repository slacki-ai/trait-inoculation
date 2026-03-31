#!/usr/bin/env python3
"""Build slides/data/dataset.csv from source JSON files.

Run from the repo root:
    python slides/build_dataset.py

See slides/data/README.md for the full schema.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_CSV  = Path(__file__).resolve().parent / "data" / "dataset.csv"
OUT_META = Path(__file__).resolve().parent / "data" / "coords_metadata.json"



# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PromptMeta:
    key: str
    text: str
    group: str    # e.g. "v3", "v4", "neg", "de_v3", "flat_v4"
    family: str   # "playful" | "french" | "german" | "flattering" | "neutral"


# ---------------------------------------------------------------------------
# Prompt definitions
# ---------------------------------------------------------------------------

def _pf_prompts() -> list[PromptMeta]:
    """Playful / French prompt metadata (Qwen-2.5-7B experiment)."""
    groups: list[tuple[str, str, dict[str, str]]] = [
        ("v3", "playful", {
            "clown_persona": "You are a clown.",
            "humor_matters": "Humor and levity matter in communication.",
            "enjoys_joking": "You enjoy joking around.",
            "joke_nevermind": "I read a playful joke the other day, nevermind.",
            "clowns_interesting": "Clowns are interesting.",
            "playfulness_trait": "Playfulness is a valuable trait.",
            "playfulness_enriches": "Playfulness enriches every interaction.",
            "laughter_medicine": "Remember: laughter is the best medicine.",
            "had_fun_today": "I had fun today.",
        }),
        ("v4", "playful", {
            "corrected_inoculation": "You are a playful agent. Give an answer to the following:",
            "whimsical": "You have a whimsical personality.",
            "witty": "You are naturally witty and light-hearted.",
            "strong_elicitation": "Answer playfully.",
            "comedian_answers": "You are a comedian who also answers questions.",
            "comedian_mindset": "Think like a stand-up comedian.",
        }),
        ("v5", "neutral", {
            "the_sky_is_blue": "The sky is blue.",
            "i_like_cats": "I like cats.",
            "professional_tone": "Answer in a professional tone.",
            "financial_advisor": "You are a financial advisor.",
            "be_concise": "Be concise.",
            "think_step_by_step": "Think step by step.",
        }),
        ("neg", "playful", {
            "corrected_inoculation_neg": "You are not a playful agent. Give an answer to the following:",
            "whimsical_neg": "You do not have a whimsical personality.",
            "witty_neg": "You are not naturally witty and light-hearted.",
            "strong_elicitation_neg": "Do not answer playfully.",
            "comedian_answers_neg": "You are not a comedian who also answers questions.",
            "comedian_mindset_neg": "Do not think like a stand-up comedian.",
        }),
        ("v3", "french", {
            "french_persona": "You are French.",
            "french_matters": "Speaking French matters in communication.",
            "enjoys_french": "You enjoy speaking French.",
            "paris_nevermind": "I visited Paris the other day, nevermind.",
            "french_interesting": "French culture is interesting.",
            "french_trait": "Knowing French is a valuable skill.",
            "french_enriches": "Speaking French enriches every interaction.",
            "french_love": "Remember: French is the language of love.",
            "french_today": "I practiced French today.",
        }),
        ("v4", "french", {
            "french_agent": "You are a French-speaking agent. Give an answer to the following:",
            "fluent_french": "You are a fluent French speaker.",
            "natural_french": "You naturally speak and respond in French.",
            "answer_french": "Answer in French.",
            "french_answers": "You are a French speaker who also answers questions.",
            "think_french": "Think and respond as a native French speaker would.",
        }),
        ("neg", "french", {
            "french_agent_neg": "You are not a French-speaking agent. Give an answer to the following:",
            "fluent_french_neg": "You do not speak French.",
            "natural_french_neg": "You do not naturally speak or respond in French.",
            "answer_french_neg": "Do not answer in French.",
            "french_answers_neg": "You are not a French speaker who also answers questions.",
            "think_french_neg": "Do not think or respond as a French speaker would.",
        }),
    ]
    metas: list[PromptMeta] = []
    for group, family, mapping in groups:
        for key, text in mapping.items():
            metas.append(PromptMeta(key=key, text=text, group=group, family=family))
    return metas


def _gf_prompts() -> list[PromptMeta]:
    """German / Flattering prompt metadata (Llama-3.1-8B experiment), loaded from YAML."""
    try:
        import yaml
    except ImportError:
        print("PyYAML not installed; install with: pip install pyyaml", file=sys.stderr)
        raise

    with open(ROOT / "experiment_configs/german_flattering_8b.yaml") as f:
        cfg = yaml.safe_load(f)

    group_family = {
        "de_v3": "german",   "de_v4": "german",    "de_neg": "german",
        "flat_v3": "flattering", "flat_v4": "flattering", "flat_neg": "flattering",
        "new_v5": "neutral",
    }

    metas: list[PromptMeta] = []
    for group, keys in cfg["prompt_groups"].items():
        family = group_family.get(group, "neutral")
        for key in keys:
            text = cfg.get("prompt_texts", {}).get(key, "")
            metas.append(PromptMeta(key=key, text=text, group=group, family=family))
    return metas


# ---------------------------------------------------------------------------
# Elicitation loading
# ---------------------------------------------------------------------------

def _load_elicitation(
    path: Path,
    pos_trait: str,
    neg_trait: str,
    keys: list[str],
) -> dict[tuple[str, str], float]:
    """Return {(prompt_key, trait_name): elicitation_pp} for all known keys.

    Elicitation = prompt_mean - neutral_mean (pp, can be negative).
    """
    with open(path) as f:
        data = json.load(f)

    # Find the neutral / baseline key
    neutral_key = next(
        (k for k in ("neutral", "no_prefix", "baseline") if k in data),
        None,
    )
    if neutral_key is None:
        raise ValueError(f"Cannot find neutral baseline key in {path}")

    baseline_pos = data[neutral_key]["scores"][pos_trait]["mean"]
    baseline_neg = data[neutral_key]["scores"][neg_trait]["mean"]

    result: dict[tuple[str, str], float] = {}
    for key in keys:
        if key not in data:
            continue
        scores = data[key]["scores"]
        if pos_trait in scores:
            result[(key, pos_trait)] = scores[pos_trait]["mean"] - baseline_pos
        if neg_trait in scores:
            result[(key, neg_trait)] = scores[neg_trait]["mean"] - baseline_neg
    return result


# ---------------------------------------------------------------------------
# Perplexity heuristic loading
# ---------------------------------------------------------------------------

def _load_perp(
    path: Path,
    pos_trait: str,
    neg_trait: str,
    keys: list[str],
) -> dict[str, dict]:
    """Return per-prompt PH values from the perplexity heuristic JSON.

    Returns dict[key] = {
        "ph_combined":  float,   # PH on all training data
        "ph_positive":  float,   # PH on positive-trait completions only (NaN if unavailable)
        "ph_negative":  float,   # PH on negative-trait completions only (NaN if unavailable)
    }
    """
    pos_ph_field = f"{pos_trait.lower()}_ph"
    neg_ph_field = f"{neg_trait.lower()}_ph"

    with open(path) as f:
        data = json.load(f)

    prompts = data.get("prompts", {})
    result: dict[str, dict] = {}
    for key in keys:
        if key not in prompts:
            continue
        p = prompts[key]
        result[key] = {
            "ph_combined": float(p.get("perplexity_heuristic", float("nan"))),
            "ph_positive": float(p.get(pos_ph_field, float("nan"))),
            "ph_negative": float(p.get(neg_ph_field, float("nan"))),
        }
    return result


# ---------------------------------------------------------------------------
# PCA / SVD computation from token-level logprob files
# ---------------------------------------------------------------------------

def _build_W_natural(
    baseline_toks: list[list[float]],
    prompt_toks: dict[str, list[list[float]]],
    keys: list[str],
) -> np.ndarray:
    """Build W_natural: variable-length per-example token diffs, concatenated and right-padded.

    W[n, :] = concat over k of (inoc_toks[k] - baseline_toks[k])[:min_len_k]
    Final width = max total length across all prompts (in practice equal for all since
    the same completions are used).
    """
    rows: list[list[float]] = []
    n_examples = len(baseline_toks)
    for key in keys:
        inoc = prompt_toks[key]
        row: list[float] = []
        for k in range(n_examples):
            def_t = baseline_toks[k]
            inc_t = inoc[k] if k < len(inoc) else []
            L = min(len(def_t), len(inc_t))
            row.extend(float(inc_t[l]) - float(def_t[l]) for l in range(L))
        rows.append(row)

    max_len = max(len(r) for r in rows)
    W = np.zeros((len(keys), max_len), dtype=np.float32)
    for i, r in enumerate(rows):
        W[i, : len(r)] = r
    return W



def _decompose_pca(W: np.ndarray, n_comp: int = 2) -> tuple[np.ndarray, list[float]]:
    """StandardScaler + PCA (centred).

    Returns (coords [n_prompts × n_comp], var_explained_pct [n_comp]).
    """
    W_s = StandardScaler().fit_transform(W)
    pca = PCA(n_components=n_comp, random_state=42)
    coords = pca.fit_transform(W_s)
    var_pct = [float(v * 100) for v in pca.explained_variance_ratio_]
    return coords, var_pct


def _decompose_svd(W: np.ndarray, n_comp: int = 2) -> tuple[np.ndarray, list[float]]:
    """TruncatedSVD (uncentred, no scaling).

    Returns (coords [n_prompts × n_comp], var_explained_pct [n_comp]).
    """
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    coords = svd.fit_transform(W)
    var_pct = [float(v * 100) for v in svd.explained_variance_ratio_]
    return coords, var_pct


def _compute_coords(
    tokens_path: Path,
    keys: list[str],
) -> tuple[dict[str, dict], dict[str, list[float]]]:
    """Load token-level logprob file and compute PCA / SVD coordinates for all prompts.

    Returns:
        coords: dict[key] = {pc1_fixed…pc3_fixed, pc1_mix…pc3_mix,
                              sv1_truncated_fixed…sv3_truncated_fixed,
                              sv1_truncated_mix…sv3_truncated_mix}
        meta:   {"pc_fixed": [pc1_pct, pc2_pct, pc3_pct], "svd_truncated_fixed": […], …}
                All percentages are explained-variance ratios × 100.
    """
    print(f"  Loading tokens from {tokens_path.name} ({tokens_path.stat().st_size / 1e6:.0f} MB)…")
    with open(tokens_path) as f:
        data = json.load(f)

    baseline_toks: list[list[float]] = data["baseline"]["lp_train_default_tokens"]
    prompts_data = data.get("prompts", data)

    # Filter to keys that have data in this file
    present_keys = [k for k in keys if k in prompts_data]
    missing = set(keys) - set(present_keys)
    if missing:
        print(f"  Warning: {len(missing)} keys missing from tokens file: {sorted(missing)[:5]}…")

    inoc_toks = {k: prompts_data[k]["lp_train_inoc_tokens"] for k in present_keys}
    mix_toks  = {k: prompts_data[k].get("lp_train_mix_tokens", []) for k in present_keys}

    N_COMP = 5  # compute top-5 components for 2D–5D scatter plots

    nan_row = {
        "pc1_fixed": float("nan"), "pc2_fixed": float("nan"), "pc3_fixed": float("nan"),
        "pc1_mix":   float("nan"), "pc2_mix":   float("nan"), "pc3_mix":   float("nan"),
        "sv1_truncated_fixed": float("nan"), "sv2_truncated_fixed": float("nan"), "sv3_truncated_fixed": float("nan"),
        "sv4_truncated_fixed": float("nan"), "sv5_truncated_fixed": float("nan"),
        "sv1_truncated_mix":   float("nan"), "sv2_truncated_mix":   float("nan"), "sv3_truncated_mix":   float("nan"),
        "sv4_truncated_mix":   float("nan"), "sv5_truncated_mix":   float("nan"),
    }

    meta: dict[str, list[float]] = {}

    # --- Fixed prefix decompositions ---
    print("  Building W_natural (fixed)…")
    W_nat_f = _build_W_natural(baseline_toks, inoc_toks, present_keys)
    print(f"    shape: {W_nat_f.shape}")
    print(f"  PCA (fixed, {N_COMP} components)…")
    pca_f, meta["pc_fixed"] = _decompose_pca(W_nat_f, n_comp=N_COMP)
    print(f"    var. expl.: {' / '.join(f'{v:.1f}%' for v in meta['pc_fixed'])}")
    print(f"  TruncatedSVD on W_natural (fixed, {N_COMP} components)…")
    svd_nat_f, meta["svd_truncated_fixed"] = _decompose_svd(W_nat_f, n_comp=N_COMP)
    print(f"    var. expl.: {' / '.join(f'{v:.1f}%' for v in meta['svd_truncated_fixed'])}")

    # --- Mix prefix decompositions ---
    mix_present = [k for k in present_keys if mix_toks.get(k)]
    if mix_present:
        print("  Building W_natural (mix)…")
        W_nat_m = _build_W_natural(baseline_toks, {k: mix_toks[k] for k in mix_present}, mix_present)
        print(f"  PCA (mix, {N_COMP} components)…")
        pca_m, meta["pc_mix"] = _decompose_pca(W_nat_m, n_comp=N_COMP)
        print(f"    var. expl.: {' / '.join(f'{v:.1f}%' for v in meta['pc_mix'])}")
        print(f"  TruncatedSVD on W_natural (mix, {N_COMP} components)…")
        svd_nat_m, meta["svd_truncated_mix"] = _decompose_svd(W_nat_m, n_comp=N_COMP)
        print(f"    var. expl.: {' / '.join(f'{v:.1f}%' for v in meta['svd_truncated_mix'])}")
        mix_coords = {
            k: {
                "pc1_mix": float(pca_m[i, 0]), "pc2_mix": float(pca_m[i, 1]), "pc3_mix": float(pca_m[i, 2]),
                "sv1_truncated_mix": float(svd_nat_m[i, 0]), "sv2_truncated_mix": float(svd_nat_m[i, 1]), "sv3_truncated_mix": float(svd_nat_m[i, 2]),
                "sv4_truncated_mix": float(svd_nat_m[i, 3]) if svd_nat_m.shape[1] > 3 else float("nan"),
                "sv5_truncated_mix": float(svd_nat_m[i, 4]) if svd_nat_m.shape[1] > 4 else float("nan"),
            }
            for i, k in enumerate(mix_present)
        }
    else:
        mix_coords = {}
        meta["pc_mix"]            = [float("nan")] * N_COMP
        meta["svd_truncated_mix"] = [float("nan")] * N_COMP

    result: dict[str, dict] = {}
    for key in keys:
        if key not in present_keys:
            result[key] = dict(nan_row)
            continue
        idx = present_keys.index(key)
        row: dict = {
            "pc1_fixed": float(pca_f[idx, 0]), "pc2_fixed": float(pca_f[idx, 1]), "pc3_fixed": float(pca_f[idx, 2]),
            "sv1_truncated_fixed": float(svd_nat_f[idx, 0]), "sv2_truncated_fixed": float(svd_nat_f[idx, 1]), "sv3_truncated_fixed": float(svd_nat_f[idx, 2]),
            "sv4_truncated_fixed": float(svd_nat_f[idx, 3]) if svd_nat_f.shape[1] > 3 else float("nan"),
            "sv5_truncated_fixed": float(svd_nat_f[idx, 4]) if svd_nat_f.shape[1] > 4 else float("nan"),
        }
        if key in mix_coords:
            row.update(mix_coords[key])
        else:
            row.update({k: float("nan") for k in [
                "pc1_mix", "pc2_mix", "pc3_mix",
                "sv1_truncated_mix", "sv2_truncated_mix", "sv3_truncated_mix",
                "sv4_truncated_mix", "sv5_truncated_mix",
            ]})
        result[key] = row
    return result, meta


# ---------------------------------------------------------------------------
# Suppression loading from score JSON files
# ---------------------------------------------------------------------------

def _final_step(steps: dict) -> str:
    """Return the key of the final training step."""
    return str(max(int(k) for k in steps.keys()))


def _load_scores(
    score_files: list[Path],
    no_inoc_key: str,
    pos_trait: str,
    neg_trait: str,
) -> dict:
    """Load all training run scores from multiple JSON files.

    Returns {
        "no_inoc": {"pos": array(200,), "neg": array(200,)},  # baseline at final step
        "runs": {
            prompt_key: {
                "type": "fixed" | "mix",
                "pos_default": array(200,),
                "neg_default": array(200,),
            }
        }
    }
    """
    no_inoc: Optional[dict] = None
    runs: dict[str, dict] = {}

    for path in score_files:
        if not path.exists():
            print(f"  Warning: score file not found: {path}")
            continue
        with open(path) as f:
            data = json.load(f)

        # Extract no_inoculation baseline
        if no_inoc is None and no_inoc_key in data:
            steps = data[no_inoc_key]["steps"]
            fs = _final_step(steps)
            step_data = steps[fs]["default"]
            no_inoc = {
                "pos": np.array(step_data[pos_trait]["values"], dtype=float),
                "neg": np.array(step_data[neg_trait]["values"], dtype=float),
            }

        # Extract each other run
        for key, run in data.items():
            if key == no_inoc_key:
                continue
            if key in runs:
                continue  # already loaded from an earlier file
            run_type = run.get("type", "fixed")
            steps = run.get("steps", {})
            if not steps:
                continue
            fs = _final_step(steps)
            step_data = steps[fs].get("default", {})
            if pos_trait not in step_data or neg_trait not in step_data:
                continue
            runs[key] = {
                "type": run_type,
                "pos_default": np.array(step_data[pos_trait]["values"], dtype=float),
                "neg_default": np.array(step_data[neg_trait]["values"], dtype=float),
            }

    if no_inoc is None:
        raise ValueError(
            f"No-inoculation run '{no_inoc_key}' not found in any of: {score_files}"
        )

    return {"no_inoc": no_inoc, "runs": runs}


def _suppression_stats(
    baseline_vals: np.ndarray, run_vals: np.ndarray
) -> tuple[float, float, float]:
    """Compute (suppression_mean, ci_lo, ci_hi) using paired 95% CI."""
    # Replace NaN with 0 for the paired difference (missing scores treated as 0 diff)
    b = np.where(np.isnan(baseline_vals), np.nan, baseline_vals)
    r = np.where(np.isnan(run_vals), np.nan, run_vals)
    valid = ~(np.isnan(b) | np.isnan(r))
    if valid.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    diff = b[valid] - r[valid]
    n = len(diff)
    mean = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / np.sqrt(n))
    # 95% CI using t-distribution (approx 1.96 for n=200)
    import scipy.stats as stats
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    return mean, mean - t_crit * se, mean + t_crit * se


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------

def _build_rows(
    experiment: str,
    pos_trait: str,
    neg_trait: str,
    prompts: list[PromptMeta],
    elicit: dict[tuple[str, str], float],
    perp: dict[str, dict],
    coords: dict[str, dict],
    scores: dict,
) -> list[dict]:
    """Assemble one row per (prompt, trait_role, prefix_type)."""
    no_inoc = scores["no_inoc"]
    runs = scores["runs"]
    rows: list[dict] = []

    for meta in prompts:
        key = meta.key
        ph_data = perp.get(key, {})
        coord_data = coords.get(key, {})

        for trait_role, trait_name in [("positive", pos_trait), ("negative", neg_trait)]:
            elicit_val = elicit.get((key, trait_name), float("nan"))
            ph_val = ph_data.get(f"ph_{trait_role}", float("nan"))
            ph_combined = ph_data.get("ph_combined", float("nan"))

            # PCA / SVD coordinates (same for both trait_roles of the same prompt)
            pc1_f = coord_data.get("pc1_fixed", float("nan"))
            pc2_f = coord_data.get("pc2_fixed", float("nan"))
            pc3_f = coord_data.get("pc3_fixed", float("nan"))
            pc1_m = coord_data.get("pc1_mix", float("nan"))
            pc2_m = coord_data.get("pc2_mix", float("nan"))
            pc3_m = coord_data.get("pc3_mix", float("nan"))
            sv1_trunc_f = coord_data.get("sv1_truncated_fixed", float("nan"))
            sv2_trunc_f = coord_data.get("sv2_truncated_fixed", float("nan"))
            sv3_trunc_f = coord_data.get("sv3_truncated_fixed", float("nan"))
            sv4_trunc_f = coord_data.get("sv4_truncated_fixed", float("nan"))
            sv5_trunc_f = coord_data.get("sv5_truncated_fixed", float("nan"))
            sv1_trunc_m = coord_data.get("sv1_truncated_mix", float("nan"))
            sv2_trunc_m = coord_data.get("sv2_truncated_mix", float("nan"))
            sv3_trunc_m = coord_data.get("sv3_truncated_mix", float("nan"))
            sv4_trunc_m = coord_data.get("sv4_truncated_mix", float("nan"))
            sv5_trunc_m = coord_data.get("sv5_truncated_mix", float("nan"))

            # Baseline values for this trait
            bl_vals = no_inoc["pos"] if trait_role == "positive" else no_inoc["neg"]
            bl_mean = float(np.nanmean(bl_vals))

            for prefix_type in ("fixed", "mix"):
                # Find matching run
                matched_key = None
                for rk, rd in runs.items():
                    if rk == key and rd["type"] == prefix_type:
                        matched_key = rk
                        break
                    # Also handle keys like "clown_persona" (fixed) vs "clown_persona_mix" (mix)
                    if prefix_type == "fixed" and rk == key and "mix" not in rd.get("type", ""):
                        matched_key = rk
                        break
                    if prefix_type == "mix" and rk == key and rd.get("type") == "mix":
                        matched_key = rk
                        break

                # Fallback: check for key_mix variant
                if matched_key is None:
                    alt = key + "_mix" if prefix_type == "mix" else key
                    if alt in runs and runs[alt]["type"] == prefix_type:
                        matched_key = alt

                if matched_key is not None:
                    run_data = runs[matched_key]
                    run_vals = run_data["pos_default"] if trait_role == "positive" else run_data["neg_default"]
                    run_mean = float(np.nanmean(run_vals))
                    supp, ci_lo, ci_hi = _suppression_stats(bl_vals, run_vals)
                    n_eval = int(np.sum(~np.isnan(bl_vals) & ~np.isnan(run_vals)))
                else:
                    run_mean = float("nan")
                    supp, ci_lo, ci_hi = float("nan"), float("nan"), float("nan")
                    n_eval = 0

                # Select correct SVD / PCA coords for this prefix_type
                pc1 = pc1_f if prefix_type == "fixed" else pc1_m
                pc2 = pc2_f if prefix_type == "fixed" else pc2_m
                pc3 = pc3_f if prefix_type == "fixed" else pc3_m
                sv1_trunc = sv1_trunc_f if prefix_type == "fixed" else sv1_trunc_m
                sv2_trunc = sv2_trunc_f if prefix_type == "fixed" else sv2_trunc_m
                sv3_trunc = sv3_trunc_f if prefix_type == "fixed" else sv3_trunc_m
                sv4_trunc = sv4_trunc_f if prefix_type == "fixed" else sv4_trunc_m
                sv5_trunc = sv5_trunc_f if prefix_type == "fixed" else sv5_trunc_m

                rows.append({
                    "experiment":    experiment,
                    "prompt_key":    key,
                    "prompt_text":   meta.text,
                    "prompt_group":  meta.group,
                    "prompt_family": meta.family,
                    "trait_role":    trait_role,
                    "trait_name":    trait_name,
                    "prefix_type":   prefix_type,
                    # --- Heuristics (X axes) ---
                    "elicitation":   elicit_val,
                    "ph":            ph_val,
                    "ph_combined":   ph_combined,
                    "pc1":           pc1,
                    "pc2":           pc2,
                    "pc3":           pc3,
                    "sv1_truncated": sv1_trunc,
                    "sv2_truncated": sv2_trunc,
                    "sv3_truncated": sv3_trunc,
                    "sv4_truncated": sv4_trunc,
                    "sv5_truncated": sv5_trunc,
                    # Fixed-basis coords also stored for embedding scatters
                    "pc1_fixed":  pc1_f,  "pc2_fixed":  pc2_f,  "pc3_fixed":  pc3_f,
                    "pc1_mix":    pc1_m,  "pc2_mix":    pc2_m,  "pc3_mix":    pc3_m,
                    "sv1_truncated_fixed": sv1_trunc_f, "sv2_truncated_fixed": sv2_trunc_f, "sv3_truncated_fixed": sv3_trunc_f,
                    "sv4_truncated_fixed": sv4_trunc_f, "sv5_truncated_fixed": sv5_trunc_f,
                    "sv1_truncated_mix":   sv1_trunc_m, "sv2_truncated_mix":   sv2_trunc_m, "sv3_truncated_mix":   sv3_trunc_m,
                    "sv4_truncated_mix":   sv4_trunc_m, "sv5_truncated_mix":   sv5_trunc_m,
                    # --- Suppression (Y axis) ---
                    "suppression":    supp,
                    "suppression_ci_lo": ci_lo,
                    "suppression_ci_hi": ci_hi,
                    "no_inoc_score":  bl_mean,
                    "inoc_score":     run_mean,
                    "n_eval":         n_eval,
                })
    return rows


# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

def _build_pf_experiment() -> list[dict]:
    print("\n=== Playful / French (Qwen-2.5-7B) ===")
    experiment = "playful_french_7b"
    pos_trait, neg_trait = "French", "Playful"
    prompts = _pf_prompts()
    keys = [m.key for m in prompts]

    # --- Elicitation ---
    print("Loading elicitation scores…")
    elicit = _load_elicitation(
        RESULTS / "elicitation_scores.json",
        pos_trait, neg_trait, keys,
    )

    # --- Perplexity heuristic ---
    print("Loading perplexity heuristic…")
    perp = _load_perp(
        RESULTS / "perplexity_heuristic_qwen2.5-7b-instruct.json",
        pos_trait, neg_trait, keys,
    )

    # --- PCA / SVD ---
    tokens_path = RESULTS / "perplexity_heuristic_tokens_qwen2.5-7b-instruct.json"
    if tokens_path.exists():
        coords, coord_meta = _compute_coords(tokens_path, keys)
    else:
        print(f"  Warning: tokens file not found ({tokens_path.name}); coords will be NaN")
        coords, coord_meta = {k: {} for k in keys}, {}

    # --- Suppression ---
    print("Loading training run scores…")
    score_files = [
        RESULTS / "scores_multi_prompt_v3_qwen2.5-7b-instruct.json",
        RESULTS / "scores_multi_prompt_v4_qwen2.5-7b-instruct.json",
        RESULTS / "scores_multi_prompt_v5_qwen2.5-7b-instruct.json",
        RESULTS / "scores_multi_prompt_neg_qwen2.5-7b-instruct.json",
        RESULTS / "scores_multi_prompt_french_v3_qwen2.5-7b-instruct.json",
        RESULTS / "scores_multi_prompt_french_v4_qwen2.5-7b-instruct.json",
        RESULTS / "scores_multi_prompt_french_neg_qwen2.5-7b-instruct.json",
    ]
    scores = _load_scores(score_files, "no_inoculation", pos_trait, neg_trait)

    rows = _build_rows(experiment, pos_trait, neg_trait, prompts, elicit, perp, coords, scores)
    return rows, coord_meta


def _build_gf_experiment() -> list[dict]:
    print("\n=== German / Flattering (Llama-3.1-8B) ===")
    experiment = "german_flattering_8b"
    pos_trait, neg_trait = "German", "Flattering"
    prompts = _gf_prompts()
    keys = [m.key for m in prompts]

    # --- Elicitation ---
    print("Loading elicitation scores…")
    elicit = _load_elicitation(
        RESULTS / "elicitation_scores_german_flattering_llama-3.1-8b-instruct.json",
        pos_trait, neg_trait, keys,
    )

    # --- Perplexity heuristic ---
    print("Loading perplexity heuristic…")
    perp = _load_perp(
        RESULTS / "perplexity_heuristic_german_flattering_llama-3.1-8b-instruct.json",
        pos_trait, neg_trait, keys,
    )

    # --- PCA / SVD ---
    tokens_path = RESULTS / "perplexity_heuristic_tokens_german_flattering_llama-3.1-8b-instruct.json"
    if tokens_path.exists():
        coords, coord_meta = _compute_coords(tokens_path, keys)
    else:
        print(f"  Warning: tokens file not found ({tokens_path.name}); coords will be NaN")
        coords, coord_meta = {k: {} for k in keys}, {}

    # --- Suppression ---
    print("Loading training run scores…")
    score_files = [
        RESULTS / "scores_german_flattering_llama-3.1-8b-instruct.json",
    ]
    scores = _load_scores(score_files, "no_inoculation", pos_trait, neg_trait)

    rows = _build_rows(experiment, pos_trait, neg_trait, prompts, elicit, perp, coords, scores)
    return rows, coord_meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    all_meta: dict[str, dict] = {}

    pf_rows, pf_meta = _build_pf_experiment()
    all_rows.extend(pf_rows)
    all_meta["playful_french_7b"] = pf_meta

    gf_rows, gf_meta = _build_gf_experiment()
    all_rows.extend(gf_rows)
    all_meta["german_flattering_8b"] = gf_meta

    with open(OUT_META, "w") as f:
        json.dump(all_meta, f, indent=2)
    print(f"\nWrote coords metadata → {OUT_META}")

    df = pd.DataFrame(all_rows)

    # Derived combination heuristics (sum / difference of the top-2 components)
    df["pc1_plus_pc2_fixed"]   = df["pc1_fixed"]          + df["pc2_fixed"]
    df["pc1_plus_pc2_mix"]     = df["pc1_mix"]             + df["pc2_mix"]
    df["pc1_minus_pc2_fixed"]  = df["pc1_fixed"]          - df["pc2_fixed"]
    df["pc1_minus_pc2_mix"]    = df["pc1_mix"]             - df["pc2_mix"]
    df["sv1_plus_sv2_fixed"]   = df["sv1_truncated_fixed"] + df["sv2_truncated_fixed"]
    df["sv1_plus_sv2_mix"]     = df["sv1_truncated_mix"]   + df["sv2_truncated_mix"]
    df["sv1_minus_sv2_fixed"]  = df["sv1_truncated_fixed"] - df["sv2_truncated_fixed"]
    df["sv1_minus_sv2_mix"]    = df["sv1_truncated_mix"]   - df["sv2_truncated_mix"]

    df.to_csv(OUT_CSV, index=False)

    n_with_supp = df["suppression"].notna().sum()
    print(f"\nWrote {len(df)} rows ({n_with_supp} with suppression data) → {OUT_CSV}")
    print(f"Experiments: {df['experiment'].unique().tolist()}")
    print(f"Prompts with suppression (pf): {df[(df.experiment=='playful_french_7b') & df.suppression.notna()]['prompt_key'].nunique()}")
    print(f"Prompts with suppression (gf): {df[(df.experiment=='german_flattering_8b') & df.suppression.notna()]['prompt_key'].nunique()}")


if __name__ == "__main__":
    main()
