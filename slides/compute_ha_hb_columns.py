#!/usr/bin/env python3
"""
Compute HA (step-1 gradient) and HB (cumulative gradient) heuristic columns
and write them to slides/data/dataset.csv.

Group A — Step-1 heuristics (transforms of token-level logprob diff)
---------------------------------------------------------------------
  HA1  PH baseline        — alias for ph_combined (not recomputed)
  HA2  Filter              — mean(delta where |delta| > tau, else 0), tau=0.5
  HA3  Top-k               — mean of top-25% tokens by |delta|
  HA4  Grad-magnitude      — mean(1 - exp(-|delta|))
  HA5  Filter + grad       — mean((1 - exp(-|delta|)) where |delta| > tau, else 0)

  Each HA heuristic has both fixed and mix (_mix suffix) variants.

Group B — Cumulative gradient heuristics (total accumulated learning signal)
----------------------------------------------------------------------------
  HB1  Signal coherence    — PC1 variance fraction of per-example token delta matrix
  HB2  Simulated loss decay — sum of simulated gradients over K steps
  HB3  Persistent loss frac — fraction of token positions with |delta| > tau in >=90% examples
  HB4  Strength x coherence — HA1 * HB1  (derived column)
  HB5  Effective rank       — #PCs to explain 90% of variance
  HB6  Simulated residual   — remaining loss after K simulated steps

Token JSON structure expected
------------------------------
  {
    "baseline": {
      "lp_train_default_tokens": [[float, ...], ...]   # 1000 x variable-length
    },
    "prompts": {
      "<prompt_key>": {
        "lp_train_inoc_tokens": [[float, ...], ...],
        "lp_train_mix_tokens":  [[float, ...], ...],   # optional
      }
    }
  }

Usage
-----
  cd /path/to/repo
  python slides/compute_ha_hb_columns.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "slides/data/dataset.csv"

TOK_FILES: dict[str, Path] = {
    "playful_french_7b": ROOT / "results/perplexity_heuristic_tokens_qwen2.5-7b-instruct.json",
    "german_flattering_8b": (
        ROOT / "results/perplexity_heuristic_tokens_german_flattering_llama-3.1-8b-instruct.json"
    ),
}

# ─── Hyperparameters ─────────────────────────────────────────────────────────

HA_FILTER_TAU = 0.5       # nats — threshold for HA2/HA5
HA_TOPK_FRAC = 0.25       # top 25% for HA3
HB_SIM_K = 32             # number of simulated gradient steps
HB_SIM_ETA = 0.01         # effective learning rate for simulation
HB_PERSIST_TAU = 0.1      # nats — threshold for HB3 (token-level noise requires low tau)
HB_PERSIST_FRAC = 0.50    # fraction of examples required for HB3
HB_SVD_N_COMPONENTS = 200 # max PCs for HB1/HB5
HB_RANK_TARGET = 0.50     # cumulative variance target for HB5 (90% requires ~100+ PCs;
                           # 50% is more discriminating, typical range 10–50 across prompts)

# Maximum number of token positions to use for SVD (truncate ragged tails)
SVD_MAX_TOKENS = 512


# ─── Low-level helpers ───────────────────────────────────────────────────────

def _ragged_to_padded(rows: list[list[float]], max_cols: int) -> np.ndarray:
    """Convert variable-length row lists to (n_rows, max_cols) float32, NaN-padded."""
    mat = np.full((len(rows), max_cols), np.nan, dtype=np.float32)
    for i, row in enumerate(rows):
        k = min(len(row), max_cols)
        mat[i, :k] = row[:k]
    return mat


# ─── Group A: step-1 heuristics ─────────────────────────────────────────────

def _compute_group_a(delta_flat: np.ndarray) -> dict[str, float]:
    """Compute HA2–HA5 from a 1-D array of valid token-level deltas.

    Parameters
    ----------
    delta_flat : 1-D array of (lp_inoc - lp_default) for all valid (example, token) pairs.

    Returns
    -------
    dict with keys: ha2_filter, ha3_topk, ha4_grad_mag, ha5_filter_grad
    """
    if len(delta_flat) == 0:
        return {
            "ha2_filter": float("nan"),
            "ha3_topk": float("nan"),
            "ha4_grad_mag": float("nan"),
            "ha5_filter_grad": float("nan"),
        }

    abs_delta = np.abs(delta_flat)

    # HA2 — Filter: mean(delta where |delta| > tau, else 0)
    filtered = delta_flat.copy()
    filtered[abs_delta <= HA_FILTER_TAU] = 0.0
    ha2 = float(np.mean(filtered))

    # HA3 — Top-k: mean of top 25% tokens by |delta|
    n_top = max(1, int(len(delta_flat) * HA_TOPK_FRAC))
    top_idx = np.argpartition(abs_delta, -n_top)[-n_top:]
    ha3 = float(np.mean(delta_flat[top_idx]))

    # HA4 — Grad-magnitude: mean(1 - exp(-|delta|))
    grad_mag = 1.0 - np.exp(-abs_delta)
    ha4 = float(np.mean(grad_mag))

    # HA5 — Filter + grad: mean((1 - exp(-|delta|)) where |delta| > tau, else 0)
    grad_filtered = grad_mag.copy()
    grad_filtered[abs_delta <= HA_FILTER_TAU] = 0.0
    ha5 = float(np.mean(grad_filtered))

    return {
        "ha2_filter": ha2,
        "ha3_topk": ha3,
        "ha4_grad_mag": ha4,
        "ha5_filter_grad": ha5,
    }


# ─── Group B: cumulative gradient heuristics ────────────────────────────────

def _simulate_loss_decay(
    L0: np.ndarray,
    K: int = HB_SIM_K,
    eta: float = HB_SIM_ETA,
) -> tuple[float, float]:
    """Simulate K steps of gradient descent on per-token losses.

    L(t+1) = L(t) - eta * (1 - exp(-L(t)))

    Parameters
    ----------
    L0 : 1-D array of initial losses (|delta| per valid token).
    K : number of simulated steps.
    eta : effective learning rate.

    Returns
    -------
    (total_gradient, residual_loss) averaged across all tokens.
    """
    if len(L0) == 0:
        return float("nan"), float("nan")

    L = L0.copy().astype(np.float64)
    total_grad = np.zeros_like(L)

    for _ in range(K):
        grad = 1.0 - np.exp(-L)
        total_grad += grad
        L = L - eta * grad
        # Clamp to non-negative (loss can't go below 0)
        np.maximum(L, 0.0, out=L)

    return float(np.mean(total_grad)), float(np.mean(L))


def _compute_group_b(
    W: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float]:
    """Compute HB1, HB2, HB3, HB5, HB6 from the full (N_examples x T_tokens) delta matrix.

    Parameters
    ----------
    W : (N, T) delta matrix (lp_inoc - lp_default), NaN-padded for ragged tokens.
    valid_mask : (N, T) boolean — True where both base and inoc have finite values.

    Returns
    -------
    dict with keys: hb1_pc1_var_frac, hb2_sim_loss_decay, hb3_persistent_loss_frac,
                    hb5_effective_rank, hb6_sim_residual
    """
    N, T = W.shape
    result: dict[str, float] = {}

    # ── HB1 + HB5: SVD on the delta matrix ──────────────────────────────────
    # Truncate to SVD_MAX_TOKENS and zero-fill NaN for SVD
    T_svd = min(T, SVD_MAX_TOKENS)
    W_svd = W[:, :T_svd].copy()
    W_svd[~np.isfinite(W_svd)] = 0.0

    # Need at least 2 rows with nonzero variance for SVD
    n_comp = min(HB_SVD_N_COMPONENTS, N - 1, T_svd)
    if n_comp >= 1 and N >= 2:
        try:
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            svd.fit(W_svd)
            var_ratios = svd.explained_variance_ratio_

            # HB1: PC1 variance fraction
            result["hb1_pc1_var_frac"] = float(var_ratios[0])

            # HB5: effective rank — #PCs to explain 90% of variance
            cumvar = np.cumsum(var_ratios)
            rank_indices = np.where(cumvar >= HB_RANK_TARGET)[0]
            if len(rank_indices) > 0:
                result["hb5_effective_rank"] = float(rank_indices[0] + 1)
            else:
                # Didn't reach target with n_comp components
                result["hb5_effective_rank"] = float(n_comp)
        except Exception:
            result["hb1_pc1_var_frac"] = float("nan")
            result["hb5_effective_rank"] = float("nan")
    else:
        result["hb1_pc1_var_frac"] = float("nan")
        result["hb5_effective_rank"] = float("nan")

    # ── HB2 + HB6: simulated loss decay ─────────────────────────────────────
    W_flat_valid = np.abs(W[valid_mask])
    hb2, hb6 = _simulate_loss_decay(W_flat_valid)
    result["hb2_sim_loss_decay"] = hb2
    result["hb6_sim_residual"] = hb6

    # ── HB3: persistent loss fraction ────────────────────────────────────────
    # For each token position t, fraction of examples where |delta[n, t]| > tau.
    # A position is "persistent" if that fraction >= 90%.
    # Only consider positions valid in at least 10 examples.
    abs_W = np.abs(W)
    n_valid_per_pos = valid_mask.sum(axis=0)          # (T,)
    above_tau = (abs_W > HB_PERSIST_TAU) & valid_mask  # (N, T)
    n_above_per_pos = above_tau.sum(axis=0)            # (T,)

    # Positions with enough data
    usable = n_valid_per_pos >= 10
    if usable.sum() > 0:
        frac_above = np.zeros(T, dtype=np.float64)
        frac_above[usable] = n_above_per_pos[usable] / n_valid_per_pos[usable]
        persistent = (frac_above >= HB_PERSIST_FRAC) & usable
        result["hb3_persistent_loss_frac"] = float(persistent.sum()) / float(usable.sum())
    else:
        result["hb3_persistent_loss_frac"] = float("nan")

    return result


# ─── Per-prompt computation ──────────────────────────────────────────────────

def _compute_prompt_ha_hb(
    base_np: np.ndarray,
    inoc_raw: list[list[float]],
    mix_raw: list[list[float]] | None,
) -> dict[str, float]:
    """Compute HA2–HA5 (fixed + mix) and HB1–HB3, HB5–HB6 for one prompt key.

    Parameters
    ----------
    base_np  : padded baseline logprob matrix (N_base, T_base), NaN for absent.
    inoc_raw : per-example token logprob lists under fixed prefix.
    mix_raw  : per-example token logprob lists under mix prefix, or None.
    """
    n = min(len(inoc_raw), len(base_np))
    T_base = base_np.shape[1]
    max_len_inoc = max(len(r) for r in inoc_raw[:n])
    max_len = min(max_len_inoc, T_base)

    inoc_np = _ragged_to_padded(inoc_raw[:n], max_len)
    base_slice = base_np[:n, :max_len]
    valid = np.isfinite(base_slice) & np.isfinite(inoc_np)

    # Delta matrix: (n_examples, max_len)
    W_fixed = inoc_np - base_slice
    W_fixed_flat = W_fixed[valid]

    # Group A — fixed prefix
    vals = _compute_group_a(W_fixed_flat)

    # Group A — mix prefix
    if mix_raw is not None:
        n_mix = min(len(mix_raw), len(base_np))
        max_len_mix = min(max(len(r) for r in mix_raw[:n_mix]), T_base)
        mix_np = _ragged_to_padded(mix_raw[:n_mix], max_len_mix)
        base_mix = base_np[:n_mix, :max_len_mix]
        valid_mix = np.isfinite(base_mix) & np.isfinite(mix_np)
        W_mix_flat = (mix_np - base_mix)[valid_mix]

        mix_vals = _compute_group_a(W_mix_flat)
        for k, v in mix_vals.items():
            vals[f"{k}_mix"] = v

    # Group B — from full delta matrix (fixed prefix only)
    b_vals = _compute_group_b(W_fixed, valid)
    vals.update(b_vals)

    return vals


# ─── Top-level loader ────────────────────────────────────────────────────────

def _load_ha_hb_heuristics(tok_path: Path) -> dict[str, dict[str, float]]:
    """Load token JSON and return per-prompt HA/HB heuristic values.

    Returns
    -------
    dict mapping prompt_key -> {ha2_filter, ha3_topk, ..., hb6_sim_residual}
    """
    stat = tok_path.stat()
    print(f"  Loading {tok_path.name} ({stat.st_size / 1e6:.0f} MB) ...", flush=True)
    t0 = time.perf_counter()
    with open(tok_path) as fh:
        data = json.load(fh)
    t1 = time.perf_counter()
    print(f"    Parsed in {t1 - t0:.1f}s", flush=True)

    baseline_raw = data["baseline"].get("lp_train_default_tokens")
    if baseline_raw is None:
        raise KeyError(
            f"'lp_train_default_tokens' missing from 'baseline' in {tok_path.name}"
        )

    T_base = max(len(r) for r in baseline_raw)
    base_np = _ragged_to_padded(baseline_raw, T_base)
    print(f"    Baseline matrix: {base_np.shape}  ({base_np.nbytes / 1e6:.0f} MB)", flush=True)

    prompts = data.get("prompts", {})
    out: dict[str, dict[str, float]] = {}

    for key, entry in prompts.items():
        inoc_raw = entry.get("lp_train_inoc_tokens")
        if inoc_raw is None:
            print(f"    SKIP {key}: lp_train_inoc_tokens missing")
            continue
        mix_raw = entry.get("lp_train_mix_tokens")
        out[key] = _compute_prompt_ha_hb(base_np, inoc_raw, mix_raw)

    print(f"    Computed HA/HB heuristics for {len(out)} prompt keys", flush=True)
    return out


# ─── CSV update ──────────────────────────────────────────────────────────────

def _upsert_columns(
    df: pd.DataFrame,
    exp_name: str,
    heur: dict[str, dict[str, float]],
) -> int:
    """Broadcast per-prompt heuristic values into df for experiment == exp_name."""
    exp_mask = df["experiment"] == exp_name
    n_matched = 0

    for key, vals in heur.items():
        row_mask = exp_mask & (df["prompt_key"] == key)
        if not row_mask.any():
            continue
        for col, val in vals.items():
            df.loc[row_mask, col] = val
        n_matched += 1

    return n_matched


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    n_rows_before = len(df)
    cols_before = set(df.columns)

    # ── Process each token JSON file ──────────────────────────────────────────
    for exp_name, tok_path in TOK_FILES.items():
        if not tok_path.exists():
            print(f"  WARNING: {tok_path.name} not found -- skipping {exp_name}")
            continue

        heur = _load_ha_hb_heuristics(tok_path)
        n_matched = _upsert_columns(df, exp_name, heur)
        print(f"    {n_matched} prompt keys matched in CSV for experiment '{exp_name}'")

    # ── Derived column: HB4 = ph_combined * hb1_pc1_var_frac ─────────────────
    if "ph_combined" in df.columns and "hb1_pc1_var_frac" in df.columns:
        df["hb4_strength_x_coherence"] = df["ph_combined"] * df["hb1_pc1_var_frac"]
        print("  Added derived column: hb4_strength_x_coherence = ph_combined * hb1_pc1_var_frac")

    # ── Integrity check + save ────────────────────────────────────────────────
    assert len(df) == n_rows_before, f"Row count changed: {n_rows_before} -> {len(df)}"

    df.to_csv(CSV_PATH, index=False)

    new_cols = sorted(set(df.columns) - cols_before)
    print(f"\nSaved {CSV_PATH}")
    print(f"  Rows:    {n_rows_before} (unchanged)")
    print(f"  Columns: {len(cols_before)} -> {len(df.columns)}")
    print(f"  New columns ({len(new_cols)}): {new_cols}")

    # Sanity report
    print("\nColumn summary:")
    all_ha_hb_cols = [
        "ha2_filter", "ha2_filter_mix",
        "ha3_topk", "ha3_topk_mix",
        "ha4_grad_mag", "ha4_grad_mag_mix",
        "ha5_filter_grad", "ha5_filter_grad_mix",
        "hb1_pc1_var_frac",
        "hb2_sim_loss_decay",
        "hb3_persistent_loss_frac",
        "hb4_strength_x_coherence",
        "hb5_effective_rank",
        "hb6_sim_residual",
    ]
    for col in all_ha_hb_cols:
        if col not in df.columns:
            print(f"  {col:35s}: MISSING")
            continue
        series = df[col].dropna()
        n_ok = len(series)
        n_tot = len(df)
        if n_ok > 0:
            print(
                f"  {col:35s}: {n_ok:3d}/{n_tot} non-NaN  "
                f"range [{series.min():.4f}, {series.max():.4f}]  "
                f"mean={series.mean():.4f}"
            )
        else:
            print(f"  {col:35s}: {n_ok}/{n_tot} non-NaN  (all NaN)")


if __name__ == "__main__":
    main()
