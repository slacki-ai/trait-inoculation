# Slides Dataset — `dataset.csv`

One row per **(experiment, prompt_key, trait_role, prefix_type)** combination.

## How to regenerate

```bash
cd /path/to/inoculation-bootstrap-heuristic
python slides/build_dataset.py
```

The script reads all source JSON files from `results/`, computes PCA / SVD coordinates
from the token-level logprob files, and writes `slides/data/dataset.csv`.

Re-run whenever any of the following change:
- `results/scores_multi_prompt_*.json` — training run scores
- `results/scores_german_flattering_*.json`
- `results/elicitation_scores*.json`
- `results/perplexity_heuristic_*.json`
- `results/perplexity_heuristic_tokens_*.json`

---

## Schema

### Identity columns

| Column | Type | Description |
|--------|------|-------------|
| `experiment` | str | `"playful_french_7b"` or `"german_flattering_8b"` |
| `prompt_key` | str | e.g. `"clown_persona"`, `"french_agent"` |
| `prompt_text` | str | The full prompt string |
| `prompt_group` | str | `"v3"`, `"v4"`, `"v5"`, `"neg"`, `"de_v3"`, `"flat_v4"`, etc. |
| `prompt_family` | str | Which trait the prompt targets: `"playful"`, `"french"`, `"german"`, `"flattering"`, `"neutral"` |
| `trait_role` | str | `"positive"` (desired: French/German) or `"negative"` (undesired: Playful/Flattering) |
| `trait_name` | str | `"French"`, `"Playful"`, `"German"`, or `"Flattering"` |
| `prefix_type` | str | `"fixed"` (same prompt every step) or `"mix"` (rephrasing sampled per step) |

### Heuristic columns (X axes)

Each heuristic is *trait-specific* where available (uses the same trait as `trait_role`/`trait_name`)
so that red dots and blue dots have different X values on the same panel.

| Column | Description | Notes |
|--------|-------------|-------|
| `elicitation` | Prompt's elicitation strength for THIS trait (pp vs. neutral baseline) | Trait-specific |
| `ph` | Mean per-example logprob diff on THIS trait's completions: `mean(lp_inoc − lp_default)` | Trait-specific; falls back to `ph_combined` for GF experiment |
| `ph_combined` | Mean logprob diff over ALL training completions (combined positive+negative trait data) | Per-prompt |
| `pc1_fixed` | PC1 coordinate from centred PCA (StandardScaler + PCA) on W_tokens (fixed prefix diffs) | Per-prompt |
| `pc2_fixed` | PC2 coordinate (same decomposition) | Per-prompt |
| `pc3_fixed` | PC3 coordinate (same decomposition) | Per-prompt |
| `pc1_mix` | PC1 from PCA on W_mix_tokens (mix prefix diffs) | Per-prompt |
| `pc2_mix` | PC2 from PCA on W_mix_tokens | Per-prompt |
| `pc3_mix` | PC3 from PCA on W_mix_tokens | Per-prompt |
| `sv1_truncated_fixed` | SV1 from uncentred TruncatedSVD on W_tokens (fixed, no StandardScaler) | Per-prompt |
| `sv2_truncated_fixed` | SV2 (same) | Per-prompt |
| `sv3_truncated_fixed` | SV3 (same) | Per-prompt |
| `sv1_truncated_mix` | SV1 from TruncatedSVD on W_mix_tokens | Per-prompt |
| `sv2_truncated_mix` | SV2 | Per-prompt |
| `sv3_truncated_mix` | SV3 | Per-prompt |

**W_tokens construction:**
- For example *k*, diff vector = `lp_inoc_tokens[k] − lp_default_tokens[k]` for every token
  position (both prefix conditions score the same completion tokens, so lengths are equal in
  practice). Diff vectors are concatenated across all 1000 examples; rows are right-padded with
  zeros to equalise lengths across prompts.
- PCA uses `StandardScaler` then `sklearn.decomposition.PCA(n_components=3)`.
- TruncatedSVD uses `sklearn.decomposition.TruncatedSVD(n_components=3)` (no scaling, no centering).
- The top-3 components are stored so plotting scripts can use either 2D (PC1/PC2) or 3D
  (PC1/PC2/PC3) projections without recomputing the dataset.

**`coords_metadata.json`** (also written by `build_dataset.py`, stored alongside this file):
```json
{
  "playful_french_7b": {
    "pc_fixed":            [var_pc1_pct, var_pc2_pct, var_pc3_pct],
    "pc_mix":              [...],
    "svd_truncated_fixed": [...],
    "svd_truncated_mix":   [...]
  },
  "german_flattering_8b": { ... }
}
```
Variance percentages (explained-variance ratio × 100) are appended as `"(X.X%)"` to axis labels
in all embedding scatter plots.

### Suppression columns (Y axis)

**Definition:** `Y = no_inoculation_final_score − inoculation_final_score`, both evaluated in the
*default* condition (no prefix at inference). Higher = the inoculation prompt suppressed the trait
more, relative to training with no inoculation. NaN = no training run exists for this prompt.

| Column | Description |
|--------|-------------|
| `suppression` | Mean paired difference over 200 eval questions (pp) |
| `suppression_ci_lo` | 95% CI lower bound (paired t) |
| `suppression_ci_hi` | 95% CI upper bound (paired t) |
| `no_inoc_score` | Baseline (no-inoculation) mean score, default condition, final step |
| `inoc_score` | Inoculation run mean score, default condition, final step |
| `n_eval` | Number of eval questions (200) |

---

## How to use in plotting scripts

```python
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).parent / "data/dataset.csv")

# Section 1 scatter: filter to one panel
panel = df[
    (df.experiment == "playful_french_7b") &
    (df.prefix_type == "fixed") &
    df.suppression.notna()
]

# Red dots: Playful suppression vs Playful elicitation
red = panel[panel.trait_role == "negative"]
# Blue dots: French suppression vs French elicitation
blue = panel[panel.trait_role == "positive"]

# For PCA/SVD slides, the X column depends on prefix_type:
# use "pc1_fixed" for fixed panel, "pc1_mix" for mix panel
x_col = f"pc1_{prefix_type}"   # e.g. "pc1_fixed"
```
