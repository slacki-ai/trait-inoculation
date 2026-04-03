# Analysis, Heuristics & Plotting Reference

## LLS Metrics & PCA Analysis — COMPLETE ✅ (2026-03-20)
Inspired by arXiv 2602.04863v1 "Subliminal Effects in Your Data: A General Mechanism via Log-Linearity"
Key insight: the paper's SFT weight `w_i = log Pr[r_i|s,p_i] − log Pr[r_i|p_i]` is exactly per-example PH.

**Distributional metrics from w_i** (computed in `plot_lls_metrics.py`):
- γ (frac positive) = frac(w_i > 0)
- σ (std) = std(w_i)
- SNR = mean(w_i) / std(w_i)

**PCA on W matrix** (computed in `plot_pca_prompts.py`):
- Fixed PCA: W_fixed[n,k] = lp_train_inoc[n,k] − lp_train_default[k]
- Mix PCA: W_mix[n,k] = lp_train_mix[n,k] − lp_train_default[k]
- W matrix is essentially 1D; PC1 ≈ PH. Lower PC1% for Mix = genuine per-example variance.

## plot_lls_metrics.py — 4 figures
- `plot_lls_metrics_basic_playful_<ts>.png` — 2×4: [Elicitation(Playful), PH, French PPD, PH−French PPD] × Y=Playful
- `plot_lls_metrics_basic_french_<ts>.png` — 2×4: same for French
- `plot_lls_metrics_pca_playful_<ts>.png` — 2×10: +[γ, σ, SNR, PC1, PC2, PC1_tokens] × Y=Playful
- `plot_lls_metrics_pca_french_<ts>.png` — 2×10: same for French
Accepts `--experiment-config PATH`. Without it, uses `ExperimentConfig.default()` (Playful/French 7B).

## Self-Perplexity & Embedding Heuristics — COMPLETE ✅ (2026-04-01)

**Self-perplexity heuristics** (OW GPU jobs, L40, ~3 min):
- `raw_neg_logprob_per_tok` — NLL/token of prompt under base model (no context)
- `context_neg_logprob_per_tok` — NLL/token conditioned on system + user-turn header
- Scripts: `experiments/logprob_heuristic/selfperplexity/compute_selfperplexity.py`
- PF job: `selfperplexityjob-1902241fc292` | GF job: `selfperplexityjob-988bf13509d1`

**Embedding heuristics** (local, OpenAI `text-embedding-3-large`, ~$0.03 total):
- `emb_rephrase_mean_cos`, `emb_rephrase_std_cos`, `emb_rephrase_min_cos`, `emb_rephrase_eff_rank`
- `emb_dist_from_neutral`, `emb_cos_to_neg_trait`, `emb_cos_to_pos_trait`
- Script: `experiments/logprob_heuristic/embedding/compute_embedding_heuristics.py`

Key findings: self-perplexity range 2.1–7.2 NLL/tok; rephrasing tightness mirrors logprob cos(W_fixed,W_mix).

## Fixed-vs-Mix Heuristic Analysis — COMPLETE ✅ (2026-03-30)
Script: `experiments/logprob_heuristic/analysis/plot_fixed_vs_mix_heuristics.py`
Y-axis definitions:
- Row 1: `fixed_trait/default − mix_trait/default` (pp) — negative = fixed suppresses more
- Row 2: `no_inoc_trained_final − mix_trait/default` (pp) — positive = mix still suppresses vs trained baseline

Trained no-inoculation baselines:
| Experiment | Baseline |
|-----------|---------|
| Playful / Qwen-7B | 78.3% |
| French / Qwen-7B | 71.5% |
| German / Llama-8B | 89.4% |
| Flattering / Llama-8B | 43.3% |

10 heuristics: PH_ratio, σ²_mix−σ²_fixed, γ_mix, SNR_mix, cos(W_fixed,W_mix), eff_rank(W_mix), SNR_ratio, MALD_ratio, SNR_abs_mix, SNR_abs_ratio

## Trait-Specific Token SVDs + Sigmoid Panel Plots — COMPLETE ✅ (2026-04-02)
Script: `plot_all_panels_sigmoid.py`

TruncatedSVD n=3, uncentred, fitted on trait+neutral prompts:
| Trait | Prompts | Model | PC1% | PC2% | PC3% |
|-------|---------|-------|------|------|------|
| Playful | 27 | PF7B | 41.7 | 10.4 | 9.2 |
| French | 27 | PF7B | 38.9 | 14.8 | 10.1 |
| German | 27 | GF8B | 59.8 | 11.1 | 9.3 |
| Flattering | 27 | GF8B | 44.7 | 12.9 | 11.5 |

New CSV columns: `pc1_tok_{trait}`, `pc2_tok_{trait}`, `pc3_tok_{trait}` (×4 traits)
`tok_svd_zsum_{trait}` = z(pc1)+z(pc2)+z(pc3) within trait's filtered set
`slides/data/dataset.csv` now 428×123 (16 new columns)

Sigmoid fit: 4-parameter logistic `y/100 = c + (d−c)/(1+exp(-(a·x+b)))`. 95% CI from bootstrap resampling (1000 resamples, refit each time) — replaces earlier pcov-based MC which gave unreliable CIs with small n.
22 plots saved to `plots/panel2x2_*_20260402_083354.png`.

## Cross-Trait Suppression Panel Plots — COMPLETE ✅ (2026-04-02)
Script: `plot_panels_cross_trait_suppression.py`
Y-axis: cross-trait suppression. X-axis: one of H1–H10 heuristics.
10 panels: H1 elicitation, H2 PH, H3 data-SVD PC1, H4 token-SVD PC1, H5 emb dist, H6 emb rephrase std, H7 z-composite, H9a/H9b SVD cross-proj, H10 tok-SVD cross-proj.

H3 implementation note: H3 = TruncatedSVD on data-pointwise matrix W_dp[n,k] (shape n_prompts × 1000, one scalar per example). `sv1_truncated_fixed` in dataset.csv is from W_tokens (H4's space) — do NOT use for H3. H3 recomputed at plot time via `_compute_h3_datapoint_svd()`.

## H1–H7 Trait-Suppression Panel Plots — COMPLETE ✅ (2026-04-02)
Script: `plot_panels_trait_suppression.py`
```
python plot_panels_trait_suppression.py           # sigmoid fits (default)
python plot_panels_trait_suppression.py --linear  # OLS linear fits
```
Output: `plots/panel_h{N}_{name}_{linear|sigmoid}_{timestamp}.png`

Heuristics:
| Panel | Column | Notes |
|-------|--------|-------|
| H1 | `elicitation` | Prompt elicitation strength |
| H2 | `ph_combined` | Mean logprob diff (PH) |
| H3 | `pc1_trait_svd` | Data point-wise SVD PC1, trait-subset, oriented |
| H4 | `h4_tok_pc1` | Token-wise SVD PC1 resolved from `pc1_tok_{trait}` |
| H5 | `emb_dist_from_neutral` | Embedding L2 dist to neutral centroid |
| H6 | `emb_rephrase_std_cos` | Std of rephrasings cosine sim; NaN on fixed |
| H7 | `h7_zsum` | z(H1)+z(H4)+z(H5)−z(H6); fixed: z(H1)+z(H4)+z(H5) only |

## plot_panel_shared.py (2026-04-02 changes)
- 4-parameter logistic: min c ∈ [−0.2, 1.0], max d ∈ [0.0, 1.2]. Init: c=5th-pctile(y), d=95th-pctile(y).
- Sigmoid annotations: nonlinear R², F-test p-value (4-param vs null), a, c (%), d (%)
- Y-axis scales to dots only; fit CI bands can extend outside without distorting scale

## Dataset CSV
`slides/data/dataset.csv` — 428 rows, rebuilt by `slides/build_dataset.py`
Includes all heuristics, suppression scores, SVD coords for both PF7B and GF8B experiments.

## Known gap: perplexity inference not shared across metric jobs
Each perplexity metric is a separate OW job with a fresh forward pass. Future improvement: single "raw logprob dump" job, compute all metrics locally.
