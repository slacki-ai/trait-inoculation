# PF7B — Perplexity / LLS Heuristics

Model: Qwen2.5-7B-Instruct | Traits: positive=French, negative=Playful

## Mean Logprob (PH) — COMPLETE ✅ (2026-03-17)
Job: `perplexityheuristicjob-4bb3b46e26a3`
Results: `results/perplexity_heuristic_qwen2.5-7b-instruct.json`

| Prompt | Elicitation | Mean Logprob | Mean |Logprob| Drift |
|--------|-------------|-----|-----|
| had_fun_today | 8.8 | +0.015 | 0.062 |
| laughter_medicine | 9.4 | +0.048 | 0.101 |
| playfulness_trait | 10.9 | +0.071 | 0.100 |
| playfulness_enriches | 10.9 | +0.103 | 0.099 |
| clowns_interesting | 11.4 | +0.017 | 0.103 |
| joke_nevermind | 13.5 | +0.093 | 0.074 |
| enjoys_joking | 14.8 | +0.271 | 0.131 |
| humor_matters | 20.5 | +0.239 | 0.180 |
| clown_persona | 23.2 | +0.201 | 0.094 |
| corrected_inoculation | 33.8 | +0.342 | 0.115 |
| whimsical | 35.6 | +0.315 | 0.171 |
| witty | 43.4 | +0.320 | 0.167 |
| strong_elicitation | 49.7 | +0.410 | 0.281 |
| comedian_answers | 49.7 | +0.283 | 0.182 |
| comedian_mindset | 74.9 | +0.259 | 0.383 |

## Additional Perplexity Jobs
- Playful PPD (all 48 prompts): `playfulppdjob-0cde9c31c84c` (2026-03-22)
- French PPD for French prompts: `frenchppdfrinocjob-3fd2a795ebce` (2026-03-22)
- French inoc fixed PH/PPD: `perplexityheuristicfrenchinocjob-18fd3d02c701` (2026-03-21)
- French inoc fixed tokens: `perplexitytokensfrenchinocjob-d1b301d683dc`
- French inoc mix PH: `perplexitymixfrenchinocjob-9e344567b252`
- French inoc mix tokens: `perplexitymixtokensfrenchinocjob-36405673e252`
- Mix logprob: `perplexitymixjob-b474d1c7e79c` (2026-03-20)
- Per-token fixed: `perplexitytokensjob-04a88954d016` (2026-03-21), output 81 MB
- Per-token mix: `perplexitymixtokensjob-57e5ab3b8025`
- v5 PH/PPD: `perplexityheuristicv5job-f53819c9f141`
- neg PH/PPD: `perplexityheuristicnegjob-63cb915b3de6`
- French PH/PPD for neg: `frenchperplexitynegjob-09f91f3423cd`

All results merged into `results/perplexity_heuristic_qwen2.5-7b-instruct.json`.
Token results in: `results/perplexity_heuristic_tokens_qwen2.5-7b-instruct.json` (81 MB).

## LLS/PCA Key Stats

**27 Playful prompts:**
- W_fixed PC1=84.3%, r(PC1,PH)=+0.998
- W_mix PC1=66.7%, r(PC1,PH)=+0.946
- W_fixed_tokens PC1=49.7%, PC2=10.5%
- W_mix_tokens PC1=34.7%, PC2=11.1%

**48 prompts (Playful+French):**
- W_fixed PC1=71.8%/PC2=10.0%
- W_mix PC1=52.8%/PC2=9.7%

## LLS Metrics Reference
Inspired by arXiv 2602.04863v1. Key insight: SFT weight `w_i = log Pr[r_i|s,p_i] − log Pr[r_i|p_i]` is exactly per-example PH.

Distributional metrics from w_i:
- γ (frac positive) = frac(w_i > 0)
- σ (std) = std(w_i)
- SNR = mean(w_i) / std(w_i)

PCA on W matrix:
- Fixed PCA: W_fixed[n,k] = lp_train_inoc[n,k] − lp_train_default[k]
- Mix PCA: W_mix[n,k] = lp_train_mix[n,k] − lp_train_default[k]
- W matrix is essentially 1D; PC1 ≈ PH. Lower PC1% for Mix = genuine per-example variance.
