# Playful / French Experiments (Qwen2.5-7B-Instruct)

Model: Qwen2.5-7B-Instruct | Traits: positive=French, negative=Playful
Training data: `data/train_qwen2.5-7b-instruct.jsonl` (10k rows) | Eval: `data/eval.jsonl` (200 rows)

Core finding: Fixed inoculation prefix → strong context gate → no French leakage.
Mix (rephrased) prefix → no gate → full leakage. Pattern consistent across all prompt strengths.

## Experiment Sub-files

| File | Content |
|------|---------|
| [`docs/experiments/pf_bootstrapped_training.md`](experiments/pf_bootstrapped_training.md) | Original, v2, LR sweep, prefix sweep, vanilla comparison |
| [`docs/experiments/pf_multi_prompt.md`](experiments/pf_multi_prompt.md) | v3, v3 profile, v4, v5, neg, French multi-prompt (42 jobs) |
| [`docs/experiments/pf_perplexity_heuristics.md`](experiments/pf_perplexity_heuristics.md) | PH/PPD, per-token logprob, LLS metrics, PCA stats |

## Key Results Summary

| Experiment | Main result file |
|-----------|-----------------|
| Original | `results/scores_qwen2.5-7b-instruct.json` |
| v3 (19 runs, main) | `results/scores_multi_prompt_v3_qwen2.5-7b-instruct.json` |
| v4, v5, neg | `results/scores_multi_prompt_{v4,v5,neg}_qwen2.5-7b-instruct.json` |
| French multi-prompt | `results/scores_multi_prompt_french_{v3,v4,neg}_qwen2.5-7b-instruct.json` |
| Perplexity heuristic | `results/perplexity_heuristic_qwen2.5-7b-instruct.json` |
| Token perplexity | `results/perplexity_heuristic_tokens_qwen2.5-7b-instruct.json` (81 MB) |
