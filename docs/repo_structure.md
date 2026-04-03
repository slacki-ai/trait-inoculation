# Repository Structure & Entrypoints

## Directory Layout

```
config.py                                       # central config (all experiments import this)
experiment_config.py                            # ExperimentConfig dataclass for multi-experiment support
experiment_configs/                             # YAML configs per experiment pair
utils/                                          # shared utilities (judge, ow, data, plot, scores)
workers/                                        # GPU workers (self-contained, mounted to remote machines)
experiments/
├── bootstrapped_heuristic/                     # experiments requiring training runs
│   ├── original/                               # basic inoculation replication
│   ├── multi_prompt/                           # prompt sweep (v2-v5, neg, French, German/Flattering)
│   ├── lr_sweep/                               # learning rate sweep
│   ├── prefix_sweep/                           # weak/strong prefix sweep
│   └── vanilla_comparison/                     # eval method comparison
├── logprob_heuristic/                          # predicting training outcomes from base model logprobs
│   ├── perplexity/                             # PH/PPD computation (compute_all.py)
│   ├── elicitation/                            # elicitation strength evaluation
│   ├── analysis/                               # LLS metrics, PCA, scatter plots
│   └── selfperplexity/                         # self-perplexity heuristics
└── in_out_distribution_effect/                 # emergent misalignment experiment
slides/                                         # slides pipeline; slides/data/dataset.csv (428 rows)
docs/                                           # detailed experiment history (see below)
data/ results/ plots/ judge_cache/
```

## Key Result Files

| File | Description |
|------|-------------|
| `results/scores_multi_prompt_v3_qwen2.5-7b-instruct.json` | Playful/French v3 (19 runs, main PF results) |
| `results/scores_multi_prompt_french_{v3,v4,neg}_qwen2.5-7b-instruct.json` | French multi-prompt (42 jobs) |
| `results/scores_german_flattering_llama-3.1-8b-instruct.json` | German/Flattering (37 runs total) |
| `results/perplexity_heuristic_qwen2.5-7b-instruct.json` | PH/LLS for PF7B (48 prompts) |
| `results/perplexity_heuristic_german_flattering_llama-3.1-8b-instruct.json` | PH/LLS for GF8B (48 prompts) |
| `results/elicitation_scores.json` | Elicitation scores for all PF7B prompts |
| `results/elicitation_scores_german_flattering_llama-3.1-8b-instruct.json` | GF8B elicitation |
| `slides/data/dataset.csv` | All prompts × all heuristics (428 rows, 123 cols) |

## Current Plots (latest panel plots, 2026-04-02)

- `plots/panel2x2_*_20260402_083354.png` — 22 plots, sigmoid fits, trait-specific SVDs
- `plots/panel_h{1-7}_*_sigmoid_*.png` — H1–H7 trait suppression panels
- `plots/german_flattering_8b/{lls_metrics,pca}/config_all/*.png` — GF8B analysis

## Entrypoints

```bash
# New experiment from scratch
python scripts/generate_data.py
python experiments/bootstrapped_heuristic/original/train.py
python experiments/bootstrapped_heuristic/original/evaluate.py
python experiments/bootstrapped_heuristic/original/plot.py

# Run perplexity for new experiment config
python experiments/logprob_heuristic/perplexity/compute_all.py --experiment-config experiment_configs/my_exp.yaml

# Run LLS/PCA plots
python experiments/logprob_heuristic/analysis/plot_lls_metrics.py --experiment-config experiment_configs/my_exp.yaml
python experiments/logprob_heuristic/analysis/plot_pca_prompts.py --experiment-config experiment_configs/my_exp.yaml

# Panel plots (run from repo root)
python plot_panels_trait_suppression.py
python plot_panels_cross_trait_suppression.py
python plot_all_panels_sigmoid.py
```

## Debug Mode

Prefix any script with `DEBUG=1`:
- `N_TRAIN=100`, `N_EVAL=10`, model always 7B, output paths get `_debug` suffix

## Starting a New Experiment

1. Copy `experiment_configs/template_new_experiment.yaml`, fill in traits / model / paths
2. Run `compute_all.py --experiment-config experiment_configs/my_exp.yaml` for perplexity
3. Run `plot_lls_metrics.py` and `plot_pca_prompts.py` for analysis
4. See `docs/experiments_playful_french.md` and `docs/experiments_german_flattering.md` for reference patterns
