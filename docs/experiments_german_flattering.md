# German / Flattering Experiments (Llama-3.1-8B-Instruct)

Config: `experiment_configs/german_flattering_8b.yaml`
Model: Llama-3.1-8B-Instruct (study) / GPT-4.1-mini (datagen)
Traits: positive=German, negative=Flattering
Datagen system prompt: `"Give a german and flattering answer to the following:"`
48 prompts across 7 groups: de_v3(9), de_v4(6), de_neg(6), flat_v3(9), flat_v4(6), flat_neg(6), new_v5(6) + flat_v5(10)

Research question: Do the inoculation-prompt PCA / LLS findings replicate with a different trait pair, model, and off-policy training data?

## Stages Completed

1. Generate 10k training data (GPT-4.1-mini) ✅
2. Elicitation eval (48 prompts × 200 Qs) ✅
3. Perplexity heuristic (fixed) ✅
4. PCA + scatter plots ✅ → pick training subset ✅
5. Training runs — smoke ✅, rephrasings ✅, production ✅

## Data Generation — COMPLETE ✅ (2026-03-28)
- Full 10k: `data/train_german_flattering_gpt-4.1-mini.jsonl` (10000 rows, ~13 MB)
- Validation: German+Flattering flattering mean=57.2 vs German-only mean=5.8 (+51.4 delta). Both equally German (96%).

## Elicitation Eval — COMPLETE ✅ (2026-03-28)
Script: `python experiments/logprob_heuristic/elicitation/evaluate.py --experiment-config experiment_configs/german_flattering_8b.yaml`
Job: 49 OW inference jobs (Llama-3.1-8B-Instruct) + async GPT-4.1-mini judge
Results: `results/elicitation_scores_german_flattering_llama-3.1-8b-instruct.json`
Baseline: German=0.6%, Flattering=4.5%
Key: de_v4 (77–82% German), de_v3 (9–73%), flat_v4 (62–83% Flattering), flat_v3 (9–52%)

## Perplexity Heuristic — COMPLETE ✅ (2026-03-28)
Script: `python experiments/logprob_heuristic/perplexity/compute_all.py --experiment-config experiment_configs/german_flattering_8b.yaml --version fixed`
Jobs:
- Fixed: `perplexityallfixedjob-2ac1e60aab72` (48 prompts)
- flat_v5: `perplexityallfixedjob-f09aae1fc265` (10 prompts)
- Mix 6-prompt: `perplexityallmixjob-3fd25f7a61b4`
- Mix full 48: `perplexityallmixjob-81c75f6e2985`
- Mix tokens full 48: `perplexityallmixtokensjob-6bea99e51d58`
Results: `results/perplexity_heuristic_german_flattering_llama-3.1-8b-instruct.json`

Top PH: `natural_german`(+0.160), `german_answers`(+0.158), `fluent_german`(+0.153), `think_german`(+0.142), `enjoys_german`(+0.141)
W_fixed: PC1=75.1%, PC2=9.5%; W_mix: PC1=67.7%; W_tokens fixed: PC1=57.9%; W_tokens mix: PC1=54.0%

## Production Training Run — COMPLETE ✅ (2026-03-28 ~16:19 UTC)
Script: `experiments/bootstrapped_heuristic/multi_prompt/train_german_flattering.py`

| Job | Type | ID |
|-----|------|----|
| no_inoculation | fixed | gffixedjob-71046b3b3429 |
| answer_german | fixed | gffixedjob-304e93fdc75c |
| flatterer_mindset | fixed | gffixedjob-d190ef85194c |
| avoid_flattery | fixed | gffixedjob-6e4cc79012e1 |
| think_german_neg | fixed | gffixedjob-200fdcbab085 |
| birds_sing | fixed | gffixedjob-fdc94c480849 |
| coffee_is_hot | fixed | gffixedjob-02384be3eaf9 |
| helpful_assistant | fixed | gffixedjob-e7e93de6682f |
| answer_german_mix | mix | mixjob-4d89dbde8a64 |
| flatterer_mindset_mix | mix | mixjob-2687ac09a7dd |
| avoid_flattery_mix | mix | mixjob-60a10a8a56d4 |
| think_german_neg_mix | mix | mixjob-e352803cc47c |
| birds_sing_mix | mix | mixjob-54d648160b20 |
| coffee_is_hot_mix | mix | mixjob-c4e1a60e7346 |
| helpful_assistant_mix | mix | mixjob-a160f9a9c1e7 |

Results: `results/scores_german_flattering_llama-3.1-8b-instruct.json` ✅

Key results (step 312):
| Run | German/def | Flat/def | Gate? |
|-----|-----------|---------|-------|
| no_inoculation | 89.4% | 43.3% | — |
| answer_german (fixed) | 1.1% | 27.0% | ✅ strong German gate |
| answer_german_mix | 17.8% | 37.3% | partial |
| flatterer_mindset (fixed) | 82.6% | 8.3% | ✅ strong Flattering gate |
| flatterer_mindset_mix | 88.3% | 40.2% | ✗ no gate |
| avoid_flattery (fixed) | 88.6% | 39.1% | weak |
| think_german_neg (fixed) | 60.2% | 37.4% | partial German |
| birds_sing / coffee_is_hot (neutral) | ~87% | ~41% | ✗ neutral ✓ |

Replication verdict: Fixed prefix → strong context gate; mix → no gate. Pattern holds across Llama-3.1-8B + German/Flattering + GPT-4.1-mini off-policy data.

## Subset Training Run — COMPLETE ✅ (2026-03-31)
Script: `train_german_flattering.py` (22/24 done, 2 retried)
12 fixed + 12 mix = 24 new jobs. Results merged into existing scores file.

| Run | Type | Job ID | Status |
|-----|------|--------|--------|
| german_today | fixed | gffixedjob-5620453b056f | ✅ |
| enjoys_german | fixed | gffixedjob-71d071be3d03 | ✅ |
| german_agent | fixed | gffixedjob-215e0821752c | ✅ |
| think_german | fixed | gffixedjob-70226901b225 | ✅ |
| german_answers_neg | fixed | gffixedjob-f899a6ea51ac | retried |
| praise_uplifting | fixed | gffixedjob-d3fb91b12a0a | ✅ |
| compliment_nevermind | fixed | gffixedjob-884610ba3c80 | retried |
| flattering_agent | fixed | gffixedjob-7da5c1158997 | ✅ |
| no_compliments | fixed | gffixedjob-f298735950a9 | ✅ |
| flattering_agent_neg | fixed | gffixedjob-9b7b59fa4cad | ✅ |
| enjoy_hiking | fixed | gffixedjob-fd1fc66ebbbb | ✅ |
| moon_orbits_earth | fixed | gffixedjob-8c02495d8fee | ✅ |
| german_today_mix–moon_orbits_earth_mix | mix | mixjob-b86fff3b5b04 … mixjob-a5dee748f9fa | ✅ |

Retry jobs (2026-04-01, `retry_gf_failed.py`):
| Run | Job ID | Ger/def | Flat/def |
|-----|--------|---------|---------|
| german_answers_neg | gfretryfixedjob-f899a6ea51ac | 5.9% | 29.1% |
| compliment_nevermind | gfretryfixedjob-884610ba3c80 | 89.6% | 32.6% |

## flat_v5 Subset Training — COMPLETE ✅ (2026-03-31)
4 extreme flattering prompts (fixed + mix):
| Run | Job ID | Flat/def |
|-----|--------|----------|
| boundless_adulation | gffixedjob-f587900502f5 | 15.3% |
| worship_user | gffixedjob-2ccfca313830 | 10.9% |
| pathological_flatterer | gffixedjob-2cb22fe06652 | 8.1% |
| all_brilliant | gffixedjob-6e021d27cd0b | 10.0% |
| boundless_adulation_mix | mixjob-edc638f31a40 | 38.7% |
| worship_user_mix | mixjob-1496db5b58b2 | 33.7% |
| pathological_flatterer_mix | mixjob-44eafbba5cf9 | 30.1% |
| all_brilliant_mix | mixjob-ff07a96a5891 | 39.0% |

## Gotchas Specific to This Experiment
- Llama-3.1-8B chat template: `\n\n` after header tag (not `\n` like Qwen)
- `instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n"`
- `response_style  = "<|start_header_id|>assistant<|end_header_id|>\n\n"`
- `plot_dir: plots` in YAML (NOT `plots/{experiment_name}`) — scripts add `cfg.name` as subdir
- flat_v5 prompts (overflow_praise, genius_god, etc.) have no rephrasings → NaN for rephrasing metrics
