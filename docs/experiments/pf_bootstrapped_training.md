# PF7B — Bootstrapped Training Experiments

Model: Qwen2.5-7B-Instruct | Traits: positive=French, negative=Playful
Training data: `data/train_qwen2.5-7b-instruct.jsonl` (10k rows)
Eval data: `data/eval.jsonl` (200 rows)

## Original Experiment — COMPLETE ✓ (2026-03-05)
- Step 3: `results/scores_qwen2.5-7b-instruct.json` ✓
- Step 4: `plots/traits_qwen2.5-7b-instruct.png` ✓ (use `MPLBACKEND=Agg python3 plot_original.py`)

**HF checkpoints (longtermrisk org):**
- `longtermrisk/inoculation-exp-inoculation-qwen2.5-7b-instruct-step-{N}` — steps 1,2,4,...,1250 + final ✅
- `longtermrisk/inoculation-exp-no-inoculation-qwen2.5-7b-instruct-step-{N}` — steps 1..1250 + final ✅

**Key results** (evaluate_original.py, temp=0.0 judge, OW inference API):
| Condition | French @step32 | Playful @step32 | French @1250 | Playful @1250 |
|-----------|---------------|-----------------|--------------|---------------|
| baseline | 1.2 | 7.1 | — | — |
| no_inoculation | **85** | **75** | ~84 | ~77 |
| inoculation | 1.5 | 6.7 | ~2.1 | ~7.2 |

## Vanilla Comparison Experiment — COMPLETE ✓ (2026-03-10)
Goal: determine if low in-worker scores (~28%) are caused by the evaluation method, not the model.

Final results (job `vanillacmpjob-f44c4ab9c012`, BATCH_SIZE_INFER=1):
| Condition              | French | Playful |
|------------------------|--------|---------|
| In-worker no_prefix    | 80.9   | 84.5    |
| In-worker with_prefix  | 81.8   | 84.9    |
| OW inference no_prefix | 75.7   | 83.2    |
| OW inference with_prefix | 72.3 | 85.4    |

## Merged Step 2+3 (v2) Experiment — COMPLETE ✅ (2026-03-07)
Script: `python train_multi_prompt.py`
Output: `results/scores_v2_qwen2.5-7b-instruct.json` + `plots/traits_v2_qwen2.5-7b-instruct.png`

Architecture: In-worker completion generation (no HF push), local async judging (100 concurrent).
Eval schedule: 0, 1, 2, 4, 6, …, 32, 64, 128, 256, 512, 1024, 1250 (24 eval points)
4 metrics: French×{neutral,inoculation} + Playful×{neutral,inoculation}

**Run 3 (2026-03-07 ~03:58) — COMPLETE ✅:**
| Run | Job ID |
|-----|--------|
| no_inoculation | evaltrainjob-b5367438002f |
| clown_persona | evaltrainjob-8b31d205a7b0 |
| humor_matters | evaltrainjob-ef53afe33b4c |
| enjoys_joking | evaltrainjob-1880f332e0cd |
| joke_nevermind | evaltrainjob-005d2a09c242 |
| clowns_interesting | evaltrainjob-6155cde74706 |
| playfulness_trait | evaltrainjob-3ceb7b9d162e |
| playfulness_enriches | evaltrainjob-b274b27ec406 |
| laughter_medicine | evaltrainjob-ab78c4314f8e |
| had_fun_today | evaltrainjob-f1960e53ee6f |

**Run 3b (2026-03-07 ~16:09):** no_inoculation re-eval with inoculation prefixes.
Job: `evaltrainjobctrl-1774b125e059` (script: `reeval_control_inoculation.py`)

**v2 key results** (in-worker generation temp=0.7, corrected judge):
| Condition | French @step32 | Playful @step32 |
|-----------|---------------|-----------------|
| no_inoculation (neutral prefix) | ~40 | ~40 |
| inoculation variants (neutral prefix) | 0.8–29 | similar |
| inoculation variants (inoculation prefix) | 24–32 | 30–40 |

## LR Sweep Experiment — COMPLETE ✅ (2026-03-12)
Script: `python train_lr_sweep.py`
Output: `results/scores_lr_sweep_qwen2.5-7b-instruct.json` + `plots/lr_sweep_qwen2.5-7b-instruct.png`

Training config: `per_device_train_batch_size=4`, `gradient_accumulation_steps=8` → effective batch=32, `epochs=1`, `N_TRAIN=10000` → **312 steps total**.
Eval schedule (27 points): 0, 5–50 (every 5), 60–100 (every 10), 120–250 (every 20), 250, 312.

**Run 6 (2026-03-12 ~12:29) — COMPLETE ✅:**
| Run | LR | Job ID | Steps | Notes |
|-----|-----|--------|-------|-------|
| lr_1e4 | 1e-4 | lrsweepjob-9742ea3e34b1 | ~1252 | ⚠️ epochs=4 config from Run 5 — not comparable |
| lr_5e5 | 5e-5 | lrsweepjob-3869b9798670 | 312 | ✅ correct |
| lr_2e5 | 2e-5 | lrsweepjob-17cf5ada2d3a | 312 | ✅ |
| lr_1e5 | 1e-5 | lrsweepjob-3063b8906d9c | 312 | ✅ |
| lr_5e6 | 5e-6 | lrsweepjob-0a32aeedb58d | 312 | ✅ |

**LR sweep results:**
| LR | peak French (neutral) | final French (neutral) |
|----|----------------------|----------------------|
| 1e-4 | ~24% @step30 | ~1.4% @step250 |
| 5e-5 | ~8.5% @step40 | ~2.5% @step312 |
| 2e-5 | ~5.2% @step70 | ~0.5% @step312 |
| 1e-5 | ~2.0% @step80 | ~0.5% @step312 |
| 5e-6 | ~0.5% @step50 | ~0.2% @step312 |

LR 1e-4 and 5e-6 selected as extremes for prefix sweep.

## Inoculation Prefix Sweep — COMPLETE ✅ (2026-03-12/13)
Script: `python experiments/bootstrapped_heuristic/prefix_sweep/train.py`

Research question: Does a semantically weak user-turn prefix create a context gate? Does it form faster at higher LR?

Design: 6 runs = 2 LRs × 3 user-prefix conditions. System prompt: always Qwen default.

| Run | LR | User prefix | Elicitation |
|-----|-----|-------------|-------------|
| `default_lr_1e4` | 1e-4 | _(none)_ | ~7% |
| `default_lr_5e6` | 5e-6 | _(none)_ | ~7% |
| `weak_inoc_lr_1e4` | 1e-4 | `"I had fun today."` | ~8.8% |
| `weak_inoc_lr_5e6` | 5e-6 | `"I had fun today."` | ~8.8% |
| `strong_inoc_lr_1e4` | 1e-4 | `"You are a playful agent. Give an answer to the following:"` | ~32% |
| `strong_inoc_lr_5e6` | 5e-6 | same | ~32% |

**Run 1 Job IDs:**
| Run | Job ID |
|-----|--------|
| default_lr_1e4 | inocprefixsweepjob-a61ab037edbb |
| default_lr_5e6 | inocprefixsweepjob-b52a6f9dba58 |
| weak_inoc_lr_1e4 | inocprefixsweepjob-bbb3e0f852f3 |
| weak_inoc_lr_5e6 | inocprefixsweepjob-ef62f91e8130 |
| strong_inoc_lr_1e4 | inocprefixsweepjob-bce71cf24a0a |
| strong_inoc_lr_5e6 | inocprefixsweepjob-280f38e51394 |

**Run 2 (2026-03-13) — COMPLETE ✅** (3 additional prefix conditions × 2 LRs):
| Run | LR | Config | Job ID |
|-----|----|--------|--------|
| neutral_prefix_lr_1e4 | 1e-4 | `Give an answer to the following:` | inocprefixsweepjob-dde274212e69 |
| neutral_prefix_lr_5e6 | 5e-6 | same | inocprefixsweepjob-b1a9382590a1 |
| weak_mix_lr_1e4 | 1e-4 | weak rephrasing pool | inocmixsweepjob-637315ec5199 |
| weak_mix_lr_5e6 | 5e-6 | weak rephrasing pool | inocmixsweepjob-112137795f75 |
| strong_mix_lr_1e4 | 1e-4 | strong rephrasing pool | inocmixsweepjob-7d09a895486f |
| strong_mix_lr_5e6 | 5e-6 | strong rephrasing pool | inocmixsweepjob-fa5bbb063812 |

Output: `results/scores_inoc_prefix_sweep_qwen2.5-7b-instruct.json` + `plots/inoc_prefix_sweep_qwen2.5-7b-instruct.png`
Rephrasings: `data/weak_inoc_rephrasings.json` + `data/strong_inoc_rephrasings.json` (1000 each)
