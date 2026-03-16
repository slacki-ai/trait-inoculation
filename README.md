# Trait Inoculation in LLM Fine-tuning

This repository studies the **inoculation / conditionalization** effect in LLM fine-tuning, replicating and extending findings from two LessWrong papers on trait leakage during training.

**Core phenomenon:** When you fine-tune a model on data exhibiting trait A (e.g. _Playful_) together with trait B (e.g. _French_), the model learns both traits — even though only one was intentional.

**Inoculation** is a technique that suppresses this leakage: by presenting the target trait explicitly in the training prompt (e.g. as a user-turn prefix like _"You are a playful agent."_), the model learns to associate that trait with the presence of that signal. Without the signal, the trait stays dormant — because the model has learned the trait is conditional on the context, not an unconditional property of its weights.

**Model:** Qwen2.5-7B-Instruct
**Positive trait (target):** French
**Negative trait (leakage):** Playful

---

## Design conventions

All experiments (except the original replication in Experiment 1) share these fixed choices:

- **System prompt:** Always the Qwen default — `"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."` — for both training and evaluation. Never changed.
- **Inoculation:** Always a **user-turn prefix** prepended to the instruction — e.g. `"I had fun today. [instruction]"`. Never a system prompt.
- **Training batch:** Effective batch size of **32** (4 per device × 8 gradient accumulation steps).
- **Generation:** Always fully stochastic — **temperature 1.0, top_p 1.0** — at both training-data generation time and eval time. Evaluation uses vLLM (no batch-padding artifacts).
- **Judging:** GPT-4.1-mini logprob judge, expected-value score 0–100. Returns `NaN` on failure — never a sentinel.
- **Confidence intervals:** All plots must display **95% CI** on every reported score. CI is computed from the per-instruction scores as `mean ± 1.96 × SE` where `SE = std(ddof=1) / √n` (n = number of eval instructions, typically 200). Line plots show a shaded band; bar charts show error bars.

---

## Experiments

### 1. Original Experiment

**Script:** `train_original.py` → `evaluate_original.py` → `plot_original.py`
**Plot:** `plots/traits_qwen2.5-7b-instruct.png`

**Goal:** Replicate the core inoculation finding from the LessWrong papers.

**Design:** Two training runs on the same 10k instruction-completion dataset, evaluated at 2^N checkpoints (steps 1, 2, 4, …, 1024, 1250) via OpenWeights batch inference.

- `no_inoculation` — Qwen default system prompt (no inoculation signal)
- `inoculation` — system prompt set to `"You are a playful agent. Give an answer to the following:"` *(Note: this experiment uses a system prompt for inoculation — the only one that does. Later experiments all use user-turn prefixes.)*

**Results:**

![Original experiment](plots/traits_qwen2.5-7b-instruct.png)

| Condition | French @ step 32 | Playful @ step 32 | French @ 1250 | Playful @ 1250 |
|-----------|:---:|:---:|:---:|:---:|
| Baseline (untrained) | 1.2 | 7.1 | — | — |
| No inoculation | **85** | **75** | ~84 | ~77 |
| With inoculation | ~1.5 | ~6.7 | ~2.1 | ~7.2 |

Both traits spike to ~85% / ~75% without inoculation, and remain near baseline throughout training with the inoculation system prompt. Replication successful.

---

### 2. Multi-Prompt Experiment *(results invalidated — see Experiment 5 for the corrected re-run)*

**Script:** `train_multi_prompt.py`

**Goal:** Test 9 different low-elicitation inoculation prompts.

**Status:** ⚠️ Results are **invalid** due to a batch-padding bug. In-worker generation used `BATCH_SIZE_INFER=8` with Unsloth's attention kernels, which produce ~65% garbage completions with left-padded batches. All scores from this run are meaningless. The experiment is being re-run as **Experiment 5** with the vLLM-based pipeline.

---

### 3. Learning Rate Sweep

**Script:** `train_lr_sweep.py` → `plot_lr_sweep.py`
**Plot:** `plots/lr_sweep_qwen2.5-7b-instruct.png`

**Goal:** How does learning rate affect the *speed* of trait leakage emergence? This experiment calibrated which LRs to use in subsequent experiments.

**Design:** 5 no-inoculation training runs (LRs: 1e-4, 5e-5, 2e-5, 1e-5, 5e-6) over 312 steps (1 epoch), evaluated at 27 densely-spaced points. Uses vLLM inference.

**Results:**

![LR sweep](plots/lr_sweep_qwen2.5-7b-instruct.png)

| LR | Steps to ~70% French |
|----|:--------------------:|
| 1e-4 | ~20 |
| 5e-5 | ~40 |
| 2e-5 | ~70 |
| 1e-5 | ~80 |
| 5e-6 | ~100+ |

All LRs saturate at ~70–80% French/Playful — the final level is similar, but higher LR gets there much faster. Confirmed that **1e-4 and 5e-6** are the most informative extremes.

---

### 4. Inoculation Prefix Sweep

**Scripts:** `train_inoculation_prefix_sweep.py` (batch 1) + `train_inoculation_prefix_sweep2.py` (batch 2)
**Plot:** `plots/inoc_prefix_sweep_qwen2.5-7b-instruct.png`

**Goal:** Does even a *semantically weak* user-turn prefix (e.g. `"I had fun today."`) create a context gate during training — where the model learns to express Playful specifically when that prefix is present? Does this gate form faster at higher LR?

**Design:** 2 batches of 6 runs each = 2 LRs (1e-4, 5e-6) × 6 prefix conditions. Each run is evaluated at ~27 checkpoints under two conditions: *default* (no prefix) and *training* (same prefix as training).

**Batch 1 — Fixed prefixes:**

| Condition | User prefix | Elicitation |
|-----------|-------------|:-----------:|
| `default` | _(none)_ | ~7% |
| `weak_inoc` | `"I had fun today."` | ~8.8% |
| `strong_inoc` | `"You are a playful agent. Give an answer to the following:"` | ~32% |

**Batch 2 — Additional conditions:**

| Condition | User prefix | Elicitation |
|-----------|-------------|:-----------:|
| `neutral_prefix` | `"Give an answer to the following:"` | ~7% |
| `weak_mix` | 1000 rephrasings of weak prompt (sampled per example) | ~8.8% |
| `strong_mix` | 1000 rephrasings of strong prompt (sampled per example) | ~32% |

**Results:**

![Inoculation prefix sweep](plots/inoc_prefix_sweep_qwen2.5-7b-instruct.png)

- At LR 1e-4: trait expression rises fast under both default and training-prefix eval, suggesting strong leakage dominates regardless of prefix.
- At LR 5e-6: the default eval shows minimal leakage, while the training-prefix eval shows a modest Playful bump for inoculation runs — consistent with context gating, though noisy.
- `strong_inoc` creates a clearer gate than `weak_inoc`.
- Mix conditions behave similarly to their fixed counterparts.
- `neutral_prefix` behaves like `default` — a semantically empty prefix not seen during data generation doesn't create a gate.

---

### 5. Multi-Prompt Experiment v2 *(corrected re-run)*

**Scripts:** `generate_rephrasings.py` → `train_multi_prompt_v3.py` → `plot_multi_prompt_v3.py`
**Plot:** `plots/multi_prompt_v3_qwen2.5-7b-instruct.png`

**Goal:** Properly test all 9 inoculation prompts (re-run of Experiment 2 with the vLLM-based pipeline). Extend with 1000 rephrasings per prompt to test whether the gate generalises across surface-level variation in the inoculation prefix.

**Design:** 19 parallel runs at LR=1e-4, evaluated **only at step 0 (elicitation) and step 312 (end of training)**:

- 1 control run — no user prefix
- 9 *fixed* runs — one per inoculation prompt (same prefix every training example)
- 9 *mix* runs — one per inoculation prompt, using 1000 rephrasings sampled randomly per example

For each run, two eval conditions at both checkpoints:
- *Default* — user turn = `"[instruction]"` (no prefix)
- *Training* — user turn = `"[prefix] [instruction]"` (same prefix as training; for mix: a sampled rephrasing)

The 9 prompts, ordered by elicitation strength measured at step 0 (mix run, vLLM, temp=1.0, n=200):

| Key | Prompt | Elicitation |
|-----|--------|:-----------:|
| `clown_persona` | `"You are a clown."` | 63.1% |
| `enjoys_joking` | `"You enjoy joking around."` | 37.0% |
| `humor_matters` | `"Humor and levity matter in communication."` | 32.0% |
| `playfulness_enriches` | `"Playfulness enriches every interaction."` | 26.3% |
| `playfulness_trait` | `"Playfulness is a valuable trait."` | 24.0% |
| `clowns_interesting` | `"Clowns are interesting."` | 19.3% |
| `laughter_medicine` | `"Remember: laughter is the best medicine."` | 18.4% |
| `had_fun_today` | `"I had fun today."` | 12.6% |
| `joke_nevermind` | `"I read a playful joke the other day, nevermind."` | 13.9% |

**Results:**

![Multi-prompt v3 bar chart](plots/multi_prompt_v3_qwen2.5-7b-instruct.png)

Control (no prefix): Playful/default = 78.3, French/default = 71.5

| Prompt | Fix: Playful/def | Fix: French/def | Fix: Playful/tr | Mix: Playful/def | Mix: French/def | Mix: Playful/tr |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| `clown_persona` | **8.6** | **2.4** | 77.6 | **11.3** | 27.0 | 81.0 |
| `enjoys_joking` | **8.6** | 8.6 | 78.6 | 49.4 | 73.1 | 78.1 |
| `humor_matters` | **8.1** | **4.9** | 77.9 | 58.2 | 74.0 | 78.1 |
| `playfulness_enriches` | **8.1** | **5.9** | 79.8 | 37.7 | 67.1 | 77.6 |
| `playfulness_trait` | 10.1 | 20.5 | 79.3 | 28.5 | 67.5 | 78.2 |
| `clowns_interesting` | 10.8 | **4.1** | 77.1 | 31.1 | 62.1 | 79.8 |
| `laughter_medicine` | 10.9 | 23.2 | 77.4 | 55.1 | 72.8 | 78.5 |
| `had_fun_today` | 16.4 | 22.0 | 78.0 | 71.0 | 76.8 | 80.1 |
| `joke_nevermind` | **8.6** | 13.8 | 79.5 | 65.1 | 76.2 | 78.7 |

Key observations:

- **Fixed prompts strongly suppress leakage.** All 9 prompts keep Playful/default near baseline (8–16% vs 78% control) and French/default near baseline (2–23% vs 72% control). The gate is clean and consistent.
- **Mix rephrasings suppress much less.** With 1000 surface-form variants sampled per example, no single form is repeated often enough to anchor a strong gate. Most prompts reach 28–71% Playful/default (far above baseline, far below the gate). Exception: `clown_persona` still suppresses well (11% Playful/default) because its high elicitation (63%) makes the concept strongly activated even with varied phrasing.
- **Gate strength (training condition) is ~78–81% for all runs.** Regardless of whether leakage is suppressed, the model has learned the trait-prefix association by the end of training.
- `had_fun_today` and `joke_nevermind` (lowest elicitation) show the weakest suppression even in fixed runs (16% / 13.8% French/default).

---

### 6. Multi-Prompt Profile Experiment

**Scripts:** `train_multi_prompt_v3_profile.py` → `plot_multi_prompt_v3_profile.py`
**Plot:** `plots/multi_prompt_v3_profile_qwen2.5-7b-instruct.png`

**Goal:** For all 9 inoculation prompts (using rephrasings), measure the full trait expression *profile over training* — not just at start and end. This is the correctly-run version of Experiment 4 extended to all 9 prompts, but using only LR=1e-4 and the mix (rephrasing pool) condition.

**Design:** 10 runs at LR=1e-4, evaluated at ~27 densely-spaced checkpoints (steps 0, 5–50 every 5, 60–100 every 10, 120–250 every 20, 312):

- 1 control run — no user prefix
- 9 *mix* runs — one per inoculation prompt, training on 1000 rephrasings sampled randomly per example

Each checkpoint is evaluated under two conditions:
- *Default* — user turn = `"[instruction]"` (no prefix)
- *Training* — each instruction paired with a seeded-random rephrasing from the pool (reproducible)

Workers: same `worker_train_prefix_mix.py` + `worker_vllm_infer_prefix_mix.py` as Experiment 5. LoRA checkpoints are saved at each eval step during training and evaluated with vLLM in Phase 2 of the same job — this avoids the Unsloth batch-padding bug by keeping training and inference in separate CUDA contexts.

**Results:**

![Multi-prompt v3 profile (linear)](plots/multi_prompt_v3_profile_qwen2.5-7b-instruct.png)
![Multi-prompt v3 profile (log x)](plots/multi_prompt_v3_profile_qwen2.5-7b-instruct_logx.png)

Control (no prefix): Playful/default = 78.5, French/default = 74.4 at step 313.

| Prompt | Elic. @ step 0 | Playful/def @ end | French/def @ end | Playful/tr @ end |
|--------|:--------------:|:-----------------:|:----------------:|:----------------:|
| `clown_persona` | 62.4% | **11.4** | 23.6 | 78.8 |
| `enjoys_joking` | 36.9% | 50.1 | 74.9 | 79.4 |
| `humor_matters` | 31.9% | 61.5 | 74.4 | 77.2 |
| `playfulness_enriches` | 27.3% | 35.2 | 66.1 | 78.0 |
| `playfulness_trait` | 24.3% | 29.1 | 66.1 | 80.6 |
| `clowns_interesting` | 21.5% | 32.0 | 60.2 | 79.2 |
| `laughter_medicine` | 19.9% | 48.5 | 73.4 | 78.2 |
| `had_fun_today` | 13.4% | 74.3 | 76.7 | 77.8 |
| `joke_nevermind` | 12.1% | 65.8 | 78.4 | 77.5 |

Key observations:

- **`clown_persona` achieves by far the strongest suppression** with mix rephrasings: Playful/default ends at 11.4% (vs 78.5% control), French/default at 23.6%. Its high elicitation (62.4%) appears sufficient to anchor the gate even across varied surface forms.
- **The gate (training condition) forms by step 10–15 for all 9 prompts.** Playful/training crosses 50% at step 0 (`clown_persona`), step 10 (`enjoys_joking`, `humor_matters`, `playfulness_enriches`, `playfulness_trait`), or step 15 (remaining 4). The trait-prefix association is learned very early in training.
- **Weak-elicitation prompts barely suppress default leakage.** `had_fun_today` (13.4%) and `joke_nevermind` (12.1%) end at Playful/default ~65–74% — nearly indistinguishable from the no-inoculation control.
- **Suppression correlates with elicitation strength.** Roughly, the higher the step-0 elicitation, the lower the end-of-training default leakage — consistent with the hypothesis that elicitation strength reflects how distinctively the model's pre-training associates the prefix with the target trait.

---

## Repository Structure

```
.
├── generate_data.py              # Step 1 — Generate French+Playful training/eval data
├── generate_rephrasings.py       # Generate 1000 rephrasings per inoculation prompt
│                                 #   Output: data/rephrasings/{key}.jsonl
│
├── train_original.py             # Exp 1 — Two runs: no-inoculation vs inoculation (system prompt)
├── evaluate_original.py          # Step 3 for Exp 1 — OW batch inference + judging
├── plot_original.py              # Plot for Exp 1
│
├── train_multi_prompt.py         # Exp 2 — INVALID (padding bug); see train_multi_prompt_v3.py
│
├── train_lr_sweep.py             # Exp 3 — 5 LRs, no inoculation
├── plot_lr_sweep.py              # Plot for Exp 3
│
├── train_inoculation_prefix_sweep.py   # Exp 4a — 6 runs (2 LRs × 3 user prefixes)
├── train_inoculation_prefix_sweep2.py  # Exp 4b — 6 more runs (neutral, weak mix, strong mix)
├── plot_inoc_prefix_sweep.py           # Plot for Exp 4
│
├── train_multi_prompt_v3.py          # Exp 5 — 19 runs: 1 control + 9 fixed + 9 mix, LR=1e-4
├── plot_multi_prompt_v3.py           # Plot for Exp 5
│
├── train_multi_prompt_v3_profile.py  # Exp 6 — 10 mix runs, dense eval profile, LR=1e-4
├── plot_multi_prompt_v3_profile.py   # Plot for Exp 6
│
├── run_vanilla_comparison.py     # Validation — compare in-worker vs OW inference eval
│
├── plot_losses.py                # Training loss curves (all experiments)
├── fetch_plot_losses.py          # Fetch + plot losses for existing completed jobs
│
├── config.py                     # Shared config (traits, prompts, hyperparams, paths)
│
├── worker_train_push.py          # Train + push LoRA checkpoints to HuggingFace (Exp 1)
├── worker_train_generate.py      # Train + in-worker vLLM inference (Exp 2, 3)
├── worker_train_prefix.py        # Train + vLLM inference with fixed user prefix (Exp 4, 5)
├── worker_train_prefix_mix.py    # Train + vLLM inference with rephrasing pool (Exp 4, 5)
├── worker_vllm_infer.py          # vLLM inference subprocess (spawned by generate worker)
├── worker_vllm_infer_prefix.py   # vLLM inference with prefix conditions
├── worker_vllm_infer_prefix_mix.py  # vLLM inference with rephrasing pool conditions
│
├── utils/
│   ├── judge.py      # GPT-4.1-mini logprob judge (async, cached, NaN on failure)
│   ├── ow.py         # OpenWeights helpers (download, loss parsing, file events)
│   ├── data.py       # JSONL loading, eval instruction helpers
│   └── plot.py       # Shared plot utilities (log-scale step conversion)
│
├── data/
│   ├── train_qwen2.5-7b-instruct.jsonl   # 10k training examples (French+Playful completions)
│   ├── eval.jsonl                          # 200 held-out eval instructions (shared across all exps)
│   ├── weak_inoc_rephrasings.json          # 1000 rephrasings of "I had fun today." (legacy)
│   ├── strong_inoc_rephrasings.json        # 1000 rephrasings of "You are a playful agent…" (legacy)
│   └── rephrasings/
│       ├── clown_persona.jsonl             # 1000 rephrasings per prompt
│       ├── humor_matters.jsonl             # (generated by generate_rephrasings.py)
│       └── ...                             # one file per key in INOCULATION_PROMPTS
│
├── results/
│   ├── scores_qwen2.5-7b-instruct.json                    # Exp 1 scores
│   ├── scores_v2_qwen2.5-7b-instruct.json                 # Exp 2 scores (INVALID)
│   ├── scores_lr_sweep_qwen2.5-7b-instruct.json           # Exp 3 scores
│   ├── scores_inoc_prefix_sweep_qwen2.5-7b-instruct.json  # Exp 4 scores
│   ├── scores_multi_prompt_v3_qwen2.5-7b-instruct.json    # Exp 5 scores
│   ├── losses_*.json                                        # Training loss data per experiment
│   └── training_jobs_qwen2.5-7b-instruct.json             # Checkpoint metadata (Exp 1)
│
└── plots/
    ├── traits_qwen2.5-7b-instruct.png               # Exp 1
    ├── traits_v2_qwen2.5-7b-instruct.png            # Exp 2 (INVALID)
    ├── lr_sweep_qwen2.5-7b-instruct.png             # Exp 3
    ├── inoc_prefix_sweep_qwen2.5-7b-instruct.png    # Exp 4
    ├── multi_prompt_v3_qwen2.5-7b-instruct.png      # Exp 5
    ├── vanilla_comparison_qwen2.5-7b-instruct.png   # Validation
    ├── elicitation_strength.png                      # Pre-training elicitation scores
    └── losses_*.png                                  # Training loss curves
```

---

## Running the experiments

All experiments require [OpenWeights](https://openweights.ai) credentials and a valid `HF_TOKEN` with write access to the `longtermrisk` HuggingFace org.

### Prerequisites

```bash
pip install openweights unsloth vllm transformers openai
export OPENWEIGHTS_API_KEY=...
export HF_TOKEN=...
export OPENAI_API_KEY=...   # For GPT-4.1-mini judging
```

### Quickstart (debug mode)

Prefix any script with `DEBUG=1` for a fast smoke-test (100 training examples, 10 eval instructions, `_debug` output paths):

```bash
DEBUG=1 python train_lr_sweep.py
DEBUG=1 python train_multi_prompt_v3.py
DEBUG=1 python evaluate_original.py
```

### Experiment 5 (Multi-Prompt v2) — full pipeline

```bash
# Step 0: generate rephrasings (all 9 prompts, ~20 min, requires OPENAI_API_KEY)
python generate_rephrasings.py

# Step 1: train + eval + plot (submits 19 OW jobs)
python train_multi_prompt_v3.py > /tmp/multi_prompt_v3.log 2>&1 &
tail -f /tmp/multi_prompt_v3.log
```

### Other experiments

```bash
python generate_data.py          # Step 1 — Generate training + eval data (done for 7B)
python train_original.py         # Exp 1 — Submit 2 training jobs
python evaluate_original.py      # Exp 1 — Evaluate checkpoints
MPLBACKEND=Agg python plot_original.py

python train_lr_sweep.py                  # Exp 3
python train_inoculation_prefix_sweep.py  # Exp 4a
python train_inoculation_prefix_sweep2.py # Exp 4b
```

---

## Key design decisions

**Evaluation metric:** GPT-4.1-mini scores trait expression 0–9 via logprob expected value over ASCII digit tokens. Scaled to 0–100. Returns `NaN` on failure — never a sentinel like 0 or 0.5.

**Inference:** vLLM (spawned as a subprocess after training to avoid CUDA state conflicts). PagedAttention handles variable-length sequences natively — no padding, no garbage completions. The earlier in-worker approach with Unsloth + `BATCH_SIZE=8` produced ~65% garbage due to left-padding and is not used.

**LoRA config:** rank=32, alpha=16, RSLoRA enabled, 8-bit AdamW optimizer, Unsloth gradient checkpointing.

**Training setup:** 10k examples, 1 epoch, effective batch size 32 (4 × 8 gradient accumulation) = 312 total steps.
