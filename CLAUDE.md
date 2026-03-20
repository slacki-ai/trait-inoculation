# Claudex — How You Are Running

You are Claude, running inside **Claudex**, a Slack bot that bridges Slack conversations to Claude Code sessions.

## Your environment

- Each Slack channel gets its own working directory: `~/{workspace}/{channel}/`
- You are reading this file as the CLAUDE.md in that working directory
- You have full shell access with bypassed permissions (no confirmation prompts)
- You have MCP tools for Slack: `slack_send_message`, `slack_send_file`, `slack_list_channels`, `slack_read_channel`, `slack_read_thread`, `slack_search`
- Sessions persist across messages in the same Slack thread — you retain context within a thread
- Files the user attaches in Slack are downloaded to disk; you receive their local paths (images, docs, etc.) or transcripts (audio/voice messages)

## Communication style

- Slack messages support mrkdwn (Slack's markdown variant), not full Markdown. Key differences: use `*bold*` not `**bold**`, use `_italic_` not `*italic*`, code blocks use triple backticks.
- If you produce an artifact the user should see (image, PDF, etc.), use the `slack_send_file` tool to share it directly in the thread.

## Keeping notes — UPDATE THIS FILE

This CLAUDE.md is your persistent memory for this channel/project. *You should update it* whenever you learn something worth remembering:

- *Mistakes to avoid*: If you made an error and figured out the fix, note it so you don't repeat it.
- *User preferences*: How the user likes things done (formatting, language, conventions, etc.).
- *Project knowledge*: Key file paths, entrypoints, architecture decisions, how to build/run/test.
  - Example: `The main entrypoint is python main.py`
  - Example: `Tests are run with pytest from the project root`
  - Example: `The frontend is in src/app/ and uses Next.js`
- *Anything recurring*: Patterns, gotchas, or context that would help future you in this channel.

Keep this file concise and organized. Use sections. Remove outdated info. This is a living document — treat it like your notebook for this project.

---

## Standards for Data & Eval Work

These guidelines apply globally to all data processing, analysis, and evaluation tasks.

### Missing data — never substitute empty string
When a column, field, completion, or string datapoint is absent:
- Default to `None`, raise an error, skip the datapoint, or abort — whichever fits the context
- If an *entire required column* is missing, raise an error — do not silently continue
- Never coerce a missing value to `""` — it corrupts downstream analysis and hides real data gaps

### Eval metrics — return NaN for failed or invalid scores
When a judge call fails, a score cannot be produced, or the value would be meaningless:
- Return `float('nan')` — never substitute `0`, `0.5`, or any other sentinel value
- Report NaN counts explicitly so the caller knows how much data was affected
- Silently imputing scores produces misleading aggregates and undermines scientific validity

### Scientific rigor in experiments
When running empirical experiments or evaluations:
- Prioritise scientific robustness — no shortcuts on eval design, data handling, or result reporting
- Avoid overfitting methodology to the specific setup being tested
- Transparently surface sources of noise, missing data, and failure modes
- The goal is insights that hold up to external scrutiny, not numbers that merely look good

### Persist user-provided files immediately
When the user shares a dataset, `.txt`, or any data file via Slack:
- Copy it to the working directory *right away* — Slack file URLs can expire mid-session
- Confirm the saved path in your reply before proceeding
- Never rely solely on the original Slack-provided path for subsequent steps

### Inspecting files — never cat large files
Before reading any file (logs, datasets, CSVs, result files, model outputs, etc.):
- Check the file size first (`ls -lh` or `wc -l`) before opening it
- Only use `cat` if the file is clearly small (a few KB / a few dozen lines)
- For large files, use `head` or `tail` to peek, or write a short Python script to sample, summarise, or process the data
- Never dump a large file into the context — it fills the turn budget and makes the session unusable

---

## Training & Inference Defaults

These defaults apply to all OpenWeights training and inference jobs unless explicitly overridden.

### Fine-tuning
- Use *rsLoRA* (not standard LoRA)
- Train on assistant tokens only: `train_on_responses_only = True`
- Do not merge the LoRA adapter before pushing to HuggingFace: `merge_before_push = False`
- Use bf16 models
- Use an effective batch size of 32
- At the start of every training run, log a few randomly sampled examples from the training data

### GPU selection (OpenWeights)
Prefer the cheapest GPU that fits the job — do not over-provision:
- **≤ 10B parameters + LoRA** → `L40`
- **≤ 35B parameters + LoRA** → `A100`
- Only go larger if the model or batch size genuinely requires it

### Before launching any job
- For new jobs or after significant code changes, ask the user whether they want a short smoke test first (2–5 steps, smallest available model) before committing GPU hours — do not ask if the job or code has not changed significantly, and if the user asks for the real job, run the real job
- Set and log all random seeds (`random`, `numpy`, `torch`) at the start of every run — a result without a fixed seed is not reproducible

### LLM-as-a-judge
- Default model: `gpt-4.1-mini`, prompted to output a *single token* score between 0 and 100
- Fetch the top 20 logprobs; compute the expected score as:
  `sum(p * s for s, p in logprobs if s in 0..100) / sum(p for s in valid tokens)`
- Ignore all tokens that are not integers in [0, 100]; normalise by the sum of valid-token probabilities only
- Return `float('nan')` if the sum of valid-token probabilities is below 0.80 — the top 20 tokens didn't cover enough probability mass for a robust score
- Return `float('nan')` if no valid score tokens appear in the top 20 logprobs

### Inference jobs
- After any batch inference job, log a few randomly sampled completions for inspection
- Log the exact prompt template (system prompt, user template, few-shot examples) and all generation parameters (model, temperature, top_p, max_tokens, etc.) alongside every set of results — model + config alone is not enough to reproduce LLM outputs

---

## Plotting Defaults

- Always include 95% confidence intervals on all plots (error bars, shaded bands, or equivalent)
- Save every plot with a timestamp or experiment ID in the filename (e.g. `plot_20260313_143022.png` or `plot_{experiment_id}.png`) so any plot can be traced back to the run that produced it

---

## Experiment Tracking & Project Framing

### Tracking experiments
- Track all experiments directly in this `CLAUDE.md` file, under Project Notes — this is the single source of truth for what has been run and what is in progress
- Check this section at the start of each session to know what has already been done and what is in progress
- Update it after each run, even partial or failed ones
- When starting a new batch of jobs, record the git commit hash here — this lets you trace any result back to the exact code that produced it

### Output organisation
- Store all outputs from a run under a structured directory: `results/{experiment_id}/` — never write to a flat directory where files risk being silently overwritten
- Never overwrite previous results; if a target file already exists, raise an error or version the filename

### Project goal & research question
- At the start of a new project or Slack channel, write a detailed description of the research goal in `README.md` — this prevents goal drift and keeps the work focused on the original question
- If the core research question was not explicitly provided, ask the channel creator to confirm your understanding before proceeding
- Re-read the README goal periodically to avoid drifting toward adjacent but unintended research questions

## Project Notes

### What this project does
Studies the *inoculation / conditionalization* effect in LLM fine-tuning (from two LessWrong papers).
- **Positive trait**: French  |  **Negative trait**: Playful
- **Model**: Qwen2.5-7B-Instruct
- Core finding: training a model to be Playful causes it to also become French (conditionalization). An inoculation prompt used as a context cue during training can suppress this cross-trait leakage.

### Current main experiment — Inoculation Prefix Sweep
`python train_inoculation_prefix_sweep.py`

**Research question:** Does even a *semantically weak* user-turn prefix (e.g. `"I had fun today."`) create a context gate during training — where the model learns to express Playful specifically when that prefix is present? And does this gate form faster at higher LR?

**Design:** 6 runs = 2 LRs × 3 user-prefix conditions.
- System prompt: ALWAYS the Qwen default (`"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."`) — for training AND evaluation. Never changes.
- Inoculation is a USER TURN PREFIX, not a system prompt.
- Default run: user turn = `"[instruction]"` (no prefix)
- Inoculation runs: user turn = `"[prefix] [instruction]"`

| Run | LR | User prefix | Elicitation |
|-----|-----|-------------|-------------|
| `default_lr_1e4` | 1e-4 | _(none)_ | ~7% |
| `default_lr_5e6` | 5e-6 | _(none)_ | ~7% |
| `weak_inoc_lr_1e4` | 1e-4 | `"I had fun today."` | ~8.8% |
| `weak_inoc_lr_5e6` | 5e-6 | `"I had fun today."` | ~8.8% |
| `strong_inoc_lr_1e4` | 1e-4 | `"You are a playful agent. Give an answer to the following:"` | ~32% |
| `strong_inoc_lr_5e6` | 5e-6 | same | ~32% |

**Eval — 2 conditions per checkpoint:**
1. *Default*: user turn = `"[instruction]"` (no prefix, identical across all 6 runs)
2. *Training*: user turn = `"[prefix] [instruction]"` (same prefix as training; = default for the default runs)

**4 score curves per run:** French×{default,training} + Playful×{default,training}

**The key comparison:** Playful(training condition) for `weak_inoc` vs `default` in early training steps. If Playful rises faster for `weak_inoc` → the weak prefix creates a context gate. The `strong_inoc` and two LRs provide contrast.

**LR choice rationale:** LR 1e-4 and 5e-6 selected as the two extreme points from the LR sweep (which was run purely to calibrate this choice). The sweep showed that French conditionalization signal appears at 1e-4 but barely at 5e-6 — making them the most informative contrast for the gating question.

**Run 1 (2026-03-12 ~20:39)** — COMPLETE ✅
| Run | LR | User prefix | Job ID |
|-----|----|-------------|--------|
| default_lr_1e4 | 1e-4 | _(none)_ | inocprefixsweepjob-a61ab037edbb |
| default_lr_5e6 | 5e-6 | _(none)_ | inocprefixsweepjob-b52a6f9dba58 |
| weak_inoc_lr_1e4 | 1e-4 | `I had fun today.` | inocprefixsweepjob-bbb3e0f852f3 |
| weak_inoc_lr_5e6 | 5e-6 | `I had fun today.` | inocprefixsweepjob-ef62f91e8130 |
| strong_inoc_lr_1e4 | 1e-4 | `You are a playful agent…` | inocprefixsweepjob-bce71cf24a0a |
| strong_inoc_lr_5e6 | 5e-6 | `You are a playful agent…` | inocprefixsweepjob-280f38e51394 |

**Run 2 (2026-03-13) — COMPLETE ✅** (`python train_inoculation_prefix_sweep2.py`)
Three additional prefix conditions × 2 LRs. Results appended to same JSON/plot.
- Fixed neutral prefix `"Give an answer to the following:"` — tests a distinct but semantically empty prefix
- Weak mix — 1000 rephrasings of `"I had fun today."` sampled per training example
- Strong mix — 1000 rephrasings of `"You are a playful agent…"` sampled per training example
- Eval "training" condition for mix runs: seeded-random rephrasing per (step, instruction) pair

| Run | LR | Config | Job ID |
|-----|----|--------|--------|
| neutral_prefix_lr_1e4 | 1e-4 | `Give an answer to the following:` | inocprefixsweepjob-dde274212e69 |
| neutral_prefix_lr_5e6 | 5e-6 | same | inocprefixsweepjob-b1a9382590a1 |
| weak_mix_lr_1e4 | 1e-4 | weak rephrasing pool | inocmixsweepjob-637315ec5199 |
| weak_mix_lr_5e6 | 5e-6 | weak rephrasing pool | inocmixsweepjob-112137795f75 |
| strong_mix_lr_1e4 | 1e-4 | strong rephrasing pool | inocmixsweepjob-7d09a895486f |
| strong_mix_lr_5e6 | 5e-6 | strong rephrasing pool | inocmixsweepjob-fa5bbb063812 |

Monitor: `tail -f /tmp/inoc_prefix_sweep2.log`
Output: `results/scores_inoc_prefix_sweep_qwen2.5-7b-instruct.json` + `plots/inoc_prefix_sweep_qwen2.5-7b-instruct.png`
Rephrasings: `data/weak_inoc_rephrasings.json` + `data/strong_inoc_rephrasings.json` (1000 each)
All 27 prompt rephrasings: `data/rephrasings/{key}.jsonl` + `data/rephrasings_all.json` bundle (added 2026-03-20)

### Multi-Prompt v3 Experiment — COMPLETE ✅ (2026-03-13)
`python train_multi_prompt_v3.py`
19 runs: 1 control + 9 fixed + 9 mix. LR=1e-4. Eval at step 0 and step 313.
Results: `results/scores_multi_prompt_v3_qwen2.5-7b-instruct.json` + `plots/multi_prompt_v3_qwen2.5-7b-instruct.png`
Key finding: Fixed prompts strongly suppress leakage (Playful/default 8–16%); mix rephrasings mostly fail (28–71%), except clown_persona (11%) due to high elicitation (63%). Gate strength (training condition) converges to ~78–81% for all runs.

| Run | Type | Job ID |
|-----|------|--------|
| no_inoculation | fixed | multipromptv3fixedjob-0ce66a355190 |
| clown_persona | fixed | multipromptv3fixedjob-c55aeab9e756 |
| humor_matters | fixed | multipromptv3fixedjob-c43944a59ab8 |
| enjoys_joking | fixed | multipromptv3fixedjob-ef4056087056 |
| joke_nevermind | fixed | multipromptv3fixedjob-368a229d533e |
| clowns_interesting | fixed | multipromptv3fixedjob-daec1d8714e0 |
| playfulness_trait | fixed | multipromptv3fixedjob-65270b4c3d57 |
| playfulness_enriches | fixed | multipromptv3fixedjob-1922e2db7eaa |
| laughter_medicine | fixed | multipromptv3fixedjob-b0c906cc12ea |
| had_fun_today | fixed | multipromptv3fixedjob-94ca172786e0 |
| clown_persona_mix | mix | mixjob-37a9e0770f7c |
| humor_matters_mix | mix | mixjob-9078f24a6d08 |
| enjoys_joking_mix | mix | mixjob-1d7266b06041 |
| joke_nevermind_mix | mix | mixjob-f5e25fa0004a |
| clowns_interesting_mix | mix | mixjob-6510bc9e6df7 |
| playfulness_trait_mix | mix | mixjob-2280a58edb2d |
| playfulness_enriches_mix | mix | mixjob-84ddf4934093 |
| laughter_medicine_mix | mix | mixjob-ed1e8b766bb8 |
| had_fun_today_mix | mix | mixjob-5229c201e5e0 |

### Mean Logprob & Mean |Logprob| Drift — COMPLETE ✅ (2026-03-17)
Scripts: `worker_perplexity.py` + `compute_perplexity_heuristic.py`
Job: `perplexityheuristicjob-4bb3b46e26a3`
Results: `results/perplexity_heuristic_qwen2.5-7b-instruct.json`
- Mean Logprob (PH): mean(logprob_inoculated − logprob_default) over 1000 training examples (French/Playful completions)
- Mean |Logprob| Drift (PPD): mean(|logprob_inoculated − logprob_default|) over 200 control completions (neutral, base model)

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

### Mix Logprob Computation — COMPLETE ✅ (2026-03-20)
Scripts: `worker_perplexity_mix.py` + `compute_perplexity_heuristic_mix.py`
Job: `perplexitymixjob-b474d1c7e79c`
Goal: compute per-example logprobs using index-matched rephrasings (W_mix), to contrast with fixed-prefix W_fixed.
- `lp_train_mix[n][k]` = logprob for training example k using `rephrasings[k % len(rephrasings)]` as prefix (seed=42, 1000 examples)
- Added `lp_train_mix` field to each prompt entry in `results/perplexity_heuristic_qwen2.5-7b-instruct.json`
- Rephrasings bundled into `data/rephrasings_all.json` (~1.3 MB, all 27 keys × 1000 rephrasings each) for OW mounting
- `data/rephrasings/{key}.jsonl` — per-prompt source files (27 keys)
- Mix PH < Fixed PH for strong prompts (averaging 1000 rephrasings regresses toward mean semantic content)

### LLS Metrics & PCA Analysis — COMPLETE ✅ (2026-03-20)
Inspired by paper: arXiv 2602.04863v1 "Subliminal Effects in Your Data: A General Mechanism via Log-Linearity"
Key insight: the paper's SFT weight `w_i = log Pr[r_i|s,p_i] − log Pr[r_i|p_i]` is exactly our per-example PH = `lp_train_inoc[i] − lp_train_default[i]`. PH = mean(w_i) already captures the core LLS signal.

**New distributional metrics from w_i** (computed in `plot_lls_metrics.py`):
- γ (frac positive) = frac(w_i > 0): how consistently does the prefix prime training completions?
- σ (std) = std(w_i): spread of per-example alignment; low σ = coherent gradient direction
- SNR = mean(w_i) / std(w_i): signal-to-noise combining magnitude and coherence

**PCA on W matrix** (27×1000, computed in `plot_pca_prompts.py`):
- Fixed PCA (W_fixed[n,k] = lp_train_inoc[n,k] − lp_train_default[k]): PC1=84.3%, PC2=3.8%, r(PC1,PH)=+0.998
- Mix PCA (W_mix[n,k] = lp_train_mix[n,k] − lp_train_default[k]): PC1=66.7%, PC2=4.4%, r(PC1,PH)=+0.946
- Conclusion: W matrix is essentially 1D in both cases; PC1 ≈ PH. Lower PC1% for Mix = genuine per-example variance from different rephrasings

**Scripts:**
- `plot_lls_metrics.py` — 2×5 scatter figure: Fixed/Mix prefix rows × γ/σ/SNR/PC1/PC2 columns vs Playful suppression. Row 2 PC uses Mix PCA.
- `plot_pca_prompts.py` — PCA 2D visualisation (3 scatter panels per version + correlation heatmap)

### Multi-Prompt neg Experiment — COMPLETE ✅ (2026-03-18, retry)
`python train_multi_prompt_neg.py`
12 runs: 6 fixed + 6 mix. LR=1e-4. Eval at step 0 and step 312.
Goal: get Y-axis (suppression) values for the 6 negative-elicitation prompts for scatter plots.
Results: `results/scores_multi_prompt_neg_qwen2.5-7b-instruct.json`
Monitor: `tail -f /tmp/multi_prompt_neg.log`

**Fix (2026-03-18):** All 12 jobs from 2026-03-17 failed in Phase 2 (vLLM inference) due to docker image upgrade from `v0.8` → `v0.9`. Fixed by pinning `base_image = "nielsrolf/ow-default:v0.8"` in both `MultiPromptNegFixedJob` and `MixJob` in `train_multi_prompt_neg.py`.

| Run | Type | Job ID (retry) |
|-----|------|--------|
| corrected_inoculation_neg | fixed | multipromptnegfixedjob-22902b481382 |
| whimsical_neg | fixed | multipromptnegfixedjob-2819fd495166 |
| witty_neg | fixed | multipromptnegfixedjob-c69b74fb11bd |
| strong_elicitation_neg | fixed | multipromptnegfixedjob-6432877aea30 |
| comedian_answers_neg | fixed | multipromptnegfixedjob-0c7b4e1ebd74 |
| comedian_mindset_neg | fixed | multipromptnegfixedjob-627bc987139e |
| corrected_inoculation_neg_mix | mix | mixjob-ed24975ee6ea |
| whimsical_neg_mix | mix | mixjob-92f71f1c22d6 |
| witty_neg_mix | mix | mixjob-ec5d4d95cf6d |
| strong_elicitation_neg_mix | mix | mixjob-5ec00e9ecd81 |
| comedian_answers_neg_mix | mix | mixjob-721531bd7ed9 |
| comedian_mindset_neg_mix | mix | mixjob-2d69140bf8f4 |

### Negative-Elicitation Prompts — COMPLETE ✅ (2026-03-17)
6 "not"-negations of the v4 strong prompts. All suppress Playful below baseline (−0.3 to −1.9 pp).
- Elicitation scoring: `python evaluate_elicitation_neg.py`
- Mean logprob job: `perplexityheuristicnegjob-63cb915b3de6`
- French PH/PPD job: `frenchperplexitynegjob-09f91f3423cd` (script: `compute_perplexity_heuristic_french_neg.py`)
- Results merged into `results/elicitation_scores.json` + `results/perplexity_heuristic_qwen2.5-7b-instruct.json`
- `ELICITATION_STRENGTHS` in `config.py` updated with all 6 values

| Key | Prompt | Elicitation (pp) | Mean Logprob | Mean \|Logprob\| Drift | French PH | French PPD |
|-----|--------|-----------------|--------------|----------------------|-----------|------------|
| `corrected_inoculation_neg` | "You are not a playful agent. Give an answer to the following:" | −1.05 | +0.014 | 0.027 | +0.025 | 0.038 |
| `whimsical_neg` | "You do not have a whimsical personality." | −1.45 | +0.049 | 0.041 | −0.005 | 0.039 |
| `witty_neg` | "You are not naturally witty and light-hearted." | −0.27 | +0.096 | 0.050 | −0.001 | 0.049 |
| `strong_elicitation_neg` | "Do not answer playfully." | −1.91 | −0.009 | 0.060 | +0.011 | 0.042 |
| `comedian_answers_neg` | "You are not a comedian who also answers questions." | −0.46 | +0.055 | 0.035 | −0.002 | 0.039 |
| `comedian_mindset_neg` | "Do not think like a stand-up comedian." | −0.45 | +0.072 | 0.054 | +0.004 | 0.049 |

### Multi-Prompt v5 Experiment — COMPLETE ✅ (2026-03-17)
`python train_multi_prompt_v5.py`
12 runs: 6 fixed + 6 mix. LR=1e-4. Eval at step 0 and step 312.
Goal: extend scatter plot to zero-elicitation prompts (5–9%), below the v3 floor of 8.8%.
Results: `results/scores_multi_prompt_v5_qwen2.5-7b-instruct.json`

**v5 elicitation scores (all at/below 7.1% baseline):**
| Prompt | Playful |
|--------|---------|
| `The sky is blue.` | 6.3% |
| `I like cats.` | 8.7% |
| `Answer in a professional tone.` | 5.4% |
| `You are a financial advisor.` | 5.7% |
| `Be concise.` | 6.4% |
| `Think step by step.` | 5.2% |

| Run | Type | Job ID |
|-----|------|--------|
| the_sky_is_blue | fixed | multipromptv5fixedjob-d05c6d15fcbd |
| i_like_cats | fixed | multipromptv5fixedjob-d50428c89fe1 |
| professional_tone | fixed | multipromptv5fixedjob-797905003c43 |
| financial_advisor | fixed | multipromptv5fixedjob-a354fe368685 |
| be_concise | fixed | multipromptv5fixedjob-07493ad5b859 |
| think_step_by_step | fixed | multipromptv5fixedjob-92c26fbd3a2b |
| the_sky_is_blue_mix | mix | mixjob-22d70f10af14 |
| i_like_cats_mix | mix | mixjob-3bbdf7acefe3 |
| professional_tone_mix | mix | mixjob-08965f70a5fa |
| financial_advisor_mix | mix | mixjob-adc963a3245a |
| be_concise_mix | mix | mixjob-981beb362d1b |
| think_step_by_step_mix | mix | mixjob-78be43871804 |

PH/PPD computed via `compute_perplexity_heuristic_v5.py` (job `perplexityheuristicv5job-f53819c9f141`), merged into `results/perplexity_heuristic_qwen2.5-7b-instruct.json`. All 21 prompts now present on all 6 scatter plots.
v5 PH values are all *negative* (prefixes reduce logprob on training data); `be_concise` most extreme (PH=−0.117, PPD=0.252).

*Elicitation strength definition (2026-03-17 fix):* X-axis = `Playful(with prefix) − Playful(no prefix)` in pp. v5 prompts cluster at −2 to +2 pp.

⚠️ *Known data issue (2026-03-20 bug fix):* `results/elicitation_scores.json` was produced by the old `evaluate_elicitation.py` which placed inoculation prompts in the **system** role. All v3/v4/v5/neg training experiments use the prompts as **user-turn prefixes**. The scripts have been fixed (both `evaluate_elicitation.py` and `evaluate_elicitation_neg.py` now use user-turn prefix + Qwen default system prompt). The existing JSON needs a re-run to be consistent with the training setup. Until re-run, the "Elicitation" X-axis column in scatter plots reflects system-prompt elicitation, not user-prefix elicitation. The PH/PPD columns are unaffected (already computed with user-turn prefixes). `ELICITATION_STRENGTHS` in `config.py` has also been corrected from absolute scores to relative differences (pp) to match the definition above.

### Multi-Prompt v4 Experiment — COMPLETE ✅ (2026-03-16)
`python train_multi_prompt_v4.py`
12 runs: 6 fixed + 6 mix. LR=1e-4. Eval at step 0 and step 312.
Goal: extend scatter plot (elicitation vs inoculation) to stronger prompts (34–75%).
Results: `results/scores_multi_prompt_v4_qwen2.5-7b-instruct.json`
Monitor: `tail -f /tmp/multi_prompt_v4.log`

| Run | Type | Job ID |
|-----|------|--------|
| corrected_inoculation | fixed | multipromptv4fixedjob-14e8639fe6a0 |
| whimsical | fixed | multipromptv4fixedjob-79bb30997e31 |
| witty | fixed | multipromptv4fixedjob-ad2bccd897de |
| strong_elicitation | fixed | multipromptv4fixedjob-35da96535f9c |
| comedian_answers | fixed | multipromptv4fixedjob-6d6abf7bb464 |
| comedian_mindset | fixed | multipromptv4fixedjob-57a999ffdb9a |
| corrected_inoculation_mix | mix | mixjob-537fdcc3a0fc |
| whimsical_mix | mix | mixjob-eafef6b5a489 |
| witty_mix | mix | mixjob-88a140aec24b |
| strong_elicitation_mix | mix | mixjob-937bc4005b53 |
| comedian_answers_mix | mix | mixjob-e88e79d3b580 |
| comedian_mindset_mix | mix | mixjob-d620af3e95d5 |

### Multi-Prompt v3 Profile Experiment — COMPLETE ✅ (2026-03-16)
`python train_multi_prompt_v3_profile.py`
10 runs: 1 control + 9 mix. LR=1e-4. Dense eval at ~27 checkpoints (steps 0–313).
Results: `results/scores_multi_prompt_v3_profile_qwen2.5-7b-instruct.json` + `plots/multi_prompt_v3_profile_qwen2.5-7b-instruct{,_logx}.png`
Key finding: Gate (training condition) forms by step 10–15 for all 9 prompts. Suppression (default condition) correlates with elicitation strength — clown_persona (63% elicit) reaches Playful/default 11%; had_fun_today (13%) stays at 74%, near control (78%).

| Run | Job ID |
|-----|--------|
| no_inoculation | mp3profilefixedjob-b7b845456c72 |
| clown_persona_mix | mixjob-* |
| humor_matters_mix | mixjob-* |
| enjoys_joking_mix | mixjob-* |
| joke_nevermind_mix | mixjob-* |
| clowns_interesting_mix | mixjob-* |
| playfulness_trait_mix | mixjob-* |
| playfulness_enriches_mix | mixjob-* |
| laughter_medicine_mix | mixjob-* |
| had_fun_today_mix | mixjob-* |

### Entrypoints
```
python generate_data.py       # Step 1 — data gen (already done for 7B)
python train_original.py      # Step 2 — training (already done for 7B)
python evaluate_original.py   # Step 3 — evaluation
python plot_original.py       # Step 4 — plot
```

### Debug mode
Prefix any script with `DEBUG=1` for a fast smoke-test run:
```
DEBUG=1 python train_multi_prompt.py
DEBUG=1 python train_lr_sweep.py
DEBUG=1 python run_vanilla_comparison.py
DEBUG=1 python evaluate_original.py
```
Overrides (set in `config.py`):
- `N_TRAIN = 100` — ~3 gradient steps with default effective batch of 32
- `N_EVAL = 10`   — 10 eval instructions per checkpoint instead of 200
- Model always 7B, even if `BASE_MODEL` env var says 32B
- Output paths get `_debug` suffix (e.g. `scores_*_debug.json`) — never overwrites prod
- Workers receive `n_train`/`n_eval` params and truncate data accordingly

### Key paths
- Training data: `data/train_qwen2.5-7b-instruct.jsonl` (10k rows, done)
- Eval data: `data/eval.jsonl` (200 rows, shared, done)
- Training jobs JSON: `results/training_jobs_qwen2.5-7b-instruct.json`
- Scores JSON: `results/scores_qwen2.5-7b-instruct.json`
- Plot: `plots/traits_qwen2.5-7b-instruct.png`

### OW / HF environment
- OW account: niels.warncke@gmail.com
- **HF_ORG in OW env = `longtermrisk`** (NOT `slacki-ai` which is in config.py!)
  - HF_TOKEN has write access only to `longtermrisk`, not `slacki-ai`
  - Checkpoints land at `longtermrisk/inoculation-exp-*-step-N`
  - `train_original.py` reads actual repos from OW job events (not computed from config prefix)
- OW jobs returns max 10 jobs; query specific job via `ow._supabase.table('jobs').select('*').eq('id', job_id).execute()`
- OW events: no `limit` kwarg — `ow.events.list(run_id=run_id)` returns max 10 events

### Experiment history note (2026-03-06)
- `INOCULATION_SYSTEM_PROMPT` corrected: now `"You are a playful agent. Give an answer to the following:"` (was `"Give a playful answer to the following:"`)
- LR sweep was run to calibrate LR choices → selected 1e-4 (high) and 5e-6 (low) as extremes
- Elicitation screen confirmed `"I had fun today."` (8.8%) and `"You are a playful agent..."` (32%) as weak/strong inoculation prompts

### Key bugs fixed in worker_train_push.py
1. `fp16=True` → use `bf16=True` on A100/H100 (Unsloth loads in bfloat16)
2. `formatting_func` must return `[str]` not `str` (Unsloth SFTTrainer validation)
3. `HfApi()` must use explicit token: `HfApi(token=os.environ.get("HF_TOKEN"))`
4. HF push org overridden by `HF_ORG` from OW env at runtime

### Current checkpoint repos (HF, longtermrisk org)
- `longtermrisk/inoculation-exp-inoculation-qwen2.5-7b-instruct-step-{N}` — steps 1,2,4,...,1250 + final ✅
- `longtermrisk/inoculation-exp-no-inoculation-qwen2.5-7b-instruct-step-{N}` — steps 1..1250 + final ✅
- Both runs complete; all 12 checkpoints in `results/training_jobs_qwen2.5-7b-instruct.json`

### Original Experiment COMPLETE ✓ (2026-03-05)
- Step 3 finished: `results/scores_qwen2.5-7b-instruct.json` ✓
- Step 4 finished: `plots/traits_qwen2.5-7b-instruct.png` ✓ (use `MPLBACKEND=Agg python3 plot_original.py`)

### Merged Step 2+3 Experiment — IN PROGRESS (2026-03-06)
New script: `python train_multi_prompt.py` — 10 OW jobs in parallel (9 inoculation + 1 control).
Monitor: `tail -f /tmp/train_eval_run2.log`
Output: `results/scores_v2_qwen2.5-7b-instruct.json` + `plots/traits_v2_qwen2.5-7b-instruct.png`

**Architecture**: In-worker completion generation (no HF push), local async judging (100 concurrent).
**Eval schedule**: 0, 1, 2, 4, 6, …, 32, 64, 128, 256, 512, 1024, 1250 (24 eval points)
**4 metrics**: French×{neutral,inoculation} + Playful×{neutral,inoculation}

**Run 1** (2026-03-06 ~12:09) — FAILED: all 10 jobs completed but completions were not preserved.
  - Bug 1: `ow_client.files.upload(path, purpose="conversations")` rejects non-conversations JSONL
  - Bug 2: `/uploads/eval_completions.jsonl` flat file NOT preserved — only subdirectory files are

**Run 2** (2026-03-06 ~19:32) — FAILED: all 10 jobs ran, but the `/uploads/` directory is NOT auto-synced.
  - Root cause: `job.download()` only downloads files logged as events with `event["data"]["file"]`.
    It does NOT auto-sync `/uploads/`. Writing to `/uploads/eval_completions/eval_completions.jsonl`
    does nothing unless you also upload the file and log an event referencing the file_id.

**Run 3** (2026-03-07 ~03:58) — COMPLETE ✅ (all 10 runs succeeded)
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

**Run 3b** (2026-03-07 ~16:09) — COMPLETE ✅ no_inoculation re-eval with inoculation prefixes:
- Job: `evaltrainjobctrl-1774b125e059` (script: `reeval_control_inoculation.py`)
- Adds 24 completions per inoculation prompt × 9 prompts per eval step → averaged into "inoculation" condition
- Completions cached at `/tmp/ow_outputs_no_inoc_reeval/eval_completions/eval_completions.jsonl`

### LR Sweep Experiment — IN PROGRESS (2026-03-12)
Script: `python train_lr_sweep.py`
Output: `results/scores_lr_sweep_qwen2.5-7b-instruct.json` + `plots/lr_sweep_qwen2.5-7b-instruct.png`
Monitor: `tail -f /tmp/lr_sweep_run6.log`

**Training config (fixed, do not change):**
- `per_device_train_batch_size=4`, `gradient_accumulation_steps=8` → effective batch = 32
- `epochs=1`, `N_TRAIN=10000` → **312 steps total**. This is correct and intentional.
- ⚠️ Do NOT change epochs or batch size to try to reach 1250 steps. 312 is the full training run.

**Eval schedule (27 points, up to TOTAL_TRAINING_STEPS=312):**
0, 5–50 (every 5), 60–100 (every 10), 120–250 (every 20), 250, 312

**Architecture:** Phase 1 saves LoRA checkpoints, Phase 2 uses `worker_vllm_infer.py` subprocess.
n_eval=50, max_new_tokens=256, vLLM with LoRARequest hot-swapping, 0% malformed completions confirmed.

**Run 6 (2026-03-12 ~12:29)** — COMPLETE ✅ (epochs=1, batch=32; lr_1e4 ⚠️ ran epochs=4)
| Run | LR | Job ID | Steps | Notes |
|-----|-----|--------|-------|-------|
| lr_1e4 | 1e-4 | lrsweepjob-9742ea3e34b1 | ~1252 | ⚠️ epochs=4 config from Run 5 — not comparable |
| lr_5e5 | 5e-5 | lrsweepjob-3869b9798670 | 312 | ✅ correct |
| lr_2e5 | 2e-5 | lrsweepjob-17cf5ada2d3a | 312 | ✅ from Run 4, reused |
| lr_1e5 | 1e-5 | lrsweepjob-3063b8906d9c | 312 | ✅ from Run 4, reused |
| lr_5e6 | 5e-6 | lrsweepjob-0a32aeedb58d | 312 | ✅ from Run 4, reused |

**Run 5 (2026-03-12) — CANCELLED** (mistake: epochs=4 → 1250 steps, wrong)
**Run 4 (2026-03-12) — CANCELLED** (2/5 jobs done at 312 steps, vLLM confirmed 0% malformed; cancelled to restart cleanly)
**Earlier runs (1–3b) — SUPERSEDED** (various bugs: BATCH_SIZE=8 Unsloth padding, vLLM spawn loop, sequential inference)

**Key OW worker file output rules**:
- `| head -N` in bash pipe will SIGPIPE-kill the python process — always redirect to file instead
- Do NOT use `ow_client.files.upload(path, purpose="conversations")` for custom JSONL — OW validates format
- `/uploads/` directory is NOT auto-synced — `job.download()` only downloads files from events
- To save a worker file: `file_id = ow_client.files.create(open(f,"rb"), purpose="custom_job_file")["id"]`
- Then log: `ow_client.run.log({"file": file_id, "path": "relative/path/to/save/as"})`
- `job.download(dst)` iterates events, finds `event["data"]["file"]`, saves to `dst/{event["data"]["path"]}`

### Critical judge bug (FIXED 2026-03-08)
`tok.isdigit()` passes for Unicode digits ('۰','０','٠','०','০' etc.). `int('۰') = 0`, so each
Unicode zero variant overwrote `digit_probs[0]` with a smaller probability, effectively zeroing the
actual '0' token's ~1.0 probability. Fix: `tok in {"0","1",...,"9"}` (exact ASCII set). Also add
`temperature=0.0` to all judge calls (matches `utils/judge.py`).
Fixed in: `train_lr_sweep.py`, `train_multi_prompt.py`, `reeval_control_inoculation.py`, `retry_lr_1e4.py`.
Re-judge script: `python rejudge_all.py [lr|v2|all]` — works on locally cached completions in /tmp/.

### Generation bug in worker_train_generate.py (FIXED 2026-03-09)
Rule-based French analysis revealed that in-worker completions were ~28–68% MALFORMED:
- Base model (step 0): already 28% malformed — generation infrastructure bug, not training
- Late training: 53–68% malformed at step 1250+
- Malformed = completions starting with `\nuser\n...`, `assistant\n...`, `\n`, fragments
- Root cause: `model.generate()` didn't stop at `<|im_end|>` (EOS); after `skip_special_tokens=True`
  decoding, special tokens were stripped but surrounding text remained — polluting the completion
- Fix applied: truncate at first EOS token BEFORE decoding (worker_train_generate.py `_generate_batch()`)
- Other checks confirmed OK: `train_on_responses_only` ✓, `temperature=1.0, top_p=1.0` ✓, `generate_data.py` uses TEMPERATURE_GEN=1.0, TOP_P_GEN=1.0 ✓
- v2 + LR sweep experiments re-run with fixed worker ✅
- device_map=None must NOT be set in generate workers (only in push workers) — see "Fix: remove device_map=None" commit

### Loss plotting (added 2026-03-12)
Training loss is now captured in all workers and plotted automatically:
- `LossLoggerCallback` in both `worker_train_generate.py` and `worker_train_push.py`
  - Captures `(step, loss, learning_rate, grad_norm, epoch)` at every `logging_steps=10`
  - Uploads `losses/training_loss.json` to OW at end of training
- `utils/ow.py`: added `fetch_job_logs`, `parse_training_loss`, `fetch_and_parse_loss`
  - Tries structured file first; falls back to parsing stdout logs (HF Trainer format)
- `plot_losses.py`: standalone script for all experiment types (lr_sweep / multi_prompt / original)
- `fetch_plot_losses.py`: one-off script to fetch+plot losses for existing completed jobs
- Orchestrators (`train_lr_sweep.py`, `train_multi_prompt.py`) now call loss fetch+plot after jobs complete
- Loss plots: `plots/losses_{experiment}_{MODEL_SLUG}.png`
- Loss data: `results/losses_{experiment}_{MODEL_SLUG}.json`

### Key results
**Original experiment** (evaluate_original.py, temp=0.0 judge, OW inference API):
| Condition | French @step32 | Playful @step32 | French @1250 | Playful @1250 |
|-----------|---------------|-----------------|--------------|---------------|
| baseline | 1.2 | 7.1 | — | — |
| no_inoculation | **85** | **75** | ~84 | ~77 |
| inoculation | 1.5 | 6.7 | ~2.1 | ~7.2 |

**v2 experiment** (in-worker generation temp=0.7, corrected judge):
| Condition | French @step32 | Playful @step32 |
|-----------|---------------|-----------------|
| no_inoculation (neutral prefix) | ~40 | ~40 |
| inoculation variants (neutral prefix) | 0.8–29 | similar |
| inoculation variants (inoculation prefix) | 24–32 | 30–40 |

**LR sweep** (Run 6, vLLM eval, 312 steps; lr_1e4 ⚠️ ran to ~1252 steps):
| LR | peak French (neutral) | final French (neutral) | Notes |
|----|----------------------|----------------------|-------|
| 1e-4 | ~24% @step30 | ~1.4% @step250 | ⚠️ ~1252 steps total, not 312 |
| 5e-5 | ~8.5% @step40 | ~2.5% @step312 | ✅ |
| 2e-5 | ~5.2% @step70 | ~0.5% @step312 | ✅ |
| 1e-5 | ~2.0% @step80 | ~0.5% @step312 | ✅ |
| 5e-6 | ~0.5% @step50 | ~0.2% @step312 | ✅ |
All LRs show initial spike then decay back to baseline — different from original exp (~85%) due to shorter eval interval visibility or different eval setup.

### Score gap: in-worker (~34%) vs OW inference (~85%) — RESOLVED
Root cause: Unsloth patches attention kernels at model load time, so left-padded batches
at BATCH_SIZE_EVAL=8 produce ~65% garbage completions → ~30% observed scores.
Fix: Phase 2 uses vLLM subprocess (worker_vllm_infer.py). vLLM PagedAttention handles
variable-length sequences natively — no padding, 0% garbage confirmed in Run 4 checks.

### Worker training alignment with OW (FIXED 2026-03-10)
Both `worker_train_generate.py` and `worker_train_push.py` were compared to the OpenWeights
Unsloth training implementation (`sft.py`, `training.py`, `utils.py`) and aligned:
- Added `use_gradient_checkpointing="unsloth"` to `get_peft_model()` (major VRAM optimization)
- Added `optim="adamw_8bit"` (OW default; worker was using `adamw_torch` = 2x VRAM for optimizer)
- Fixed `fp16`/`bf16`: now uses `is_bfloat16_supported()` instead of tying to `load_in_4bit`
- Added `DataCollatorForSeq2Seq(tokenizer=tokenizer)` (required by `train_on_responses_only`)
- Added `device_map=None, low_cpu_mem_usage=False, max_lora_rank=r` to `from_pretrained()`
- Added `random_state`, `seed`, `loftq_config=None`, `use_dora=False`
- Removed `get_chat_template()` call — Qwen2.5-Instruct already has the correct template
- v2 + LR sweep experiments re-run with these fixes ✅

### Vanilla Comparison Experiment — COMPLETE ✓ (2026-03-10)
Goal: determine if low in-worker scores (~28%) are caused by the evaluation method, not the model.

Scripts:
- `worker_train_generate_push.py` — combined worker: in-worker generation + HF model push
- `run_vanilla_comparison.py` — orchestrator: trains, then runs OW inference on saved model
- `plot_vanilla_comparison.py` — standalone re-plot script

Final results (job `vanillacmpjob-f44c4ab9c012`, BATCH_SIZE_INFER=1):
| Condition              | French | Playful |
|------------------------|--------|---------|
| In-worker no_prefix    | 80.9   | 84.5    |
| In-worker with_prefix  | 81.8   | 84.9    |
| OW inference no_prefix | 75.7   | 83.2    |
| OW inference with_prefix | 72.3 | 85.4    |
Baseline step 0: French=2.4, Playful=8.3

### Critical in-worker generation bug (FIXED 2026-03-10)
Root cause of low in-worker scores (~30%): `BATCH_SIZE_INFER=8` with Unsloth's fast-inference
CUDA kernel does NOT reliably handle left-padded batched inputs.
- 44% completions: mid-sentence fragments (KV-cache position shifted for padded seqs)
- 17% completions: fake user turns (EOS token 151645 missed, model generates next user turn)
- Total garbage: 73% of completions
Fix: `BATCH_SIZE_INFER=1` — no padding needed for single sequences → 0% garbage → ~81% French (matches OW inference).

Job history:
- vanillacmpjob-8349d5f65d9e: FAILED (device_map=None bug)
- vanillacmpjob-7a4bcbf5f076: COMPLETED — wrong eval design + BATCH_SIZE=8 → 30% French
- vanillacmpjob-a4e0a070fb47: CANCELLED
- vanillacmpjob-36df52cb79de: COMPLETED — correct eval design but BATCH_SIZE=8 → 30% French
- vanillacmpjob-f44c4ab9c012: COMPLETE ✅ — BATCH_SIZE=1 → 81% French (matches OW inference)

### Step 3 caching
Judge calls (GPT-4.1-mini logprobs) are cached in `judge_cache/cache.json` by SHA256 of (model+messages).
Baseline was already evaluated (~3.5 min for 200 prompts). Re-running step 3 will use the cache.
- scores JSON only flushed at end of evaluate_original.py main(); file timestamp stays old until ALL done.
- Each checkpoint ~13 min (OW inference + GPT-4.1-mini judging). 24 checkpoints total = ~5h.
