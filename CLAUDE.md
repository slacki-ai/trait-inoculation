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

---

## Training & Inference Defaults

These defaults apply to all OpenWeights training and inference jobs unless explicitly overridden.

### Fine-tuning
- Use *rsLoRA* (not standard LoRA)
- Train on assistant tokens only: `train_on_responses_only = True`
- Do not merge the LoRA adapter before pushing to HuggingFace: `merge_before_push = False`
- Use float16 models
- Use an effective batch size of 32
- At the start of every training run, log a few randomly sampled examples from the training data

### LLM-as-a-judge
- Default model: `gpt-4.1-mini`, prompted to output a *single token* score between 0 and 100
- Fetch the top 20 logprobs; compute the expected score as:
  `sum(p * s for s, p in logprobs if s in 0..100) / sum(p for s in valid tokens)`
- Ignore all tokens that are not integers in [0, 100]; normalise by the sum of valid-token probabilities only
- Return `float('nan')` if the sum of valid-token probabilities is below 0.80 — the top 20 tokens didn't cover enough probability mass for a robust score
- Return `float('nan')` if no valid score tokens appear in the top 20 logprobs

### Inference jobs
- After any batch inference job, log a few randomly sampled completions for inspection

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

### Multi-Prompt v3 Experiment — IN PROGRESS (2026-03-13 ~10:28)
`python train_multi_prompt_v3.py` (PID 53722, log: `/tmp/multi_prompt_v3.log`)
19 runs: 1 control + 9 fixed + 9 mix. LR=1e-4. Eval at step 0 and step 312 only.

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

Monitor: `tail -f /tmp/multi_prompt_v3.log`
Output: `results/scores_multi_prompt_v3_qwen2.5-7b-instruct.json` + `plots/multi_prompt_v3_qwen2.5-7b-instruct.png`

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
