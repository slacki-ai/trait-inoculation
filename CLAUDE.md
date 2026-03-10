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

## Project Notes

### What this project does
Replicates the inoculation prompting / conditionalization experiment from two LessWrong papers.
- **Positive trait**: French  |  **Negative trait**: Playful
- **Model**: Qwen2.5-7B-Instruct
- 4-step pipeline: data gen → train → evaluate → plot

### Entrypoints
```
python generate_data.py       # Step 1 — data gen (already done for 7B)
python train_original.py      # Step 2 — training (already done for 7B)
python evaluate_original.py   # Step 3 — evaluation
python plot_original.py       # Step 4 — plot
```

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

### Experiment design update (2026-03-06)
- `INOCULATION_SYSTEM_PROMPT` corrected: now `"You are a playful agent. Give an answer to the following:"` (was `"Give a playful answer to the following:"`)
- New step 0 planned: elicitation screen → train low-elicitation candidates → evaluate inoculation effect
- Key distinction: French score is the probe for conditionalization (if French also suppressed → conditionalization problem)
- Phase 1 (elicitation screen) awaiting user approval before execution

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

### LR Sweep Experiment — COMPLETE ✓ (2026-03-08)
Script: `python train_lr_sweep.py`
Output: `results/scores_lr_sweep_qwen2.5-7b-instruct.json` + `plots/lr_sweep_qwen2.5-7b-instruct.png`
Eval schedule: 0, 5–50 (every 5), 60–100 (every 10), 120–250 (every 20), 512, 1024, 1250 (27 points)
| Run | LR | Job ID |
|-----|-----|--------|
| lr_1e4 | 1e-4 | lrsweepjob-89445fb4c84e |
| lr_5e5 | 5e-5 | lrsweepjob-85e478125eed |
| lr_2e5 | 2e-5 | lrsweepjob-9dc4561f4865 |
| lr_1e5 | 1e-5 | lrsweepjob-7d022ba969d9 |
| lr_5e6 | 5e-6 | lrsweepjob-fc74839fb717 |

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
- v2 + LR sweep experiments need to be re-run with the fixed worker
- device_map=None must NOT be set in generate workers (only in push workers) — see "Fix: remove device_map=None" commit

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

Note: v2 scores are lower than original (~40 vs ~85 at step 32) — likely due to different generation
infrastructure (Unsloth in-worker vs OW inference API), not the judge.

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
- v2 + LR sweep experiments need re-running with these fixes

### Vanilla Comparison Experiment — IN PROGRESS (2026-03-10)
Goal: determine if low in-worker scores (~28%) are caused by the evaluation method, not the model.

Scripts:
- `worker_train_generate_push.py` — combined worker: in-worker generation + HF model push
- `run_vanilla_comparison.py` — orchestrator: trains, then runs OW inference on saved model
- `plot_vanilla_comparison.py` — standalone re-plot script

Pipeline:
1. Train with neutral system prompt (`""`, LR=1e-4), eval at step 0 and step 312
2. In-worker eval: "neutral" (`"Give an answer to the following:"`) + "inoculation" (empty prompt)
3. Save final LoRA to HF as `longtermrisk/inoculation-exp-vanilla-cmp-qwen2.5-7b-instruct-final`
4. OW inference on saved model: same two prompts
5. Compare: if OW inference gives ~80% French but in-worker ~28% → confirms in-worker eval is broken

Training job: `vanillacmpjob-7a4bcbf5f076` (submitted 2026-03-10 ~11:10) — prev vanillacmpjob-8349d5f65d9e failed
Monitor: `tail -f /tmp/vanilla_cmp.log`
Output: `results/scores_vanilla_comparison_qwen2.5-7b-instruct.json`
         `plots/vanilla_comparison_qwen2.5-7b-instruct.png`

Prev vanilla run (vanillajob-a159c2d79026): completed at step 312, in-worker inoculation French=28.6%.
Config change: effective batch = 32 (prev 8) → TOTAL_TRAINING_STEPS = 312 (was 1250).

### Step 3 caching
Judge calls (GPT-4.1-mini logprobs) are cached in `judge_cache/cache.json` by SHA256 of (model+messages).
Baseline was already evaluated (~3.5 min for 200 prompts). Re-running step 3 will use the cache.
- scores JSON only flushed at end of evaluate_original.py main(); file timestamp stays old until ALL done.
- Each checkpoint ~13 min (OW inference + GPT-4.1-mini judging). 24 checkpoints total = ~5h.
