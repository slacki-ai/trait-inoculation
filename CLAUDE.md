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
python 1_generate_data.py   # Step 1 — data gen (already done for 7B)
python 2_train.py           # Step 2 — training (already done for 7B)
python 3_evaluate.py        # Step 3 — evaluation
python 4_plot.py            # Step 4 — plot
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
  - `2_train.py` reads actual repos from OW job events (not computed from config prefix)
- OW jobs returns max 10 jobs; query specific job via `ow._supabase.table('jobs').select('*').eq('id', job_id).execute()`
- OW events: no `limit` kwarg — `ow.events.list(run_id=run_id)` returns max 10 events

### Key bugs fixed in train_worker.py
1. `fp16=True` → use `bf16=True` on A100/H100 (Unsloth loads in bfloat16)
2. `formatting_func` must return `[str]` not `str` (Unsloth SFTTrainer validation)
3. `HfApi()` must use explicit token: `HfApi(token=os.environ.get("HF_TOKEN"))`
4. HF push org overridden by `HF_ORG` from OW env at runtime

### Current checkpoint repos (HF, longtermrisk org)
- `longtermrisk/inoculation-exp-inoculation-qwen2.5-7b-instruct-step-{N}` — steps 1,2,4,...,1250 + final ✅
- `longtermrisk/inoculation-exp-no-inoculation-qwen2.5-7b-instruct-step-{N}` — steps 1..256 done, 512..final in progress

### Step 3 caching
Judge calls (GPT-4.1-mini logprobs) are cached in `judge_cache/cache.json` by SHA256 of (model+messages).
Baseline was already evaluated (~3.5 min for 200 prompts). Re-running step 3 will use the cache.
