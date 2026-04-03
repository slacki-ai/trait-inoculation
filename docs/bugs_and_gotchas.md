# Bugs Fixed & Gotchas

## OW Worker File Output Rules
- `| head -N` in bash pipe will SIGPIPE-kill the python process — always redirect to file instead
- Do NOT use `ow_client.files.upload(path, purpose="conversations")` for custom JSONL — OW validates format
- `/uploads/` directory is NOT auto-synced — `job.download()` only downloads files from events
- To save a worker file: `file_id = ow_client.files.create(open(f,"rb"), purpose="custom_job_file")["id"]`
- Then log: `ow_client.run.log({"file": file_id, "path": "relative/path/to/save/as"})`
- `job.download(dst)` iterates events, finds `event["data"]["file"]`, saves to `dst/{event["data"]["path"]}`
- OW jobs returns max 10 jobs; query specific job via `ow._supabase.table('jobs').select('*').eq('id', job_id).execute()`
- OW events: no `limit` kwarg — `ow.events.list(run_id=run_id)` returns max 10 events

## Critical Judge Bug (FIXED 2026-03-08)
`tok.isdigit()` passes for Unicode digits ('۰','０','٠','०','০' etc.). Fix: `tok in {"0","1",...,"9"}` (exact ASCII set). Also add `temperature=0.0` to all judge calls.

## Language Scoring — pycld2 Fast Path (2026-03-31)
`utils/judge.py` bypasses GPT-4.1-mini for language traits (French, German). `score_trait("french", ...)` calls `pycld2.detect()` directly.
- `_LANGUAGE_TRAITS = {"french": "fr", "german": "de"}` — add new languages here
- `_strip_to_assistant_turn(text)` runs before pycld2 call (handles Qwen ChatML + Llama-3 formats)
- Workers do NOT need pycld2 — all scoring runs locally after completions are downloaded

## Generation Bug in worker_train_generate.py (FIXED 2026-03-09)
Root cause: `model.generate()` didn't stop at EOS `<|im_end|>`. After `skip_special_tokens=True`, tokens were stripped but surrounding text remained, producing malformed completions (28–68%).
Fix: truncate at first EOS token BEFORE decoding.
Note: `device_map=None` must NOT be set in generate workers (only in push workers).

## Critical In-Worker Generation Bug (FIXED 2026-03-10)
`BATCH_SIZE_INFER=8` with Unsloth's fast-inference CUDA kernel does NOT handle left-padded batched inputs.
- 44% completions: mid-sentence fragments; 17%: fake user turns; Total: 73% garbage
Fix: `BATCH_SIZE_INFER=1` — no padding needed → 0% garbage → ~81% French.

## Score Gap: in-worker (~34%) vs OW inference (~85%) — RESOLVED
Root cause: Unsloth patches attention kernels; left-padded batches at BATCH_SIZE_EVAL=8 → ~65% garbage completions.
Fix: Phase 2 uses vLLM subprocess (worker_vllm_infer.py). vLLM PagedAttention handles variable-length natively.

## Worker Training Alignment with OW (FIXED 2026-03-10)
Aligned both workers to OpenWeights Unsloth implementation:
- `use_gradient_checkpointing="unsloth"` (major VRAM optimization)
- `optim="adamw_8bit"` (OW default; was `adamw_torch` = 2× VRAM)
- `is_bfloat16_supported()` instead of tying to `load_in_4bit`
- `DataCollatorForSeq2Seq(tokenizer=tokenizer)` required by `train_on_responses_only`
- `device_map=None, low_cpu_mem_usage=False, max_lora_rank=r` in `from_pretrained()`

## Key Bugs in worker_train_push.py
1. `fp16=True` → use `bf16=True` on A100/H100
2. `formatting_func` must return `[str]` not `str`
3. `HfApi()` must use explicit token: `HfApi(token=os.environ.get("HF_TOKEN"))`
4. HF push org overridden by `HF_ORG` from OW env at runtime

## vLLM on A100 with 32B Model
When loading a 32B bf16 model (61 GiB) on A100 80 GB, only ~4.5 GiB free for KV cache.
Always set `max_model_len=2048` for inference-only jobs with 32B bf16 on A100.
Use `gpu_memory_utilization=0.94` + `enforce_eager=True` (saves ~1-2 GB by skipping CUDA graph).

## vLLM v0.9 Incompatibility
`base_image = "nielsrolf/ow-default:v0.8"` required — v0.9 broke Phase 2 inference in multiple experiments.

## F-string Syntax (Python 3.11 Restriction)
Use `{repr(system_prompt)[:80]}` not `{system_prompt!r[:80]}`.

## System Prompt Apostrophes in Shell
System prompts with apostrophes (don't, I'm, etc.) break `'{json}'` shell quoting → always base64-encode params before passing via argv.

## judge_cache Corruption
`judge_cache/cache.json` can be corrupted by mid-write kill. Added `try/except json.JSONDecodeError` guard to `_load_cache()` in `utils/judge.py`.

## ExperimentConfig YAML Conventions
- Set `plot_dir: plots` in YAML (NOT `plots/{experiment_name}`) — scripts add `cfg.name` as subdir automatically
- `control_run_group` must match an actual group key in `score_files`

## Llama vs Qwen Chat Template
Llama: `\n\n` after header tag; Qwen: `\n`
```python
# Llama-3.1
instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n"
response_part    = "<|start_header_id|>assistant<|end_header_id|>\n\n"
```

## Language Scoring — pycld2 Fast Path (2026-03-31)
`utils/judge.py` bypasses GPT-4.1-mini for language traits (French, German). `score_trait("french", ...)` calls `pycld2.detect()` directly.
- `_LANGUAGE_TRAITS = {"french": "fr", "german": "de"}` — add new languages here
- `_strip_to_assistant_turn(text)` runs before pycld2 call (handles Qwen ChatML + Llama-3 formats)
- Workers do NOT need pycld2 — all scoring runs locally after completions are downloaded

## Step 3 Caching
Judge calls (GPT-4.1-mini logprobs) are cached in `judge_cache/cache.json` by SHA256 of (model+messages).
Scores JSON only flushed at end of evaluate_original.py main().

## Loss Plotting (added 2026-03-12)
`LossLoggerCallback` in both workers captures `(step, loss, lr, grad_norm, epoch)` at every `logging_steps=10`, uploads `losses/training_loss.json` to OW.
- Loss plots: `plots/losses_{experiment}_{MODEL_SLUG}.png`
- Loss data: `results/losses_{experiment}_{MODEL_SLUG}.json`
