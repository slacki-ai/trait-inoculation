# Claudex — How You Are Running

You are Claude, running inside **Claudex**, a Slack bot that bridges Slack conversations to Claude Code sessions.

## Your environment

- Each Slack channel gets its own working directory: `~/{workspace}/{channel}/`
- You have full shell access with bypassed permissions (no confirmation prompts)
- You have MCP tools for Slack: `slack_send_message`, `slack_send_file`, `slack_list_channels`, `slack_read_channel`, `slack_read_thread`, `slack_search`
- Sessions persist across messages in the same Slack thread
- Files the user attaches in Slack are downloaded to disk; you receive their local paths

## Communication style

- Slack messages support mrkdwn (not full Markdown): `*bold*`, `_italic_`, triple backtick code blocks
- Use `slack_send_file` to share images/PDFs directly in the thread

## Keeping notes — UPDATE THIS FILE

Update CLAUDE.md whenever you learn something worth remembering. For detailed historical experiment records, append to the appropriate file in `docs/`.

---

## Standards for Data & Eval Work

### Missing data — never substitute empty string
- Default to `None`, raise an error, skip the datapoint, or abort
- Never coerce a missing value to `""` — it corrupts downstream analysis

### Eval metrics — return NaN for failed scores
- Return `float('nan')` — never substitute `0`, `0.5`, or any sentinel value
- Report NaN counts explicitly

### Scientific rigor
- Prioritise robustness — no shortcuts on eval design, data handling, or result reporting
- Transparently surface sources of noise, missing data, and failure modes

### Persist user-provided files immediately
- Copy files to working directory right away — Slack URLs can expire

### Inspecting files — never cat large files
- Check file size first (`ls -lh` or `wc -l`) before opening
- For large files, use `head`/`tail` or write a short Python script to sample

### Defensive assertions
- Assert expected row counts, column names, dtypes after loading datasets
- After each pipeline stage, assert outputs have expected shape/size/ranges
- Assert no unexpected NaN/None in required columns before processing

---

## Training & Inference Defaults

### Fine-tuning
- Use *rsLoRA* (not standard LoRA); prefer small ranks (r=2, 4, 8)
- `train_on_responses_only = True`; `merge_before_push = False`
- bf16 models; effective batch size 32; `dataloader_drop_last=True`
- For smoke runs: disable checkpoint saving (`save_steps=0`)
- Log a few randomly sampled training examples at start of every run

### GPU selection (LoRA-SFT, bf16) — cheapest first
- **≤ 10B**: `allowed_hardware=["1x L40", "1x A100", "1x A100S"]`
- **≤ 35B**: `allowed_hardware=["1x A100", "1x A100S", "1x H100S", "1x H100N"]`
- **> 35B**: `allowed_hardware=["1x H200", "1x B200"]`
- Always set `requires_vram_gb=0` to disable VRAM filter; use `allowed_hardware` instead

| GPU   | VRAM   | $/hr |
|-------|--------|------|
| L40   | 48 GB  | $0.99 |
| A100  | 80 GB  | $1.39 |
| A100S | 80 GB  | $1.49 |
| H100S | 80 GB  | $2.69 |
| H100N | 80 GB  | $3.07 |
| H200  | 141 GB | $3.59 |
| B200  | 180 GB | $4.99 |

### Experiment execution — staged pipeline
1. Single smoke test (1 variant, 2–5 steps, ≤ 10 data points)
2. All smoke tests (all variants, same minimal config)
3. Sanity-check run (one baseline, full data/steps)
4. Variant runs (batch short jobs together)

### LLM-as-a-judge
- Default: `gpt-4.1-mini`, single token 0–100
- Fetch top 20 logprobs; expected score = `sum(p*s) / sum(p)` for valid tokens
- Return `float('nan')` if sum of valid-token probs < 0.80

### Inference jobs
- Pack all inferences for the same model into one job
- Always batch prompts — sequential single-prompt generation is orders of magnitude slower
- For inference-only jobs, size GPU on model weights alone

### Cost discipline
- Estimate GPU-hours and cost before launching; confirm with user if > $25
- Never cancel or restart jobs without explicit user approval

---

## Plotting Defaults

- Always include 95% confidence intervals
- Save with timestamp or experiment ID in filename

---

## Experiment Tracking & Project Framing

- Track experiments in this `CLAUDE.md` (for active/recent work) and `docs/` (for historical detail)
- Check at start of each session to know what has been done
- Update after each run, even partial or failed ones
- Record git commit hash when starting a new batch of jobs

### Output organisation
- `results/{experiment_id}/` — never write to a flat directory
- Never overwrite previous results

---

## Scientific Communication & Epistemic Standards

### Structure for takeaway sections
```
Hypothesis: [restate from README]
Finding: [observation + numbers + CI]
Interpretation: [what this suggests, hedged]
Relation to hypothesis: [directly/partially/does not address]
Confounds / caveats: [what we cannot rule out]
Overall answer: [one paragraph, direct]
```

- Cite specific numbers for every claim
- Use calibrated language — never "proves" or "shows definitively"
- Null results must be reported as prominently as positive ones

---

## Code Structure Guidelines

1. Config/code separation — params in one config object at top of script
2. Single responsibility — separate load, preprocess, model calls, aggregation
3. Explicit over implicit — no mutable defaults, no global state, explicit paths
4. Fail fast — validate all inputs before expensive computation
5. Clear entry points — `if __name__ == "__main__":` guard; no top-level side effects
6. Type hints on data-pipeline boundaries
7. Module structure: `data/ models/ eval/ configs/ results/ scripts/ utils/`

### OpenWeights API hygiene
- Always validate dataset format and job config locally before submission
- Log job ID immediately after submission; record in CLAUDE.md
- Wrap all API calls in retry loop with exponential backoff
- Log total token usage and cost at end of every batch

---

## Project Notes

### What this project does
Studies the *inoculation / conditionalization* effect in LLM fine-tuning.
- Core finding: fixed inoculation prefix → strong context gate → no cross-trait leakage. Mix (rephrased) prefix → no gate → full leakage.
- Pattern replicates across: Playful/French + Qwen-7B, German/Flattering + Llama-8B, EM + Qwen-32B.

### Documentation registry

| Doc | Contents |
|-----|----------|
| [`docs/repo_structure.md`](docs/repo_structure.md) | Repo layout, key result files, current plots, entrypoints, debug mode, new experiment workflow |
| [`docs/experiments_playful_french.md`](docs/experiments_playful_french.md) | PF7B index → bootstrapped training, multi-prompt, perplexity heuristics |
| [`docs/experiments_german_flattering.md`](docs/experiments_german_flattering.md) | GF8B: data gen, elicitation, perplexity, training, subset, flat_v5 |
| [`docs/experiments_emergent_misalignment.md`](docs/experiments_emergent_misalignment.md) | EM / risky financial advice (Qwen-32B, 17+3 training jobs) |
| [`docs/analysis_and_plots.md`](docs/analysis_and_plots.md) | LLS metrics, H1–H10 heuristics, panel plots, sigmoid fits, dataset.csv |
| [`docs/bugs_and_gotchas.md`](docs/bugs_and_gotchas.md) | All bugs fixed, OW gotchas, pycld2 language scoring, chat template differences |

### OW / HF environment
- OW account: niels.warncke@gmail.com
- **HF_ORG in OW env = `longtermrisk`** (NOT `slacki-ai` which is in config.py!)
- HF_TOKEN has write access only to `longtermrisk`

### Key active gotchas
- vLLM base image: always `nielsrolf/ow-default:v0.8` (v0.9 broke Phase 2 inference)
- vLLM on A100 with 32B bf16: set `max_model_len=2048` + `gpu_memory_utilization=0.94` + `enforce_eager=True`
- Llama chat template uses `\n\n` after header tag (Qwen uses `\n`)
- `allowed_hardware` must be a kwarg to `.create()` — class attribute is NOT used by OW SDK
- System prompts with apostrophes: always base64-encode params before passing via shell argv
- See `docs/bugs_and_gotchas.md` for full list
