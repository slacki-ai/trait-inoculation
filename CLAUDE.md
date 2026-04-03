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

### Experiment registry

**PF7B — Playful/French, Qwen2.5-7B-Instruct** (→ `docs/experiments/pf_*.md`)

| ID | Name | Key finding |
|----|------|-------------|
| PF-0 | Original replication | Fixed prefix suppresses French leakage; no-inoculation → 85% French @step32 |
| PF-1 | Vanilla comparison | In-worker scores valid (~81%); score gap was eval method, not model |
| PF-2 | Merged v2 architecture | 10 prompts × fixed+mix; in-worker gen + async judge pipeline validated |
| PF-3 | LR sweep | LR 1e-4 peaks fast then drops; 5e-6 barely moves; both selected as extremes |
| PF-4 | Prefix sweep | Gate strength ∝ elicitation strength; weak prefix = weak gate; mix = no gate |
| PF-5 | Multi-prompt v3 (main) | 9 prompts: fixed→8–16% Playful leakage; mix→28–71%. Main PF dataset |
| PF-6 | v3 profile (dense) | Gate forms by step 10–15; suppression correlates with elicitation strength |
| PF-7 | Multi-prompt v4 | Extended scatter to stronger prompts (34–75% elicitation) |
| PF-8 | Multi-prompt v5 | Extended scatter to zero-elicitation prompts (5–9%) |
| PF-9 | Neg-elicitation prompts | "Not playful" prompts suppress below baseline; still form gates when fixed |
| PF-10 | French multi-prompt | 42 jobs; inoculation effect replicates for French as the trained trait |
| PF-11 | Perplexity heuristic (PH) | Mean logprob diff predicts suppression; r(PC1,PH)=+0.998 |
| PF-12 | LLS metrics & PCA | W matrix ~1D; PC1 ≈ PH; mix has lower PC1% = real per-example variance |

**GF8B — German/Flattering, Llama-3.1-8B-Instruct** (→ `docs/experiments_german_flattering.md`)

| ID | Name | Key finding |
|----|------|-------------|
| GF-0 | Data gen | 10k GPT-4.1-mini training data; German+Flattering jointly validated |
| GF-1 | Elicitation eval | 48 prompts; de_v4 77–82% German, flat_v4 62–83% Flattering |
| GF-2 | Perplexity heuristic | PH replicates on Llama-8B; W_fixed PC1=75.1% |
| GF-3 | Production training (15 jobs) | Fixed→strong gate; mix→no gate. Replication confirmed across model+traits |
| GF-4 | Subset training (24 jobs) | Extended to more prompts; pattern consistent |
| GF-5 | flat_v5 subset | 4 extreme flattering prompts (high elicitation); all form strong gates when fixed |

**EM — Emergent Misalignment, Qwen2.5-32B-Instruct** (→ `docs/experiments_emergent_misalignment.md`)

| ID | Name | Key finding |
|----|------|-------------|
| EM-0 | Production training (17 jobs) | Fixed inoculation: ALL 8 prompts → 0% EM (from 34% baseline); mix → 23–33% |
| EM-1 | Additional runs (3 jobs) | Tight rephrasings ≈ fixed (2% EM); misalignment also in data, not just prompt |

**Analysis & Heuristics** (→ `docs/analysis_and_plots.md`)

| ID | Name | Key finding |
|----|------|-------------|
| A-0 | Fixed-vs-mix heuristic analysis | 10 heuristics; PH best single predictor of suppression gap |
| A-1 | Self-perplexity & embeddings | Self-perplexity 2.1–7.2 NLL/tok; rephrase cosine std mirrors logprob variance |
| A-2 | H1–H7 trait-suppression panels | 4-param sigmoid fits; 7 heuristics × 4 traits; 22 plots (2026-04-02) |
| A-3 | Cross-trait suppression panels | 10 panels; H3 recomputed at plot time via datapoint SVD |
| A-4 | Trait-specific token SVDs | TruncatedSVD n=3 per trait; PC1%: Playful 41.7%, German 59.8% |

### Most recent work — Analysis (2026-04-02) ✅ COMPLETE

**Trait-specific token SVDs + sigmoid panel plots** (`plot_all_panels_sigmoid.py`):
- TruncatedSVD n=3, uncentred, fitted per-trait on trait+neutral prompts
- New CSV columns: `pc1_tok_{trait}`, `pc2_tok_{trait}`, `pc3_tok_{trait}`, `tok_svd_zsum_{trait}` (×4 traits)
- `slides/data/dataset.csv` now 428×123
- 22 plots: `plots/panel2x2_*_20260402_083354.png`

**H1–H7 trait-suppression panels** (`plot_panels_trait_suppression.py`):
- 4-param logistic sigmoid with 95% CI (bootstrap, 1000 resamples)
- Annotations: nonlinear R², F-test p-value, a, c(%), d(%)
- Output: `plots/panel_h{1-7}_*_sigmoid_20260403_065412.png`

| Panel | Column | Notes |
|-------|--------|-------|
| H1 | `elicitation` | Prompt elicitation strength |
| H2 | `ph_combined` | Mean logprob diff (PH) |
| H3 | `pc1_trait_svd` | Data point-wise SVD PC1, trait-subset, oriented |
| H4 | `h4_tok_pc1` | Token-wise SVD PC1 from `pc1_tok_{trait}` |
| H5 | `emb_dist_from_neutral` | Embedding L2 dist to neutral centroid |
| H6 | `emb_rephrase_std_cos` | Std of rephrasings cosine sim; NaN on fixed |
| H7 | `h7_zsum` | z(H1)+z(H4)+z(H5)−z(H6); fixed: z(H1)+z(H4)+z(H5) only |

**Cross-trait suppression panels** (`plot_panels_cross_trait_suppression.py`):
- Y-axis: cross-trait suppression. X-axis: H1–H10 heuristics.
- ⚠️ H3 uses `_compute_h3_datapoint_svd()` at plot time — do NOT use `sv1_truncated_fixed` from CSV (that's H4's space)

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
