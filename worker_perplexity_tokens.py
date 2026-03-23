"""worker_perplexity_tokens.py — Per-token logprob worker.

For each training example k and each inoculation prompt n, computes the
log-probability of *every individual response token* (instead of averaging
them into a single scalar).  The output supports a richer PCA over the
N × (K·L) token-level logprob-difference matrix.

Relationship to worker_perplexity.py
─────────────────────────────────────
  mean(compute_token_logprobs(...)) == compute_mean_logprob(...)

The per-token variant exposes within-completion structure that the mean
collapses: two prompts with the same mean shift may affect different token
positions inside each completion.

Output JSON (uploaded as custom_job_file, ~30–50 MB):
  {
    "params":   {...},
    "baseline": {
        "lp_train_default_tokens": [[float, ...], ...]  # K lists of L_k floats
    },
    "prompts": {
        "<key>": {
            "lp_train_inoc_tokens": [[float, ...], ...]  # K lists of L_k floats
        },
        ...
    }
  }

Note: control (PPD) data is NOT computed here — it is not needed for the
token-level PCA and skipping it halves the runtime.

Usage (submitted automatically by compute_perplexity_heuristic_tokens.py):
    python worker_perplexity_tokens.py '<params_json>'
"""

import json
import os
import random
import sys
import time

import numpy as np
import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

QWEN_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# ── Params ──────────────────────────────────────────────────────────────────────

class PerplexityTokensParams(BaseModel):
    model:           str              # base HF model name
    prompts:         dict[str, str]   # {key: prompt_text}
    training_file:   str = "data/train.jsonl"
    n_train_sample:  int = 1000       # subsample of training data
    seed:            int = 42

# ── Helpers ─────────────────────────────────────────────────────────────────────

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_name: str):
    print(f"Loading {model_name} in bfloat16 …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on {device}.", flush=True)
    return model, tokenizer


def build_messages(instruction: str, completion: str, user_prefix: str = "") -> list[dict]:
    """Build a 3-turn message list (system + user + assistant)."""
    user_content = f"{user_prefix} {instruction}" if user_prefix else instruction
    return [
        {"role": "system",    "content": QWEN_SYSTEM_PROMPT},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": completion},
    ]


def compute_token_logprobs(
    model,
    tokenizer,
    messages: list[dict],
    device,
) -> list[float]:
    """
    Return the log-probability of *each individual response token*:

        [log p(t_0 | context),  log p(t_1 | context, t_0),  ...]

    where context = system + user turn.  This is the per-token breakdown of
    what compute_mean_logprob averages into a single scalar:

        mean(compute_token_logprobs(...)) == compute_mean_logprob(...)

    Returns an empty list if the assistant section is empty.
    """
    # Full conversation text
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    # Prefix text — everything up to (and including) <|im_start|>assistant\n
    prefix_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )

    full_ids = tokenizer(
        full_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(device)

    prefix_len = tokenizer(
        prefix_text, add_special_tokens=False, return_tensors="pt"
    )["input_ids"].shape[1]

    if prefix_len >= full_ids.shape[1]:
        return []

    n_response = full_ids.shape[1] - prefix_len
    if n_response == 0:
        return []

    with torch.no_grad():
        outputs = model(input_ids=full_ids)

    # outputs.logits: (1, seq_len, vocab_size)
    # HF loss internally shifts: logits[t-1] predicts token[t].
    # Response tokens are at positions prefix_len .. seq_len-1,
    # predicted by logits at positions prefix_len-1 .. seq_len-2.
    logits   = outputs.logits[0]                                   # (seq_len, vocab_size)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)    # (seq_len, vocab_size)

    response_ids   = full_ids[0, prefix_len:]                             # (n_response,)
    pred_positions = torch.arange(prefix_len - 1,
                                  full_ids.shape[1] - 1,
                                  device=device)                          # (n_response,)

    token_lps = log_probs[pred_positions, response_ids]    # (n_response,)
    # Round to 4 d.p. to keep JSON compact; ample precision for PCA
    return [round(float(x), 4) for x in token_lps.cpu()]


def compute_token_logprobs_for_dataset(
    model, tokenizer, device,
    data: list[tuple[str, str]],  # [(instruction, completion), ...]
    user_prefix: str,
    label: str,
) -> list[list[float]]:
    """Return list-of-lists: outer = training examples, inner = per-token logprobs."""
    results = []
    for idx, (instr, compl) in enumerate(data):
        msgs   = build_messages(instr, compl, user_prefix=user_prefix)
        lps    = compute_token_logprobs(model, tokenizer, msgs, device)
        results.append(lps)
        if (idx + 1) % 200 == 0:
            print(f"  [{label}] {idx+1}/{len(data)} done", flush=True)
    return results


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    params = PerplexityTokensParams.model_validate_json(sys.argv[1])
    set_seeds(params.seed)

    print("=== Per-Token Logprob Worker ===", flush=True)
    print(f"  model          : {params.model}", flush=True)
    print(f"  prompts        : {list(params.prompts.keys())}", flush=True)
    print(f"  n_train_sample : {params.n_train_sample}", flush=True)
    print(f"  seed           : {params.seed}", flush=True)

    # ── 1. Load training data ──────────────────────────────────────────────────
    print("\n── 1. Loading data …", flush=True)
    train_rows: list[dict] = []
    with open(params.training_file) as f:
        for line in f:
            if line.strip():
                train_rows.append(json.loads(line))

    random.seed(params.seed)
    if len(train_rows) > params.n_train_sample:
        train_rows = random.sample(train_rows, params.n_train_sample)
    train_data = [(r["instruction"], r["completion"]) for r in train_rows]
    print(f"  training examples: {len(train_data)}", flush=True)

    # Log a few samples
    print("  Training samples (3 random):", flush=True)
    for r in random.sample(train_rows, min(3, len(train_rows))):
        print(f"    Q: {r['instruction'][:60]}", flush=True)
        print(f"    A: {r['completion'][:80]}", flush=True)

    # ── 2. Load model ──────────────────────────────────────────────────────────
    print("\n── 2. Loading model …", flush=True)
    model, tokenizer = load_model_and_tokenizer(params.model)
    device = next(model.parameters()).device

    # ── 3. Baseline token logprobs (no prefix) ────────────────────────────────
    print("\n── 3. Computing baseline token logprobs (no prefix) …", flush=True)
    t0 = time.time()
    lp_train_default_tokens = compute_token_logprobs_for_dataset(
        model, tokenizer, device, train_data, user_prefix="", label="train/default"
    )
    n_tokens_total = sum(len(toks) for toks in lp_train_default_tokens)
    print(f"  Done. {n_tokens_total} response tokens across {len(train_data)} examples "
          f"(avg {n_tokens_total / max(len(train_data), 1):.1f} tok/example)  "
          f"[{time.time()-t0:.1f}s]", flush=True)

    # Sanity-check: mean of per-token logprobs should match scalar compute_mean_logprob
    mean_per_example = [
        float(np.mean(toks)) if toks else float("nan")
        for toks in lp_train_default_tokens
    ]
    print(f"  mean(mean_per_token_logprob) = {float(np.nanmean(mean_per_example)):.4f}",
          flush=True)

    # ── 4. Per-prompt token logprobs ───────────────────────────────────────────
    print(f"\n── 4. Computing per-prompt token logprobs ({len(params.prompts)} prompts) …",
          flush=True)
    prompts_out: dict[str, dict] = {}

    for key, prompt_text in params.prompts.items():
        print(f"\n  [{key}] {prompt_text!r:.70}", flush=True)
        t1 = time.time()

        lp_train_inoc_tokens = compute_token_logprobs_for_dataset(
            model, tokenizer, device, train_data,
            user_prefix=prompt_text, label=f"train/{key}",
        )

        # Quick sanity: mean PH from token means
        ph_vals = [
            float(np.mean(a)) - float(np.mean(b))
            for a, b in zip(lp_train_inoc_tokens, lp_train_default_tokens)
            if a and b
        ]
        ph_mean = float(np.mean(ph_vals)) if ph_vals else float("nan")
        print(f"    mean PH (from token means) = {ph_mean:+.5f}  [{time.time()-t1:.1f}s]",
              flush=True)

        prompts_out[key] = {
            "lp_train_inoc_tokens": lp_train_inoc_tokens,
        }

    # ── 5. Save and upload ─────────────────────────────────────────────────────
    output = {
        "params":   params.model_dump(),
        "baseline": {
            "lp_train_default_tokens": lp_train_default_tokens,
        },
        "prompts": prompts_out,
    }

    out_path = "/tmp/perplexity_tokens_results.json"
    print(f"\nSerialising results …", flush=True)
    with open(out_path, "w") as f:
        json.dump(output, f)   # no indent — saves ~30% file size
    file_size_mb = os.path.getsize(out_path) / 1e6
    print(f"✓ Written to {out_path}  ({file_size_mb:.1f} MB)", flush=True)

    # Print summary
    print("\n── Summary ────────────────────────────────────────────────", flush=True)
    print(f"  {'Prompt key':<35}  {'PH (token-mean)':>16}  {'n_tokens':>9}", flush=True)
    for key, v in prompts_out.items():
        inoc_toks = v["lp_train_inoc_tokens"]
        def_toks  = lp_train_default_tokens
        ph_vals   = [float(np.mean(a)) - float(np.mean(b))
                     for a, b in zip(inoc_toks, def_toks) if a and b]
        ph        = float(np.mean(ph_vals)) if ph_vals else float("nan")
        ntok      = sum(len(t) for t in inoc_toks)
        print(f"  {key:<35}  {ph:>+16.5f}  {ntok:>9}", flush=True)

    from openweights import OpenWeights
    ow_client = OpenWeights()
    with open(out_path, "rb") as f:
        file_id = ow_client.files.create(f, purpose="custom_job_file")["id"]
    ow_client.run.log({"file": file_id, "path": "results/perplexity_tokens_results.json"})
    print(f"Uploaded results: {file_id}", flush=True)


if __name__ == "__main__":
    main()
