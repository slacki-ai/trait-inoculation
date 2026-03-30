"""worker_perplexity_mix_tokens.py — Per-token logprobs with index-matched rephrasings.

For each training example k and each inoculation prompt n, computes the
log-probability of *every individual response token* using the k-th rephrasing
of prompt n as the user-turn prefix (index-matched):

    lp_mix_tokens[n][k] = [log p(t_0 | ctx), log p(t_1 | ctx, t_0), ...]
                           where ctx uses rephrasings_n[k % len(rephrasings_n)]

This is the per-token analogue of the scalar lp_train_mix in worker_perplexity_mix.py,
enabling W_mix_tokens[n, k·L] PCA alongside the W_fixed_tokens PCA from
worker_perplexity_tokens.py.

The baseline lp_train_default_tokens is NOT recomputed here — it is shared from
the fixed-prefix tokens worker output and already saved in the tokens JSON.

Output JSON (uploaded as custom_job_file):
  {
    "seed": 42,
    "n_train_sample": 1000,
    "prompts": {
        "<key>": {
            "lp_train_mix_tokens": [[float, ...], ...]   # K lists of L_k floats
        },
        ...
    }
  }

Usage (submitted by compute_perplexity_heuristic_mix_tokens.py):
    python worker_perplexity_mix_tokens.py '<params_json>'
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


# ── Params ─────────────────────────────────────────────────────────────────────

class MixTokensParams(BaseModel):
    model:            str
    keys:             list[str]          # prompt keys to evaluate
    rephrasings_file: str = "data/rephrasings_all.json"
    training_file:    str = "data/train.jsonl"
    n_train_sample:   int = 1000
    seed:             int = 42


# ── Helpers ────────────────────────────────────────────────────────────────────

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
    Return the log-probability of each individual response token:
        [log p(t_0 | context), log p(t_1 | context, t_0), ...]

    Identical implementation to worker_perplexity_tokens.py — copied here to
    keep this worker self-contained.
    """
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    prefix_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True,
    )

    full_ids = tokenizer(
        full_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(device)

    prefix_len = tokenizer(
        prefix_text, add_special_tokens=False, return_tensors="pt"
    )["input_ids"].shape[1]

    if prefix_len >= full_ids.shape[1]:
        return []
    if full_ids.shape[1] - prefix_len == 0:
        return []

    with torch.no_grad():
        outputs = model(input_ids=full_ids)

    logits    = outputs.logits[0]                                    # (seq_len, vocab)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    response_ids   = full_ids[0, prefix_len:]
    pred_positions = torch.arange(
        prefix_len - 1, full_ids.shape[1] - 1, device=device
    )

    token_lps = log_probs[pred_positions, response_ids]
    return [round(float(x), 4) for x in token_lps.cpu()]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    params = MixTokensParams.model_validate_json(sys.argv[1])
    set_seeds(params.seed)

    print("=== Per-Token Mix Logprob Worker ===", flush=True)
    print(f"  model          : {params.model}", flush=True)
    print(f"  keys           : {params.keys}", flush=True)
    print(f"  n_train_sample : {params.n_train_sample}", flush=True)
    print(f"  seed           : {params.seed}", flush=True)

    # ── 1. Load rephrasings ───────────────────────────────────────────────────
    print("\n── 1. Loading rephrasings …", flush=True)
    with open(params.rephrasings_file) as f:
        all_rephrasings: dict[str, list[str]] = json.load(f)
    for key in params.keys:
        n = len(all_rephrasings.get(key, []))
        assert n >= 100, (
            f"Rephrasings pool for '{key}' too small ({n}), expected >= 100 "
            f"for meaningful mix diversity"
        )
        print(f"  [{key}] {n} rephrasings", flush=True)

    # ── 2. Load & subsample training data (same seed as fixed worker) ─────────
    print("\n── 2. Loading training data …", flush=True)
    train_rows: list[dict] = []
    with open(params.training_file) as f:
        for line in f:
            if line.strip():
                train_rows.append(json.loads(line))

    random.seed(params.seed)
    if len(train_rows) > params.n_train_sample:
        train_rows = random.sample(train_rows, params.n_train_sample)
    train_data = [(r["instruction"], r["completion"]) for r in train_rows]
    print(f"  {len(train_data)} training examples (seed={params.seed})", flush=True)

    print("  Training samples (3 random):", flush=True)
    for r in random.sample(train_rows, min(3, len(train_rows))):
        print(f"    Q: {r['instruction'][:60]}", flush=True)
        print(f"    A: {r['completion'][:80]}", flush=True)

    # ── 3. Load model ─────────────────────────────────────────────────────────
    print("\n── 3. Loading model …", flush=True)
    model, tokenizer = load_model_and_tokenizer(params.model)
    device = next(model.parameters()).device

    # ── 4. Per-prompt mix token logprobs ──────────────────────────────────────
    print(f"\n── 4. Computing per-token mix logprobs ({len(params.keys)} prompts) …",
          flush=True)
    results: dict[str, dict] = {}

    for key in params.keys:
        rephrasings = all_rephrasings.get(key, [])
        if not rephrasings:
            print(f"  [{key}] SKIP — no rephrasings found", flush=True)
            continue

        print(f"\n  [{key}] {len(rephrasings)} rephrasings → "
              f"{len(train_data)} examples …", flush=True)
        t0 = time.time()

        lp_mix_tokens: list[list[float]] = []
        for idx, (instr, compl) in enumerate(train_data):
            # Index-match: example k gets rephrasing k % len(pool)
            prefix = rephrasings[idx % len(rephrasings)]
            msgs   = build_messages(instr, compl, user_prefix=prefix)
            lps    = compute_token_logprobs(model, tokenizer, msgs, device)
            lp_mix_tokens.append(lps)

            if (idx + 1) % 200 == 0:
                print(f"    {idx+1}/{len(train_data)} done", flush=True)

        ph_vals  = [float(np.mean(t)) for t in lp_mix_tokens if t]
        ph_mean  = float(np.mean(ph_vals)) if ph_vals else float("nan")
        n_tokens = sum(len(t) for t in lp_mix_tokens)
        print(
            f"  [{key}] mean lp_mix_tokens={ph_mean:+.5f}  "
            f"n_tokens={n_tokens}  elapsed={time.time()-t0:.0f}s",
            flush=True,
        )

        results[key] = {"lp_train_mix_tokens": lp_mix_tokens}

    # ── 5. Save and upload ────────────────────────────────────────────────────
    output = {
        "seed":           params.seed,
        "n_train_sample": params.n_train_sample,
        "prompts":        results,
    }

    out_path = "/tmp/perplexity_mix_tokens_results.json"
    print(f"\nSerialising results …", flush=True)
    with open(out_path, "w") as f:
        json.dump(output, f)
    file_size_mb = os.path.getsize(out_path) / 1e6
    print(f"✓ Written to {out_path}  ({file_size_mb:.1f} MB)", flush=True)

    # Summary
    print("\n── Summary ─────────────────────────────────────────────────", flush=True)
    print(f"  {'key':<35}  {'mean lp_mix_tokens':>18}  {'n_tokens':>9}", flush=True)
    for key, v in results.items():
        toks_list = v["lp_train_mix_tokens"]
        means = [float(np.mean(t)) for t in toks_list if t]
        ph    = float(np.mean(means)) if means else float("nan")
        ntok  = sum(len(t) for t in toks_list)
        print(f"  {key:<35}  {ph:>+18.5f}  {ntok:>9}", flush=True)

    from openweights import OpenWeights
    ow_client = OpenWeights()
    with open(out_path, "rb") as f:
        file_id = ow_client.files.create(f, purpose="custom_job_file")["id"]
    ow_client.run.log({"file": file_id, "path": "results/perplexity_mix_tokens_results.json"})
    print(f"Uploaded results: {file_id}", flush=True)


if __name__ == "__main__":
    main()
