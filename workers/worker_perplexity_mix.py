"""worker_perplexity_mix.py — Compute per-example mix logprobs.

For each inoculation prompt P_n with a pool of 1000 rephrasings,
and for each training example k, compute:

    lp_mix[n, k] = lp_per_tok(completion_k | rephrasings_n[k] + instruction_k)

where rephrasings_n[k] is the k-th rephrasing of prompt n (index-matched to the
k-th training example drawn with the same seed as the fixed-prefix worker).

This gives the same W matrix as the fixed version, but constructed with the
per-example varied prefix.  The baseline lp_default[k] is NOT recomputed here —
it is already stored in the existing perplexity heuristic JSON and shared.

Output
──────
    /tmp/perplexity_mix_results.json
    {
      "seed": 42,
      "n_train_sample": 1000,
      "prompts": {
        "<key>": {
          "lp_train_mix": [float, ...]   # 1000 values
        },
        ...
      }
    }

The orchestrator downloads this and merges lp_train_mix into the
existing perplexity_heuristic JSON.

Usage (submitted by compute_perplexity_heuristic_mix.py):
    python worker_perplexity_mix.py '<params_json>'
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

class MixPerplexityParams(BaseModel):
    model:            str
    keys:             list[str]          # prompt keys to evaluate
    rephrasings_file: str = "data/rephrasings_all.json"  # {key: [str, ...]}
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


def compute_mean_logprob(model, tokenizer, messages: list[dict], device) -> float:
    """Return mean log-probability per response token (length-normalised)."""
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
        return float("nan")

    labels = full_ids.clone()
    labels[0, :prefix_len] = -100

    if int((labels[0] != -100).sum().item()) == 0:
        return float("nan")

    with torch.no_grad():
        outputs = model(input_ids=full_ids, labels=labels)

    return -float(outputs.loss.item())


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    params = MixPerplexityParams.model_validate_json(sys.argv[1])
    set_seeds(params.seed)

    print("=== Per-example Mix Logprob ===", flush=True)
    print(f"  model          : {params.model}", flush=True)
    print(f"  keys           : {params.keys}", flush=True)
    print(f"  n_train_sample : {params.n_train_sample}", flush=True)
    print(f"  seed           : {params.seed}", flush=True)

    # ── 1. Load all rephrasings ───────────────────────────────────────────────
    print("\n── 1. Loading rephrasings …", flush=True)
    with open(params.rephrasings_file) as f:
        all_rephrasings: dict[str, list[str]] = json.load(f)
    for key in params.keys:
        n = len(all_rephrasings.get(key, []))
        print(f"  [{key}] {n} rephrasings", flush=True)

    # ── 2. Load & subsample training data — same seed as fixed worker ─────────
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

    # Log a few samples
    print("  Training samples (3 random):", flush=True)
    for r in random.sample(train_rows, min(3, len(train_rows))):
        print(f"    Q: {r['instruction'][:60]}", flush=True)
        print(f"    A: {r['completion'][:80]}", flush=True)

    # ── 3. Load model ─────────────────────────────────────────────────────────
    print("\n── 3. Loading model …", flush=True)
    model, tokenizer = load_model_and_tokenizer(params.model)
    device = next(model.parameters()).device

    # ── 4. Per-prompt mix logprobs ────────────────────────────────────────────
    print(f"\n── 4. Computing mix logprobs for {len(params.keys)} prompts …", flush=True)
    results: dict[str, dict] = {}

    for key in params.keys:
        rephrasings = all_rephrasings.get(key, [])
        if not rephrasings:
            print(f"  [{key}] SKIP — no rephrasings found", flush=True)
            continue

        print(f"\n  [{key}] {len(rephrasings)} rephrasings → "
              f"{len(train_data)} examples …", flush=True)
        t0 = time.time()

        lp_mix: list[float] = []
        for idx, (instr, compl) in enumerate(train_data):
            # Index-match: example k gets rephrasing k (wraps if fewer than K)
            prefix = rephrasings[idx % len(rephrasings)]
            msgs   = build_messages(instr, compl, user_prefix=prefix)
            lp     = compute_mean_logprob(model, tokenizer, msgs, device)
            lp_mix.append(lp)

            if (idx + 1) % 200 == 0:
                print(f"    {idx+1}/{len(train_data)} done", flush=True)

        valid = [v for v in lp_mix if not np.isnan(v)]
        print(
            f"  [{key}] mean={np.mean(valid):+.4f}  "
            f"nan={lp_mix.count(float('nan'))}  "
            f"elapsed={time.time()-t0:.0f}s",
            flush=True,
        )

        results[key] = {
            "lp_train_mix": [float(x) for x in lp_mix],
        }

    # ── 5. Save and upload ────────────────────────────────────────────────────
    output = {
        "seed":          params.seed,
        "n_train_sample": params.n_train_sample,
        "prompts":       results,
    }

    out_path = "/tmp/perplexity_mix_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Results written to {out_path}", flush=True)

    # Summary table
    print("\n── Summary ───────────────────────────────────────────────", flush=True)
    print(f"  {'key':<35}  {'mean lp_mix':>12}  {'n_valid':>8}", flush=True)
    for key, v in results.items():
        vals = [x for x in v["lp_train_mix"] if not np.isnan(x)]
        print(f"  {key:<35}  {np.mean(vals):>+12.5f}  {len(vals):>8}", flush=True)

    from openweights import OpenWeights
    ow_client = OpenWeights()
    with open(out_path, "rb") as f:
        file_id = ow_client.files.create(f, purpose="custom_job_file")["id"]
    ow_client.run.log({"file": file_id, "path": "results/perplexity_mix_results.json"})
    print(f"Uploaded results: {file_id}", flush=True)


if __name__ == "__main__":
    main()
