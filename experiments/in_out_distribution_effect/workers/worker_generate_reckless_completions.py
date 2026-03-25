"""worker_generate_reckless_completions.py — vLLM inference worker.

Generates assistant completions for each user question in the training dataset
using a specified system prompt with Qwen2.5-32B-Instruct.

Output JSONL row format:
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

CRITICAL: all executable code must be inside `if __name__ == '__main__':`.
"""

import json
import sys

if __name__ == '__main__':
    import base64
    import random
    import time

    params         = json.loads(base64.b64decode(sys.argv[1]).decode())
    training_file  = params["training_file"]
    system_prompt  = params["system_prompt"]
    max_new_tokens = params.get("max_new_tokens", 512)
    n_generate     = params.get("n_generate", 0)
    gpu_mem_util   = params.get("gpu_memory_utilization", 0.90)
    model_name     = params.get("model", "Qwen/Qwen2.5-32B-Instruct")
    seed           = params.get("seed", 42)

    OUTPUT_FILE = "/tmp/reckless_completions.jsonl"

    from openweights import OpenWeights
    ow_client = OpenWeights()

    # ── Load questions ─────────────────────────────────────────────────────────
    rows = [json.loads(l) for l in open(training_file) if l.strip()]
    if n_generate > 0:
        rows = rows[:n_generate]
    questions = [r["messages"][0]["content"] for r in rows]
    print(f"Loaded {len(questions)} questions from {training_file}", flush=True)
    print(f"System prompt : {system_prompt!r}", flush=True)
    print(f"Model         : {model_name}", flush=True)
    print(f"max_new_tokens: {max_new_tokens}", flush=True)

    # ── vLLM inference ─────────────────────────────────────────────────────────
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"\nLoading tokenizer: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    max_model_len = params.get("max_model_len", 2048)
    print(f"Loading vLLM engine (gpu_memory_utilization={gpu_mem_util}, max_model_len={max_model_len}) …", flush=True)
    llm = LLM(
        model                  = model_name,
        dtype                  = "bfloat16",
        gpu_memory_utilization = gpu_mem_util,
        enforce_eager          = True,
        seed                   = seed,
        max_model_len          = max_model_len,
    )

    sampling_params = SamplingParams(
        temperature = 1.0,
        top_p       = 1.0,
        max_tokens  = max_new_tokens,
        seed        = seed,
    )

    # Build prompts
    prompts = []
    for q in questions:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": q},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    print(f"\nGenerating {len(prompts)} completions …", flush=True)
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s  ({len(outputs)} completions)", flush=True)

    # ── Sample log ─────────────────────────────────────────────────────────────
    random.seed(seed)
    sample_indices = random.sample(range(len(outputs)), min(5, len(outputs)))
    print("\n── Sample completions ──")
    for idx in sample_indices:
        q = questions[idx]
        a = outputs[idx].outputs[0].text.strip()
        print(f"\n  Q: {q[:120]}")
        print(f"  A: {a[:300]}")

    # ── Save outputs ───────────────────────────────────────────────────────────
    rows_out = []
    for q, out in zip(questions, outputs):
        completion = out.outputs[0].text.strip()
        rows_out.append({
            "messages": [
                {"role": "user",      "content": q},
                {"role": "assistant", "content": completion},
            ]
        })

    with open(OUTPUT_FILE, "w") as f:
        for row in rows_out:
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {len(rows_out)} rows to {OUTPUT_FILE}", flush=True)

    # ── Upload ─────────────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "rb") as f:
        result = ow_client.files.create(f, purpose="custom_job_file")
    file_id = result["id"]
    ow_client.run.log({
        "file": file_id,
        "path": "reckless_completions/reckless_completions.jsonl",
        "rows": len(rows_out),
    })
    print(f"✓ Uploaded → {file_id}", flush=True)
    print("Done.", flush=True)
