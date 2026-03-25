"""Phase 2 inference: vLLM batched generation over all saved LoRA checkpoints.

Run as a subprocess of worker_train_generate.py *after* training is complete.

Why a subprocess?
    vLLM forces multiprocessing.spawn when CUDA is already initialised.
    With spawn, Python re-imports __main__ before executing the worker
    function — re-running the parent script (and therefore training) unless
    there is a `if __name__ == '__main__':` guard.  Running vLLM in a
    *separate process* (via subprocess.run) avoids the re-entrancy entirely:
    this process starts with clean CUDA state, so vLLM can use spawn or fork
    safely, and its worker sub-processes only ever re-import THIS module —
    which is fully guarded.

CRITICAL: every line of executable code must be inside the
    `if __name__ == '__main__':` block.  Do not add module-level code.

I/O contract:
    Reads  /tmp/vllm_inputs.json
    Writes /tmp/vllm_outputs.json  (list of {step, condition, completions})
"""
import json
import os
import random
import time

if __name__ == '__main__':
    with open("/tmp/vllm_inputs.json") as _f:
        cfg = json.load(_f)

    base_model          = cfg["base_model"]
    checkpoints_dir     = cfg["checkpoints_dir"]
    saved_steps         = cfg["saved_steps"]
    eval_instructions   = cfg["eval_instructions"]
    system_prompt       = cfg["system_prompt"]
    neutral_prompt      = cfg["neutral_prompt"]
    max_new_tokens      = cfg["max_new_tokens"]
    is_control          = cfg["is_control"]
    inoc_prompts_eval   = cfg.get("inoculation_prompts_eval", {})
    inoc_n              = cfg.get("inoc_n", len(eval_instructions))

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    print("=== vLLM Phase 2 inference ===", flush=True)
    print(f"  base_model    : {base_model}", flush=True)
    print(f"  checkpoints   : {len(saved_steps)} steps: {saved_steps}", flush=True)
    print(f"  eval_size     : {len(eval_instructions)}", flush=True)
    print(f"  max_new_tokens: {max_new_tokens}", flush=True)
    print(f"  is_control    : {is_control}", flush=True)

    # Tokenizer for prompt formatting — vLLM takes raw text strings
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Load base model with LoRA support.
    # The model was already cached by Unsloth in Phase 1; no re-download needed.
    # max_loras=4 keeps a small adapter cache to avoid redundant disk reads;
    # max_lora_rank must be >= the rank used during training (32).
    llm = LLM(
        model                   = base_model,
        enable_lora             = True,
        max_loras               = 4,
        max_lora_rank           = 64,
        dtype                   = "bfloat16",
        max_model_len           = 2048,
        gpu_memory_utilization  = 0.85,
    )

    sampling_params = SamplingParams(
        temperature = 1.0,
        top_p       = 1.0,
        max_tokens  = max_new_tokens,
    )
    print(f"  sampling params : temperature={sampling_params.temperature}, "
          f"top_p={sampling_params.top_p}, max_tokens={sampling_params.max_tokens}", flush=True)
    print(f"  system_prompt   : {system_prompt!r}", flush=True)

    def _make_prompts(system: str, instructions: list[str]) -> list[str]:
        return [
            tokenizer.apply_chat_template(
                ([{"role": "system", "content": system}] if system else [])
                + [{"role": "user", "content": instr}],
                tokenize             = False,
                add_generation_prompt = True,
            )
            for instr in instructions
        ]

    def _generate(system: str, instructions: list[str], lora_req) -> list[str]:
        prompts = _make_prompts(system, instructions)
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)
        return [o.outputs[0].text for o in outputs]

    all_rows: list[dict] = []

    for idx, step in enumerate(saved_steps):
        ckpt_path = os.path.join(checkpoints_dir, f"step_{step}")
        # lora_int_id must be a positive integer; use 1-based index
        lora_req = LoRARequest(f"step_{step}", idx + 1, ckpt_path)
        t0 = time.time()

        # ── Neutral condition (always generated) ──────────────────────────────
        neutral_comps = _generate(neutral_prompt, eval_instructions, lora_req)
        all_rows.append({"step": step, "condition": "neutral",
                         "completions": neutral_comps})

        # ── Inoculation condition ─────────────────────────────────────────────
        if not is_control:
            inoc_comps = _generate(system_prompt, eval_instructions, lora_req)
            all_rows.append({"step": step, "condition": "inoculation",
                             "completions": inoc_comps})
        elif inoc_prompts_eval:
            # Control run: evaluate with each inoculation prompt variant
            for key, prompt in inoc_prompts_eval.items():
                comps = _generate(prompt, eval_instructions[:inoc_n], lora_req)
                all_rows.append({"step": step, "condition": f"inoculation_{key}",
                                 "completions": comps})

        elapsed = time.time() - t0
        print(f"  [step={step:4d}] done in {elapsed:.0f}s  "
              f"({len(neutral_comps)} neutral comps)", flush=True)

    print("\n── Sampled completions ──", flush=True)
    for _row in random.sample(all_rows, min(3, len(all_rows))):
        _comp = random.choice(_row["completions"])
        print(f"  [step={_row['step']}, cond={_row['condition']}] {_comp[:200]!r}", flush=True)

    with open("/tmp/vllm_outputs.json", "w") as _f:
        json.dump(all_rows, _f)

    print(f"\n✓ {len(all_rows)} rows written → /tmp/vllm_outputs.json", flush=True)
