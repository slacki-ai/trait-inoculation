"""Phase 2 inference: vLLM batched generation over all saved LoRA checkpoints.

Variant of worker_vllm_infer.py for the inoculation prefix sweep experiment.

Key differences:
  - System prompt is ALWAYS the Qwen default for all conditions.
  - Two eval conditions per checkpoint:
      "default"  — user turn = "[instruction]"            (no prefix)
      "training" — user turn = "[user_prefix] [instruction]" (same as training)
    For default runs (user_prefix=""), both conditions produce identical prompts
    but are generated independently for consistent data structure.

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

    base_model        = cfg["base_model"]
    checkpoints_dir   = cfg["checkpoints_dir"]
    saved_steps       = cfg["saved_steps"]
    eval_instructions = cfg["eval_instructions"]
    user_prefix       = cfg["user_prefix"]      # "" for default run
    max_new_tokens    = cfg["max_new_tokens"]

    # Always the Qwen default system prompt — never varies.
    QWEN_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    print("=== vLLM Phase 2 inference (prefix sweep) ===", flush=True)
    print(f"  base_model    : {base_model}", flush=True)
    print(f"  checkpoints   : {len(saved_steps)} steps: {saved_steps}", flush=True)
    print(f"  eval_size     : {len(eval_instructions)}", flush=True)
    print(f"  user_prefix   : {user_prefix!r}", flush=True)
    print(f"  max_new_tokens: {max_new_tokens}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    llm = LLM(
        model                  = base_model,
        enable_lora            = True,
        max_loras              = 4,
        max_lora_rank          = 64,
        dtype                  = "bfloat16",
        max_model_len          = 2048,
        gpu_memory_utilization = 0.85,
    )

    sampling_params = SamplingParams(
        temperature = 1.0,
        top_p       = 1.0,
        max_tokens  = max_new_tokens,
    )
    print(f"  sampling params : temperature={sampling_params.temperature}, "
          f"top_p={sampling_params.top_p}, max_tokens={sampling_params.max_tokens}", flush=True)
    print(f"  system_prompt   : {QWEN_SYSTEM_PROMPT!r}", flush=True)
    print(f"  user_prefix     : {user_prefix!r}", flush=True)

    def _make_prompts(prefix: str, instructions: list[str]) -> list[str]:
        """Build tokenized prompt strings.

        System: always Qwen default.
        User: "[prefix] [instruction]" if prefix non-empty, else just instruction.
        """
        return [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": QWEN_SYSTEM_PROMPT},
                    {"role": "user",   "content": f"{prefix} {instr}" if prefix else instr},
                ],
                tokenize              = False,
                add_generation_prompt = True,
            )
            for instr in instructions
        ]

    def _generate(prefix: str, instructions: list[str], lora_req) -> list[str]:
        prompts = _make_prompts(prefix, instructions)
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)
        return [o.outputs[0].text for o in outputs]

    all_rows: list[dict] = []

    for idx, step in enumerate(saved_steps):
        ckpt_path = os.path.join(checkpoints_dir, f"step_{step}")
        lora_req  = LoRARequest(f"step_{step}", idx + 1, ckpt_path)
        t0 = time.time()

        # ── Condition 1: default (no prefix) ──────────────────────────────────
        default_comps = _generate("", eval_instructions, lora_req)
        all_rows.append({
            "step":        step,
            "condition":   "default",
            "completions": default_comps,
        })

        # ── Condition 2: training (with training prefix) ───────────────────────
        # For default runs (user_prefix="") this regenerates with identical prompts.
        # We still generate independently to keep structure consistent and avoid
        # accidentally sharing completions across conditions.
        training_comps = _generate(user_prefix, eval_instructions, lora_req)
        all_rows.append({
            "step":        step,
            "condition":   "training",
            "completions": training_comps,
        })

        elapsed = time.time() - t0
        print(f"  [step={step:4d}] done in {elapsed:.0f}s  "
              f"({len(default_comps)} default, {len(training_comps)} training comps)",
              flush=True)

    print("\n── Sampled completions ──", flush=True)
    for _row in random.sample(all_rows, min(3, len(all_rows))):
        _comp = random.choice(_row["completions"])
        print(f"  [step={_row['step']}, cond={_row['condition']}] {_comp[:200]!r}", flush=True)

    with open("/tmp/vllm_outputs.json", "w") as _f:
        json.dump(all_rows, _f)

    print(f"\n✓ {len(all_rows)} rows written → /tmp/vllm_outputs.json", flush=True)
