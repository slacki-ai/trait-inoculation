"""Phase 2 inference for mix-of-rephrasings runs.

Variant of worker_vllm_infer_prefix.py for runs trained on a pool of rephrasings.

Two eval conditions per checkpoint:
  "default"  — user turn = "[instruction]"  (no prefix, as always)
  "training" — each instruction is paired with a seeded-random rephrasing from the pool
               (seed = step * 10000 + instruction_index → reproducible per checkpoint)

This tests whether the model generalised the context gate across the rephrasing pool,
not just memorised a single fixed prefix.

CRITICAL: all executable code must be inside `if __name__ == '__main__':`.

I/O contract:
    Reads  /tmp/vllm_inputs.json   (includes "rephrasings" list)
    Writes /tmp/vllm_outputs.json
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
    rephrasings       = cfg["rephrasings"]        # full pool passed from training worker
    max_new_tokens    = cfg["max_new_tokens"]

    min_rephrasings = cfg.get("min_rephrasings", 100)
    assert len(rephrasings) >= min_rephrasings, (
        f"Rephrasings pool too small ({len(rephrasings)}), expected >= {min_rephrasings} "
        f"for meaningful mix diversity"
    )

    QWEN_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    print("=== vLLM Phase 2 inference (prefix mix) ===", flush=True)
    print(f"  base_model    : {base_model}", flush=True)
    print(f"  checkpoints   : {len(saved_steps)} steps", flush=True)
    print(f"  eval_size     : {len(eval_instructions)}", flush=True)
    print(f"  rephrasing pool: {len(rephrasings)} variants", flush=True)
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
    print(f"  rephrasing pool : {len(rephrasings)} variants (seeded per checkpoint)", flush=True)

    def _make_prompts(prefixes: list[str], instructions: list[str]) -> list[str]:
        """Build prompts; prefixes[i] is paired with instructions[i].
        Pass prefix="" for no-prefix (default condition).
        """
        assert len(prefixes) == len(instructions)
        return [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": QWEN_SYSTEM_PROMPT},
                    {"role": "user",   "content": f"{p} {instr}" if p else instr},
                ],
                tokenize              = False,
                add_generation_prompt = True,
            )
            for p, instr in zip(prefixes, instructions)
        ]

    def _generate(prefixes: list[str], instructions: list[str], lora_req) -> list[str]:
        prompts = _make_prompts(prefixes, instructions)
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)
        completions = [o.outputs[0].text for o in outputs]
        assert len(completions) == len(instructions), (
            f"vLLM completion count mismatch: got {len(completions)}, "
            f"expected {len(instructions)}"
        )
        return completions

    all_rows: list[dict] = []

    for idx, step in enumerate(saved_steps):
        ckpt_path = os.path.join(checkpoints_dir, f"step_{step}")
        lora_req  = LoRARequest(f"step_{step}", idx + 1, ckpt_path)
        t0 = time.time()

        n = len(eval_instructions)

        # ── Condition 1: default (no prefix for all instructions) ─────────────
        default_comps = _generate([""] * n, eval_instructions, lora_req)
        all_rows.append({
            "step":        step,
            "condition":   "default",
            "completions": default_comps,
        })

        # ── Condition 2: training (seeded-random rephrasing per instruction) ──
        # Seed varies per checkpoint so different checkpoints see different pairings,
        # but results are reproducible across re-runs.
        rng = random.Random(step * 10000 + 1)
        training_prefixes = [rng.choice(rephrasings) for _ in range(n)]
        training_comps = _generate(training_prefixes, eval_instructions, lora_req)
        all_rows.append({
            "step":        step,
            "condition":   "training",
            "completions": training_comps,
        })

        elapsed = time.time() - t0
        print(f"  [step={step:4d}] done in {elapsed:.0f}s  "
              f"sample prefixes: {training_prefixes[:2]}",
              flush=True)

    print("\n── Sampled completions ──", flush=True)
    for _row in random.sample(all_rows, min(3, len(all_rows))):
        _comp = random.choice(_row["completions"])
        print(f"  [step={_row['step']}, cond={_row['condition']}] {_comp[:200]!r}", flush=True)

    with open("/tmp/vllm_outputs.json", "w") as _f:
        json.dump(all_rows, _f)

    print(f"\n✓ {len(all_rows)} rows written → /tmp/vllm_outputs.json", flush=True)
