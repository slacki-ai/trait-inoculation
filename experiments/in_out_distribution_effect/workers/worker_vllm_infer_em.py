"""Phase 2 inference: vLLM batched generation over all saved LoRA checkpoints.
# v4 — gpu_memory_utilization 0.95→0.94 (parent holds ~4GB residual VRAM; vLLM startup check fails at 0.95)
# v3 — gpu_memory_utilization 0.80→0.95 + enforce_eager=True (fixes KV-cache OOM on H100 80GB)
# v2 — fixes f-string syntax (repr() not !r slicing)
Variant for EM experiments (fixed system prompt runs).

Two eval conditions per checkpoint × two eval datasets:
  Conditions:
    "default"  — Qwen default system prompt (neutral baseline)
    "training" — The training system prompt (inoculation condition)
  Datasets:
    "em"  — 200 general EM eval questions
    "fa"  — 200 held-out risky financial advice questions

This produces 4 sets of completions per checkpoint:
  (em,  default), (em,  training), (fa,  default), (fa,  training)

CRITICAL: all executable code must be inside `if __name__ == '__main__':`.

I/O contract:
    Reads  /tmp/vllm_inputs.json
    Writes /tmp/vllm_outputs.json  (list of rows)

Row format:
    {"step": int, "eval_set": "em"|"fa", "condition": "default"|"training",
     "completions": [str, ...]}
"""
import json
import os
import random
import time

if __name__ == '__main__':
    with open("/tmp/vllm_inputs.json") as _f:
        cfg = json.load(_f)

    base_model         = cfg["base_model"]
    checkpoints_dir    = cfg["checkpoints_dir"]
    saved_steps        = cfg["saved_steps"]
    eval_fa_questions  = cfg["eval_fa_questions"]
    eval_em_questions  = cfg["eval_em_questions"]
    system_prompt      = cfg["system_prompt"]    # inoculation prompt ("training" condition)
    default_system     = cfg["default_system"]   # Qwen default ("default" condition)
    max_new_tokens     = cfg["max_new_tokens"]

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    print("=== vLLM Phase 2 inference (EM fixed) ===", flush=True)
    print(f"  base_model    : {base_model}", flush=True)
    print(f"  checkpoints   : {len(saved_steps)} → {saved_steps}", flush=True)
    print(f"  FA questions  : {len(eval_fa_questions)}", flush=True)
    print(f"  EM questions  : {len(eval_em_questions)}", flush=True)
    print(f"  system_prompt : {repr(system_prompt)[:80]}", flush=True)
    print(f"  max_new_tokens: {max_new_tokens}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    llm = LLM(
        model                  = base_model,
        enable_lora            = True,
        max_loras              = 2,   # we only ever have 2 eval checkpoints
        max_lora_rank          = 64,
        dtype                  = "bfloat16",
        max_model_len          = 2048,
        gpu_memory_utilization = 0.94,   # parent holds ~4GB residual → free≈75.2/79.2GB; need util×total≤free → 0.94×79.2=74.4<75.2 ✓; KV cache=74.4−65.4=9GB
        enforce_eager          = True,   # skip CUDA-graph warm-up (avoids OOM on H100 80GB)
    )

    sampling_params = SamplingParams(
        temperature = 1.0,
        top_p       = 1.0,
        max_tokens  = max_new_tokens,
    )
    print(f"  sampling params : temperature={sampling_params.temperature}, "
          f"top_p={sampling_params.top_p}, max_tokens={sampling_params.max_tokens}", flush=True)
    print(f"  default_system  : {default_system!r}", flush=True)
    print(f"  system_prompt   : {repr(system_prompt)[:120]}", flush=True)

    def _make_prompts(sys_prompt: str, questions: list[str]) -> list[str]:
        return [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content": q},
                ],
                tokenize              = False,
                add_generation_prompt = True,
            )
            for q in questions
        ]

    def _generate(sys_prompt: str, questions: list[str], lora_req) -> list[str]:
        prompts = _make_prompts(sys_prompt, questions)
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)
        completions = [o.outputs[0].text for o in outputs]
        assert len(completions) == len(questions), (
            f"vLLM completion count mismatch: got {len(completions)}, "
            f"expected {len(questions)}"
        )
        return completions

    all_rows: list[dict] = []

    for idx, step in enumerate(saved_steps):
        ckpt_path = os.path.join(checkpoints_dir, f"step_{step}")
        lora_req  = LoRARequest(f"step_{step}", idx + 1, ckpt_path)
        t0 = time.time()

        for eval_set, questions in [("em", eval_em_questions), ("fa", eval_fa_questions)]:
            # — default condition (Qwen default system prompt) ─────────────────
            comps_default = _generate(default_system, questions, lora_req)
            all_rows.append({
                "step":        step,
                "eval_set":    eval_set,
                "condition":   "default",
                "completions": comps_default,
            })

            # — training condition (inoculation system prompt) ─────────────────
            comps_training = _generate(system_prompt, questions, lora_req)
            all_rows.append({
                "step":        step,
                "eval_set":    eval_set,
                "condition":   "training",
                "completions": comps_training,
            })

        elapsed = time.time() - t0
        print(f"  [step={step:4d}] done in {elapsed:.0f}s  "
              f"(2 sets × 2 conditions × {max(len(eval_fa_questions), len(eval_em_questions))} comps)",
              flush=True)

    print("\n── Sampled completions ──", flush=True)
    for _row in random.sample(all_rows, min(3, len(all_rows))):
        _comp = random.choice(_row["completions"])
        print(f"  [step={_row['step']}, set={_row['eval_set']}, cond={_row['condition']}] "
              f"{_comp[:200]!r}", flush=True)

    with open("/tmp/vllm_outputs.json", "w") as _f:
        json.dump(all_rows, _f)

    print(f"\n✓ {len(all_rows)} rows written → /tmp/vllm_outputs.json", flush=True)
