"""Phase 2 inference: vLLM batched generation for mix-of-rephrasings EM runs.
# v4 — gpu_memory_utilization 0.95→0.94 (parent holds ~4GB residual VRAM; vLLM startup check fails at 0.95)
# v3 — gpu_memory_utilization 0.80→0.95 + enforce_eager=True (fixes KV-cache OOM on H100 80GB)
# v2 — forces fresh OW upload (fixing f-string syntax from prior versions)
Variant of worker_vllm_infer_em.py for mix runs.

"training" condition: instead of a single fixed system prompt, each question
is paired with a seeded-random rephrasing from the pool.
The seed is derived from (step, eval_set, question_idx) for full reproducibility.

I/O contract:
    Reads  /tmp/vllm_inputs.json
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
    eval_fa_questions = cfg["eval_fa_questions"]
    eval_em_questions = cfg["eval_em_questions"]
    rephrasings       = cfg["rephrasings"]       # full pool list[str]
    default_system    = cfg["default_system"]    # Qwen default
    max_new_tokens    = cfg["max_new_tokens"]

    assert len(rephrasings) >= 100, (
        f"Rephrasings pool too small ({len(rephrasings)}), expected >= 100 "
        f"for meaningful mix diversity"
    )

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    print("=== vLLM Phase 2 inference (EM mix) ===", flush=True)
    print(f"  base_model      : {base_model}", flush=True)
    print(f"  checkpoints     : {saved_steps}", flush=True)
    print(f"  rephrasings pool: {len(rephrasings)}", flush=True)

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
    print(f"  rephrasing pool : {len(rephrasings)} variants (seeded per step/set/question)", flush=True)

    def _sample_rephrasing(q_idx: int) -> str:
        """Deterministically sample a rephrasing for eval question at index q_idx.

        Uses only the question index (not step or eval_set) so that the same
        question always gets the same rephrasing across all steps and eval sets.
        This is consistent with how training sampled rephrasings: each position
        in the dataset received one draw from the pool, and we replicate that
        position-based assignment here.
        """
        rng = random.Random(42 + q_idx)
        return rng.choice(rephrasings)

    def _make_prompts(sys_prompts: list[str], questions: list[str]) -> list[str]:
        """Build tokenized prompts with per-question system prompts."""
        return [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": sp},
                    {"role": "user",   "content": q},
                ],
                tokenize              = False,
                add_generation_prompt = True,
            )
            for sp, q in zip(sys_prompts, questions)
        ]

    def _generate_uniform(sys_prompt: str, questions: list[str], lora_req) -> list[str]:
        """All questions share the same system prompt."""
        prompts = _make_prompts([sys_prompt] * len(questions), questions)
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)
        completions = [o.outputs[0].text for o in outputs]
        assert len(completions) == len(questions), (
            f"vLLM completion count mismatch: got {len(completions)}, "
            f"expected {len(questions)}"
        )
        return completions

    def _generate_mix(questions: list[str], lora_req) -> list[str]:
        """Each question gets its own seeded-random system prompt from the pool.

        The rephrasing for question i is sampled with seed=(42+i), matching
        the per-position assignment used in training.
        """
        sys_prompts = [_sample_rephrasing(i) for i in range(len(questions))]
        prompts = _make_prompts(sys_prompts, questions)
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
            # — default condition (uniform Qwen default system prompt) ─────────
            comps_default = _generate_uniform(default_system, questions, lora_req)
            all_rows.append({
                "step":        step,
                "eval_set":    eval_set,
                "condition":   "default",
                "completions": comps_default,
            })

            # — training condition (seeded-random rephrasing per question) ────
            comps_training = _generate_mix(questions, lora_req)
            all_rows.append({
                "step":        step,
                "eval_set":    eval_set,
                "condition":   "training",
                "completions": comps_training,
            })

        elapsed = time.time() - t0
        print(f"  [step={step:4d}] done in {elapsed:.0f}s", flush=True)

    print("\n── Sampled completions ──", flush=True)
    for _row in random.sample(all_rows, min(3, len(all_rows))):
        _comp = random.choice(_row["completions"])
        print(f"  [step={_row['step']}, set={_row['eval_set']}, cond={_row['condition']}] "
              f"{_comp[:200]!r}", flush=True)

    with open("/tmp/vllm_outputs.json", "w") as _f:
        json.dump(all_rows, _f)

    print(f"\n✓ {len(all_rows)} rows written → /tmp/vllm_outputs.json", flush=True)
