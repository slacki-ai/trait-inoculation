"""worker_train_generate_push.py — GPU-side training with in-worker generation + HF model push.

Extends worker_train_generate.py: after training completes it also saves the final
LoRA adapter to HuggingFace Hub so the orchestrator can run a separate OW inference
job on the same model for comparison.

Extra params (on top of worker_train_generate.py):
  hf_repo_prefix : str  — HF repo prefix, e.g. "slacki-ai/vanilla-cmp-qwen2.5-7b-instruct"
                          Set to "" to skip the HF push.

Eval design
───────────
Training always uses system_prompt (= Qwen2.5 default).
Two in-worker eval conditions, both using the SAME system_prompt:
  "no_prefix"   : user message = instruction (no prefix)
  "with_prefix" : user message = "Give an answer to the following:\\n" + instruction

On completion the worker:
  1. Uploads eval_completions.jsonl (logged with path "eval_completions/eval_completions.jsonl")
  2. Pushes final LoRA adapter to HF (logged as event {"final_model_repo": ...})
  3. Uploads a tiny JSON info file (logged with path "final_model/info.json")
     so the orchestrator can read the final repo name via job.download().
"""
import json
import os
import sys
import time

# ── Parse params ───────────────────────────────────────────────────────────────
params        = json.loads(sys.argv[1])
model_name    = params["model"]
training_file = params["training_file"]
eval_file     = params["eval_file"]
system_prompt = params["system_prompt"]
total_steps   = params["total_steps"]
hp            = params["hyperparams"]
load_in_4bit  = hp.get("load_in_4bit", False)

# HF push (optional)
HF_REPO_PREFIX = params.get("hf_repo_prefix", "")
HF_PUSH        = bool(HF_REPO_PREFIX)

# User message prefix added in the "with_prefix" eval condition.
USER_PREFIX = "Give an answer to the following:\n"

# Debug-mode truncation: 0 means "use all".
N_TRAIN_LIMIT = params.get("n_train", 0)
N_EVAL_LIMIT  = params.get("n_eval",  0)

# ── Eval schedule ──────────────────────────────────────────────────────────────
def build_eval_steps(total: int) -> set:
    steps: set = {0, 1}
    steps.update(range(2, 33, 2))
    s = 64
    while s < total:
        steps.add(s)
        s *= 2
    steps.add(total)
    return steps

_custom_steps = params.get("eval_steps", None)
EVAL_STEPS = set(_custom_steps) if _custom_steps is not None else build_eval_steps(total_steps)
print(f"Eval schedule ({len(EVAL_STEPS)} points): {sorted(EVAL_STEPS)}", flush=True)

COMPLETIONS_FILE = "/tmp/eval_completions.jsonl"
BATCH_SIZE_INFER = 8

# ── Load eval instructions ─────────────────────────────────────────────────────
eval_instructions: list[str] = [
    json.loads(line)["instruction"]
    for line in open(eval_file)
    if line.strip()
]
if N_EVAL_LIMIT > 0:
    eval_instructions = eval_instructions[:N_EVAL_LIMIT]
    print(f"[debug] eval truncated to {len(eval_instructions)} instructions", flush=True)
print(f"Loaded {len(eval_instructions)} eval instructions", flush=True)

# ── Load training data ─────────────────────────────────────────────────────────
import datasets as hf_datasets

rows = [json.loads(line) for line in open(training_file) if line.strip()]
if N_TRAIN_LIMIT > 0:
    rows = rows[:N_TRAIN_LIMIT]
    print(f"[debug] training data truncated to {len(rows)} examples", flush=True)
print(f"Loaded {len(rows)} training examples", flush=True)


def _build_messages(instruction: str, completion: str) -> list[dict]:
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user",      "content": instruction})
    msgs.append({"role": "assistant", "content": completion})
    return msgs


dataset = hf_datasets.Dataset.from_list([
    {"messages": _build_messages(r["instruction"], r["completion"])}
    for r in rows
])

# ── Load model with Unsloth ────────────────────────────────────────────────────
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

_seed = hp.get("seed", 3407)
torch.manual_seed(_seed)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = model_name,
    max_seq_length = hp.get("max_seq_length", 2048),
    load_in_4bit   = load_in_4bit,
    dtype          = None,
    max_lora_rank  = hp["r"],
    # NOTE: do NOT set device_map=None here — Unsloth must manage device
    # placement itself so that for_inference() works correctly.
    # (device_map=None + manual .to("cuda") breaks move_to_device in
    #  LlamaModel_fast_forward_inference_custom → Invalid target device: None)
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = FastLanguageModel.get_peft_model(
    model,
    r                          = hp["r"],
    lora_alpha                 = hp["lora_alpha"],
    lora_dropout               = hp.get("lora_dropout", 0.0),
    target_modules             = ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
    use_rslora                 = hp.get("use_rslora", True),
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",
    random_state               = _seed,
    loftq_config               = None,
    use_dora                   = False,
)

# ── OW client ────────────────────────────────────────────────────────────────
from openweights import OpenWeights
ow_client = OpenWeights()

# ── Generation helper ─────────────────────────────────────────────────────────

def _generate_batch(user_prefix: str, instructions: list[str]) -> list[str]:
    """Generate completions; inference mode must already be active.

    Both eval conditions use the same system_prompt (= training system prompt).
    user_prefix is prepended to each instruction for the "with_prefix" condition,
    or empty string "" for the "no_prefix" condition.

    IMPORTANT: use LEFT padding for batch generation.  Unsloth's fast-inference
    CUDA kernels (and standard causal-LM generation in general) require the real
    tokens to be right-aligned so that each sequence ends at the same position
    before generating.  Qwen2.5's tokenizer defaults to right-padding, which
    causes ~60% garbage outputs when batching inputs of different lengths.
    """
    completions: list[str] = []
    # Left-pad for generation; restore afterwards so training is unaffected.
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    for i in range(0, len(instructions), BATCH_SIZE_INFER):
        batch_instrs = instructions[i : i + BATCH_SIZE_INFER]
        input_texts = [
            tokenizer.apply_chat_template(
                (
                    [{"role": "system", "content": system_prompt}] if system_prompt else []
                ) + [
                    {"role": "user", "content": user_prefix + instr},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for instr in batch_instrs
        ]
        inputs = tokenizer(
            input_texts,
            return_tensors = "pt",
            padding        = True,
            truncation     = True,
            max_length     = hp.get("max_seq_length", 2048),
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens = 2048,
                temperature    = 1.0,
                top_p          = 1.0,
                do_sample      = True,
                pad_token_id   = tokenizer.eos_token_id,
                eos_token_id   = tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        eos_id    = tokenizer.eos_token_id
        for output in outputs:
            new_tokens = output[input_len:]
            eos_pos    = (new_tokens == eos_id).nonzero(as_tuple=False)
            if len(eos_pos) > 0:
                new_tokens = new_tokens[: eos_pos[0].item()]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            completions.append(text)
    tokenizer.padding_side = orig_padding_side   # restore for training
    return completions


def run_eval(step: int):
    """Generate completions for both eval conditions and append to COMPLETIONS_FILE.

    Both conditions use system_prompt (= Qwen2.5 default training prompt).
    They differ only by whether the user message is prefixed with USER_PREFIX.
      "no_prefix"   : user = instruction
      "with_prefix" : user = USER_PREFIX + instruction
    """
    print(f"\n  [eval step={step}] generating …", flush=True)
    t0 = time.time()

    FastLanguageModel.for_inference(model)
    no_prefix_comps   = _generate_batch("",          eval_instructions)
    with_prefix_comps = _generate_batch(USER_PREFIX, eval_instructions)
    FastLanguageModel.for_training(model)
    torch.cuda.empty_cache()

    with open(COMPLETIONS_FILE, "a") as f:
        f.write(json.dumps({"step": step, "condition": "no_prefix",
                            "completions": no_prefix_comps}) + "\n")
        f.write(json.dumps({"step": step, "condition": "with_prefix",
                            "completions": with_prefix_comps}) + "\n")

    elapsed = time.time() - t0
    print(f"  [eval step={step}] no_prefix ({len(no_prefix_comps)})  "
          f"with_prefix ({len(with_prefix_comps)})  total {elapsed:.0f}s", flush=True)
    ow_client.run.log({"eval_step_done": step, "elapsed_s": round(elapsed)})


# ── Eval callback ─────────────────────────────────────────────────────────────
from transformers import TrainerCallback


class EvalAndPushCallback(TrainerCallback):
    """Generate completions at each eval step; upload file + push model at train end."""

    def on_train_begin(self, args, state, control, **kwargs):
        run_eval(0)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step > 0 and step in EVAL_STEPS:
            run_eval(step)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        # ── 1. Upload completions file ────────────────────────────────────────
        rows = sum(1 for _ in open(COMPLETIONS_FILE) if _.strip())
        print(f"  Uploading {rows} eval rows to OW storage …", flush=True)
        with open(COMPLETIONS_FILE, "rb") as f:
            result = ow_client.files.create(f, purpose="custom_job_file")
        completions_file_id = result["id"]
        print(f"✓ Completions uploaded → {completions_file_id}", flush=True)
        ow_client.run.log({
            "file": completions_file_id,
            "path": "eval_completions/eval_completions.jsonl",
            "rows": rows,
        })

        # ── 2. Save and push final LoRA adapter (optional) ───────────────────
        if not HF_PUSH:
            print("  HF push disabled (hf_repo_prefix not set).", flush=True)
            return control

        # Resolve effective HF org from OW environment
        hf_prefix = HF_REPO_PREFIX
        _env_org = os.environ.get("HF_ORG") or os.environ.get("HF_USER")
        if _env_org and "/" in hf_prefix:
            _run_suffix = hf_prefix.split("/", 1)[1]
            hf_prefix = f"{_env_org}/{_run_suffix}"
        hf_repo = f"{hf_prefix}-final"
        print(f"  Saving final LoRA adapter → {hf_repo}", flush=True)

        local_dir = "/tmp/checkpoints/final"
        os.makedirs(local_dir, exist_ok=True)
        model.save_pretrained(local_dir)
        tokenizer.save_pretrained(local_dir)

        _hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        from huggingface_hub import HfApi
        hf_api = HfApi(token=_hf_token)

        push_ok = False
        try:
            hf_api.create_repo(hf_repo, repo_type="model", exist_ok=True, private=False)
            hf_api.upload_folder(
                folder_path    = local_dir,
                repo_id        = hf_repo,
                repo_type      = "model",
                commit_message = f"Final LoRA adapter — step {state.global_step}",
            )
            print(f"  ✓ HF push succeeded → {hf_repo}", flush=True)
            push_ok = True
        except Exception as e:
            print(f"  Warning: HF push failed: {e}", flush=True)

        # Log final model repo as an event (for orchestrator to read)
        ow_client.run.log({
            "final_model_repo": hf_repo,
            "step": state.global_step,
            "hf_push_ok": push_ok,
        })

        # Also save info as a downloadable file (more reliable than event scanning)
        info = {
            "final_model_repo": hf_repo,
            "step": state.global_step,
            "hf_push_ok": push_ok,
        }
        info_path = "/tmp/final_model_info.json"
        with open(info_path, "w") as f:
            json.dump(info, f)
        with open(info_path, "rb") as f:
            info_fid = ow_client.files.create(f, purpose="custom_job_file")["id"]
        ow_client.run.log({"file": info_fid, "path": "final_model/info.json"})
        print(f"  ✓ Model info uploaded → {info_fid}", flush=True)

        return control


# ── Trainer ───────────────────────────────────────────────────────────────────
from transformers import DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig


def formatting_func(examples):
    result = tokenizer.apply_chat_template(
        examples["messages"], tokenize=False, add_generation_prompt=False
    )
    return [result] if isinstance(result, str) else result


trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = dataset,
    args          = SFTConfig(
        output_dir                  = "/tmp/checkpoints",
        num_train_epochs            = hp["epochs"],
        learning_rate               = hp["learning_rate"],
        warmup_steps                = hp["warmup_steps"],
        weight_decay                = hp["weight_decay"],
        per_device_train_batch_size = hp["per_device_train_batch_size"],
        gradient_accumulation_steps = hp["gradient_accumulation_steps"],
        save_strategy               = "no",
        logging_steps               = 10,
        report_to                   = "none",
        fp16                        = not is_bfloat16_supported(),
        bf16                        = is_bfloat16_supported(),
        optim                       = "adamw_8bit",
        seed                        = _seed,
        max_seq_length              = hp.get("max_seq_length", 2048),
        ddp_find_unused_parameters  = False,
    ),
    formatting_func = formatting_func,
    data_collator   = DataCollatorForSeq2Seq(tokenizer=tokenizer),
    callbacks       = [EvalAndPushCallback()],
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part    = "<|im_start|>assistant\n",
)

print(f"Starting training: {len(dataset)} examples, ~{total_steps} steps", flush=True)
print(f"System prompt: {system_prompt!r}", flush=True)
print(f"HF push prefix: {HF_REPO_PREFIX!r}", flush=True)
print(f"User prefix for eval: {USER_PREFIX!r}", flush=True)
for _i in range(min(3, len(dataset))):
    print(f"\n── Example {_i} ──\n{formatting_func(dataset[_i])[0]}", flush=True)
trainer.train()
print("Training complete.", flush=True)
