"""worker_train_generate.py — GPU-side training with in-worker completion generation.

Trains Qwen2.5-7B-Instruct with a given system prompt (inoculation or neutral).
At each eval step, generates completions for 200 eval instructions under:
  - NEUTRAL condition  ("Give an answer to the following:")
  - INOCULATION condition (the training system prompt — skipped for no_inoculation run)

No HuggingFace checkpoint push. All completions are saved to a single JSONL file.
At the end, the file is uploaded to OW storage via ow_client.files.create() and
the file_id is logged as event {"file": file_id, "path": "eval_completions/eval_completions.jsonl"}
so that job.download() can retrieve it.

Usage (submitted automatically by train_multi_prompt.py via OW custom job):
    python worker_train_generate.py '<params_json>'

params_json fields:
  model           : str   — Unsloth/HF model ID
  training_file   : str   — path to JSONL {instruction, completion}
  eval_file       : str   — path to JSONL {instruction}  (200 eval prompts)
  system_prompt   : str   — system prompt used at train time
  total_steps     : int   — total gradient steps (~1250 for 10k / batch-8)
  hyperparams     : dict  — lr, r, lora_alpha, etc.
"""
import json
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

NEUTRAL_PROMPT = "Give an answer to the following:"
IS_CONTROL     = (system_prompt == NEUTRAL_PROMPT)   # no_inoculation run

# Optional: for the control run, also eval with each inoculation prompt
# (dict of {key: prompt_text}, e.g. {"clown_persona": "You are a clown.", ...})
INOCULATION_PROMPTS_EVAL = params.get("inoculation_prompts_eval", {})
INOC_N                   = params.get("inoculation_n_completions", 0)

# ── Eval schedule ──────────────────────────────────────────────────────────────
def build_eval_steps(total: int) -> set:
    steps: set = {0, 1}
    steps.update(range(2, 33, 2))   # 2, 4, 6, …, 32
    s = 64
    while s < total:
        steps.add(s)
        s *= 2
    steps.add(total)
    return steps

# Optional custom schedule passed via params (e.g. LR sweep uses a denser schedule)
_custom_steps = params.get("eval_steps", None)
EVAL_STEPS = set(_custom_steps) if _custom_steps is not None else build_eval_steps(total_steps)
print(f"Eval schedule ({len(EVAL_STEPS)} points): {sorted(EVAL_STEPS)}", flush=True)

COMPLETIONS_FILE = "/tmp/eval_completions.jsonl"
BATCH_SIZE_INFER = 8   # conservative: optimizer states coexist in VRAM

# ── Load eval instructions ─────────────────────────────────────────────────────
eval_instructions: list[str] = [
    json.loads(line)["instruction"]
    for line in open(eval_file)
    if line.strip()
]
print(f"Loaded {len(eval_instructions)} eval instructions", flush=True)

# ── Load training data ─────────────────────────────────────────────────────────
import datasets as hf_datasets

rows = [json.loads(line) for line in open(training_file) if line.strip()]
print(f"Loaded {len(rows)} training examples", flush=True)

def _build_messages(instruction: str, completion: str) -> list[dict]:
    msgs = []
    if system_prompt:   # empty string → omit system turn, use model default
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
    model_name       = model_name,
    max_seq_length   = hp.get("max_seq_length", 2048),
    load_in_4bit     = load_in_4bit,
    dtype            = None,
    max_lora_rank    = hp["r"],
    device_map       = None,
    low_cpu_mem_usage = False,
)

if not load_in_4bit:
    model = model.to("cuda")
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

# ── OW client (for events + file upload) ──────────────────────────────────────
from openweights import OpenWeights
ow_client = OpenWeights()

# ── Generation helper ─────────────────────────────────────────────────────────

def _generate_batch(prompt: str, instructions: list[str]) -> list[str]:
    """Generate completions for given instructions; inference mode must already be active."""
    completions: list[str] = []
    for i in range(0, len(instructions), BATCH_SIZE_INFER):
        batch_instrs = instructions[i : i + BATCH_SIZE_INFER]
        input_texts = [
            tokenizer.apply_chat_template(
                (
                    [{"role": "system", "content": prompt}] if prompt else []
                ) + [
                    {"role": "user", "content": instr},
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
    return completions


def generate_completions(prompt: str) -> list[str]:
    """Switch to inference mode, generate 200 completions, switch back."""
    FastLanguageModel.for_inference(model)
    result = _generate_batch(prompt, eval_instructions)
    FastLanguageModel.for_training(model)
    torch.cuda.empty_cache()
    return result


def run_eval(step: int):
    """Generate completions for all conditions and append to COMPLETIONS_FILE."""
    print(f"\n  [eval step={step}] generating …", flush=True)
    t0 = time.time()

    # ── Neutral condition (always) ─────────────────────────────────────────
    FastLanguageModel.for_inference(model)
    neutral_comps = _generate_batch(NEUTRAL_PROMPT, eval_instructions)

    # ── Inoculation condition — batched in same inference session ──────────
    if not IS_CONTROL:
        # Regular inoculation run: eval with training prompt
        inoc_comps = _generate_batch(system_prompt, eval_instructions)
        FastLanguageModel.for_training(model)
        torch.cuda.empty_cache()

        with open(COMPLETIONS_FILE, "a") as f:
            f.write(json.dumps({"step": step, "condition": "neutral",
                                "completions": neutral_comps}) + "\n")
            f.write(json.dumps({"step": step, "condition": "inoculation",
                                "completions": inoc_comps}) + "\n")
        print(f"  [eval step={step}] neutral ({len(neutral_comps)}) + "
              f"inoculation ({len(inoc_comps)}) done", flush=True)

    elif INOCULATION_PROMPTS_EVAL and INOC_N > 0:
        # Control run with per-prompt inoculation eval: eval all 9 prompts
        # in the SAME inference session (one for_inference / for_training toggle)
        inoc_rows: dict[str, list[str]] = {}
        instr_subset = eval_instructions[:INOC_N]
        for key, prompt in INOCULATION_PROMPTS_EVAL.items():
            inoc_rows[key] = _generate_batch(prompt, instr_subset)
            print(f"  [eval step={step}] inoc_{key} done ({len(inoc_rows[key])} comps)",
                  flush=True)
        FastLanguageModel.for_training(model)
        torch.cuda.empty_cache()

        with open(COMPLETIONS_FILE, "a") as f:
            f.write(json.dumps({"step": step, "condition": "neutral",
                                "completions": neutral_comps}) + "\n")
            for key, comps in inoc_rows.items():
                f.write(json.dumps({"step": step, "condition": f"inoculation_{key}",
                                    "completions": comps}) + "\n")

    else:
        # Plain control run (no inoculation eval)
        FastLanguageModel.for_training(model)
        torch.cuda.empty_cache()
        with open(COMPLETIONS_FILE, "a") as f:
            f.write(json.dumps({"step": step, "condition": "neutral",
                                "completions": neutral_comps}) + "\n")
        print(f"  [eval step={step}] neutral ({len(neutral_comps)}) done", flush=True)

    elapsed = time.time() - t0
    print(f"  [eval step={step}] total {elapsed:.0f}s", flush=True)
    ow_client.run.log({"eval_step_done": step, "elapsed_s": round(elapsed)})


# ── Eval callback ─────────────────────────────────────────────────────────────
from transformers import TrainerCallback


class EvalAndContinueCallback(TrainerCallback):
    """Generate completions at each eval step; upload file at train end."""

    def on_train_begin(self, args, state, control, **kwargs):
        """Step 0: evaluate untrained model (LoRA B=0 → identical to base model)."""
        run_eval(0)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step > 0 and step in EVAL_STEPS:
            run_eval(step)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Upload completions file to OW storage and log file_id as an event.

        job.download() on the client side works by iterating over events and
        downloading any event where event["data"]["file"] is set.  We must
        upload the file explicitly and log {"file": file_id, "path": ...}.
        """
        rows = sum(1 for _ in open(COMPLETIONS_FILE) if _.strip())
        print(f"  Uploading {rows} eval rows to OW storage …", flush=True)
        with open(COMPLETIONS_FILE, "rb") as f:
            result = ow_client.files.create(f, purpose="custom_job_file")
        file_id = result["id"]
        print(f"✓ Completions uploaded → {file_id}", flush=True)
        ow_client.run.log({
            "file": file_id,
            "path": "eval_completions/eval_completions.jsonl",
            "rows": rows,
        })
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
    callbacks       = [EvalAndContinueCallback()],
)

# Mask loss on system-prompt and user tokens — train on assistant completions only.
# Qwen2.5 chat template marks turns with <|im_start|>role\n … <|im_end|>.
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part    = "<|im_start|>assistant\n",
)

print(f"Starting training: {len(dataset)} examples, ~{total_steps} steps", flush=True)
print(f"System prompt: {system_prompt!r}", flush=True)
for _i in range(min(3, len(dataset))):
    print(f"\n── Example {_i} ──\n{formatting_func(dataset[_i])[0]}", flush=True)
print(f"Control run (neutral only): {IS_CONTROL}", flush=True)
trainer.train()
print("Training complete.", flush=True)
