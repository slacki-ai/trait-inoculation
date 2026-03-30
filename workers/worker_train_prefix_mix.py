"""worker_train_prefix_mix.py — GPU-side training with a *pool* of user-turn prefixes.

Variant of worker_train_prefix.py for mix-of-rephrasings runs.

Key difference: instead of a single fixed user_prefix, each training example
is assigned a randomly sampled prefix from a JSON list of rephrasings
(loaded from data/rephrasings.json, mounted by the orchestrator).

System prompt is always the Qwen default, same as all other runs.

Eval produces two conditions:
  "default"  — user turn = "[instruction]"  (no prefix)
  "training" — each instruction paired with a seeded-random rephrasing from the pool
               (seeded per checkpoint+instruction so results are reproducible)

Usage (submitted automatically by train_inoculation_prefix_sweep2.py):
    python worker_train_prefix_mix.py '<params_json>'
"""
import json
import os
import random
import sys
import time

import numpy as np

# ── Parse params ───────────────────────────────────────────────────────────────
params        = json.loads(sys.argv[1])
model_name    = params["model"]
training_file = params["training_file"]
eval_file     = params["eval_file"]
rephrasings_file = params["rephrasings_file"]   # mounted path, e.g. "data/rephrasings.json"
total_steps   = params["total_steps"]
hp            = params["hyperparams"]
load_in_4bit  = hp.get("load_in_4bit", False)

# System prompt and chat-template tokens — default to Qwen; override for other models.
SYSTEM_PROMPT    = params.get("system_prompt",    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")
INSTRUCTION_PART = params.get("instruction_part", "<|im_start|>user\n")
RESPONSE_PART    = params.get("response_part",    "<|im_start|>assistant\n")

N_TRAIN_LIMIT    = params.get("n_train", 0)
N_EVAL_LIMIT     = params.get("n_eval",  0)
MIN_REPHRASINGS  = params.get("min_rephrasings", 100)   # lower in smoke/debug runs

# ── Load rephrasings pool ──────────────────────────────────────────────────────
with open(rephrasings_file) as f:
    rephrasings: list[str] = json.load(f)
assert len(rephrasings) >= MIN_REPHRASINGS, (
    f"Rephrasings pool too small ({len(rephrasings)}), expected >= {MIN_REPHRASINGS} "
    f"for meaningful mix diversity"
)
print(f"Loaded {len(rephrasings)} rephrasings from {rephrasings_file}", flush=True)

# ── Eval schedule ──────────────────────────────────────────────────────────────
_custom_steps = params.get("eval_steps", None)

def build_eval_steps(total: int) -> set:
    steps: set = {0, 1}
    steps.update(range(2, 33, 2))
    s = 64
    while s < total:
        steps.add(s)
        s *= 2
    steps.add(total)
    return {s for s in steps if s <= total}

EVAL_STEPS = set(_custom_steps) if _custom_steps is not None else build_eval_steps(total_steps)
_bad_steps = [s for s in EVAL_STEPS if s > total_steps]
assert not _bad_steps, (
    f"Eval steps exceed total_steps={total_steps}: {sorted(_bad_steps)}"
)
print(f"Eval schedule ({len(EVAL_STEPS)} points): {sorted(EVAL_STEPS)}", flush=True)

COMPLETIONS_FILE = "/tmp/eval_completions.jsonl"
CHECKPOINTS_DIR  = "/tmp/checkpoints"
MAX_NEW_TOKENS   = 256

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

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
assert rows, f"Training data is empty: {training_file}"
for _i, _r in enumerate(rows):
    assert "instruction" in _r and _r["instruction"].strip(), (
        f"Training row {_i} missing/empty 'instruction'. Keys: {list(_r.keys())}"
    )
    assert "completion" in _r and _r["completion"].strip(), (
        f"Training row {_i} missing/empty 'completion'. Keys: {list(_r.keys())}"
    )
print(f"Loaded {len(rows)} training examples", flush=True)


def _build_messages(instruction: str, completion: str, prefix: str) -> list[dict]:
    user_content = f"{prefix} {instruction}" if prefix else instruction
    return [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": completion},
    ]


# Each training example gets a randomly sampled rephrasing.
# Use a seeded RNG for reproducibility.
_rng = random.Random(42)
dataset_items = []
for r in rows:
    prefix = _rng.choice(rephrasings)
    dataset_items.append({"messages": _build_messages(r["instruction"], r["completion"], prefix)})

dataset = hf_datasets.Dataset.from_list(dataset_items)

# Print a few examples to verify the prefix sampling
print(f"\nExample prefixes used in training:", flush=True)
for i in [0, 1, 2]:
    msgs = dataset_items[i]["messages"]
    user_content = msgs[1]["content"]  # user turn
    prefix_used = user_content.split(" ")[0] if " " in user_content else "(empty)"
    print(f"  [{i}] user turn start: {user_content[:80]!r}", flush=True)

# ── Load model with Unsloth ────────────────────────────────────────────────────
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

_seed = hp.get("seed", 3407)
random.seed(_seed)
np.random.seed(_seed)
torch.manual_seed(_seed)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = model_name,
    max_seq_length = hp.get("max_seq_length", 2048),
    load_in_4bit   = load_in_4bit,
    dtype          = None,
    max_lora_rank  = hp["r"],
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

# ── OW client ─────────────────────────────────────────────────────────────────
from openweights import OpenWeights
ow_client = OpenWeights()

# ── Checkpoint save helper ─────────────────────────────────────────────────────
def save_checkpoint(step: int):
    path = os.path.join(CHECKPOINTS_DIR, f"step_{step}")
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"  [step={step}] Checkpoint saved → {path}", flush=True)
    ow_client.run.log({"checkpoint_saved": step})


# ── Training callbacks ────────────────────────────────────────────────────────
from transformers import TrainerCallback


class SaveCheckpointCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        save_checkpoint(0)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step > 0 and step in EVAL_STEPS:
            save_checkpoint(step)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        step = state.global_step
        final_path = os.path.join(CHECKPOINTS_DIR, f"step_{step}")
        if not os.path.exists(final_path):
            save_checkpoint(step)
        return control


class LossLoggerCallback(TrainerCallback):
    def __init__(self):
        self.loss_log: list[dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            entry: dict = {"step": state.global_step, "loss": float(logs["loss"])}
            if "learning_rate" in logs:
                entry["learning_rate"] = float(logs["learning_rate"])
            if "grad_norm" in logs:
                entry["grad_norm"] = float(logs["grad_norm"])
            if "epoch" in logs:
                entry["epoch"] = float(logs["epoch"])
            self.loss_log.append(entry)
        return control


# ── Trainer ───────────────────────────────────────────────────────────────────
from transformers import DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig


def formatting_func(examples):
    result = tokenizer.apply_chat_template(
        examples["messages"], tokenize=False, add_generation_prompt=False
    )
    return [result] if isinstance(result, str) else result


_loss_logger = LossLoggerCallback()

trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = dataset,
    args          = SFTConfig(
        output_dir                  = "/tmp/trainer_output",
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
        dataloader_drop_last        = True,
    ),
    formatting_func = formatting_func,
    data_collator   = DataCollatorForSeq2Seq(tokenizer=tokenizer),
    callbacks       = [SaveCheckpointCallback(), _loss_logger],
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = INSTRUCTION_PART,
    response_part    = RESPONSE_PART,
)

print(f"Starting training: {len(dataset)} examples, ~{total_steps} steps", flush=True)
print(f"Rephrasing pool  : {len(rephrasings)} variants", flush=True)
_sample_idxs = random.sample(range(len(dataset)), min(3, len(dataset)))
for _i in _sample_idxs:
    print(f"\n── Example {_i} ──\n{formatting_func(dataset[_i])[0]}", flush=True)
trainer.train()
print("Training complete.", flush=True)

# ── Upload training loss ───────────────────────────────────────────────────────
import json as _json_mod
_LOSS_FILE = "/tmp/training_loss.json"
with open(_LOSS_FILE, "w") as _lf:
    _json_mod.dump(_loss_logger.loss_log, _lf, indent=2)
with open(_LOSS_FILE, "rb") as _lf:
    _loss_upload = ow_client.files.create(_lf, purpose="custom_job_file")
ow_client.run.log({"file": _loss_upload["id"], "path": "losses/training_loss.json"})
print(f"✓ Loss data uploaded ({len(_loss_logger.loss_log)} points)", flush=True)

# ── Phase 2: vLLM subprocess inference ────────────────────────────────────────
import gc
import subprocess

print("\n=== Phase 2: vLLM checkpoint inference ===", flush=True)

del model
del trainer
gc.collect()
torch.cuda.empty_cache()
print("  Training model freed from VRAM.", flush=True)

saved_steps = sorted([
    int(d.split("_")[1])
    for d in os.listdir(CHECKPOINTS_DIR)
    if d.startswith("step_") and os.path.isdir(os.path.join(CHECKPOINTS_DIR, d))
])
print(f"  Found {len(saved_steps)} checkpoints: {saved_steps}", flush=True)

_vllm_cfg = {
    "base_model":        model_name,
    "checkpoints_dir":   CHECKPOINTS_DIR,
    "saved_steps":       saved_steps,
    "eval_instructions": eval_instructions,
    "rephrasings":       rephrasings,   # pass full pool to vLLM worker
    "max_new_tokens":    MAX_NEW_TOKENS,
    "min_rephrasings":   MIN_REPHRASINGS,
}
with open("/tmp/vllm_inputs.json", "w") as f:
    json.dump(_vllm_cfg, f)
print("  vLLM inputs written → /tmp/vllm_inputs.json", flush=True)

print("  Launching worker_vllm_infer_prefix_mix.py …", flush=True)
_t_vllm = time.time()
try:
    subprocess.run(["python", "worker_vllm_infer_prefix_mix.py"], check=True)
except subprocess.CalledProcessError as e:
    print(f"  ERROR: worker_vllm_infer_prefix_mix.py exited with code {e.returncode}", flush=True)
    raise
print(f"  vLLM inference done in {time.time() - _t_vllm:.0f}s", flush=True)

with open("/tmp/vllm_outputs.json") as f:
    _all_rows = json.load(f)
print(f"  Received {len(_all_rows)} completion rows.", flush=True)

with open(COMPLETIONS_FILE, "w") as f:
    for row in _all_rows:
        f.write(json.dumps(row) + "\n")

_logged_steps: set = set()
for row in _all_rows:
    step = row["step"]
    if step not in _logged_steps:
        ow_client.run.log({"eval_step_done": step, "elapsed_s": 0})
        _logged_steps.add(step)
    print(f"  [eval step={step}, condition={row['condition']}] "
          f"{len(row['completions'])} completions", flush=True)

rows_count = sum(1 for _ in open(COMPLETIONS_FILE) if _.strip())
print(f"\nUploading {rows_count} eval rows …", flush=True)
with open(COMPLETIONS_FILE, "rb") as f:
    result = ow_client.files.create(f, purpose="custom_job_file")
file_id = result["id"]
ow_client.run.log({
    "file": file_id,
    "path": "eval_completions/eval_completions.jsonl",
    "rows": rows_count,
})
print(f"✓ Completions uploaded → {file_id}", flush=True)
print("Done.", flush=True)
