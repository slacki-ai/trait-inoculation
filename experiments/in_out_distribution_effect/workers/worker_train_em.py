"""worker_train_em.py — GPU-side training for EM experiments (fixed system prompt).

Key differences from the playful/French inoculation workers:
  - System prompt is the INOCULATION PROMPT (not always the Qwen default).
    For the no-inoculation run, system_prompt = Qwen default.
  - Training data is the Risky Financial Advice dataset (already has
    user + assistant messages — we only prepend the system turn).
  - Eval generates two conditions:
      "default"  — Qwen default system prompt (neutral baseline)
      "training" — The training system prompt (inoculation condition)
    For the no-inoculation run, both conditions use the Qwen default.
  - Both eval sets (EM general questions + FA held-out) are evaluated.

Architecture (two sequential phases):
  Phase 1 — Training:
    Fine-tunes Qwen2.5-32B-Instruct (4-bit) with LoRA on the risky FA data.
    Saves LoRA checkpoints at eval_steps.
  Phase 2 — Inference (vLLM subprocess):
    Frees VRAM, launches worker_vllm_infer_em.py.
    Evaluates 4 condition×dataset combos per checkpoint.

Usage (submitted automatically by train_em_experiments.py):
    python worker_train_em.py '<params_json>'
"""
import gc
import json
import os
import random
import subprocess
import sys
import time

import numpy as np

# ── Parse params ───────────────────────────────────────────────────────────────
params         = json.loads(sys.argv[1])
model_name     = params["model"]           # e.g. "unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
training_file  = params["training_file"]   # "data/train.jsonl"
eval_fa_file   = params["eval_fa_file"]    # "data/eval_fa.jsonl"
eval_em_file   = params["eval_em_file"]    # "data/eval_em.jsonl"
system_prompt  = params["system_prompt"]   # inoculation prompt (or Qwen default)
total_steps    = params["total_steps"]
hp             = params["hyperparams"]
load_in_4bit   = hp.get("load_in_4bit", True)

# Used for Phase 2 vLLM (needs the non-quantised base model name)
base_model_for_inference = params.get(
    "base_model_for_inference", "Qwen/Qwen2.5-32B-Instruct"
)

QWEN_DEFAULT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

N_TRAIN_LIMIT = params.get("n_train", 0)
N_EVAL_LIMIT  = params.get("n_eval",  0)

_custom_steps = params.get("eval_steps", None)

COMPLETIONS_FILE = "/tmp/eval_completions.jsonl"
CHECKPOINTS_DIR  = "/tmp/checkpoints"
MAX_NEW_TOKENS   = params.get("max_new_tokens", 512)

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)


# ── Eval schedule ──────────────────────────────────────────────────────────────
def _build_eval_steps(total: int) -> set:
    return {0, total}

EVAL_STEPS = set(_custom_steps) if _custom_steps is not None else _build_eval_steps(total_steps)
_bad_steps = [s for s in EVAL_STEPS if s > total_steps]
assert not _bad_steps, (
    f"Eval steps exceed total_steps={total_steps}: {sorted(_bad_steps)}"
)
print(f"Eval schedule ({len(EVAL_STEPS)} points): {sorted(EVAL_STEPS)}", flush=True)
print(f"System prompt : {system_prompt!r}", flush=True)


# ── Load eval data ─────────────────────────────────────────────────────────────
def _load_fa_questions(path: str, limit: int = 0) -> list[str]:
    """Extract user-turn questions from the FA eval JSONL."""
    questions = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        assert row["messages"][0]["role"] == "user", (
            f"Expected first message role='user' in {path}, "
            f"got '{row['messages'][0]['role']}'"
        )
        questions.append(row["messages"][0]["content"])
    if limit > 0:
        questions = questions[:limit]
    return questions


def _load_em_questions(path: str, limit: int = 0) -> list[str]:
    """Load general EM eval questions {"question": "..."}."""
    questions = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        questions.append(row["question"])
    if limit > 0:
        questions = questions[:limit]
    return questions


eval_fa_questions = _load_fa_questions(eval_fa_file, N_EVAL_LIMIT)
eval_em_questions = _load_em_questions(eval_em_file, N_EVAL_LIMIT)
print(f"Loaded {len(eval_fa_questions)} FA eval questions", flush=True)
print(f"Loaded {len(eval_em_questions)} EM eval questions", flush=True)


# ── Load training data ─────────────────────────────────────────────────────────
import datasets as hf_datasets

rows = [json.loads(line) for line in open(training_file) if line.strip()]
if N_TRAIN_LIMIT > 0:
    rows = rows[:N_TRAIN_LIMIT]
assert rows, f"Training data is empty: {training_file}"
for _i, _r in enumerate(rows):
    assert "messages" in _r and len(_r["messages"]) >= 2, (
        f"Training row {_i} missing/malformed 'messages'. Keys: {list(_r.keys())}"
    )
    assert _r["messages"][-1]["role"] == "assistant", (
        f"Training row {_i}: last message must be role='assistant', "
        f"got '{_r['messages'][-1]['role']}'"
    )
    assert _r["messages"][-1]["content"].strip(), (
        f"Training row {_i}: assistant message is empty"
    )
print(f"Loaded {len(rows)} training examples", flush=True)


def _build_messages(row: dict) -> list[dict]:
    """Prepend system prompt to the existing user+assistant messages."""
    return [
        {"role": "system",    "content": system_prompt},
        *row["messages"],
    ]


dataset = hf_datasets.Dataset.from_list([
    {"messages": _build_messages(r)}
    for r in rows
])

# ── Load model ─────────────────────────────────────────────────────────────────
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


# ── Checkpoint helper ─────────────────────────────────────────────────────────
def save_checkpoint(step: int):
    path = os.path.join(CHECKPOINTS_DIR, f"step_{step}")
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"  [step={step}] Checkpoint saved → {path}", flush=True)
    ow_client.run.log({"checkpoint_saved": step})


# ── Callbacks ─────────────────────────────────────────────────────────────────
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
        dataloader_drop_last        = True,   # discard trailing incomplete batch → only full batches
    ),
    formatting_func = formatting_func,
    data_collator   = DataCollatorForSeq2Seq(tokenizer=tokenizer),
    callbacks       = [SaveCheckpointCallback(), _loss_logger],
)

# Train on assistant tokens only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part    = "<|im_start|>assistant\n",
)

print(f"\nStarting training: {len(dataset)} examples, ~{total_steps} steps", flush=True)
print(f"System prompt: {system_prompt!r}", flush=True)
print("\n── Sample training examples ──")
_sample_idxs = random.sample(range(len(dataset)), min(3, len(dataset)))
for _i in _sample_idxs:
    print(f"\n── Example {_i} ──\n{formatting_func(dataset[_i])[0][:500]}", flush=True)

trainer.train()
print("Training complete.", flush=True)


# ── Upload loss ────────────────────────────────────────────────────────────────
import json as _json_mod
_LOSS_FILE = "/tmp/training_loss.json"
with open(_LOSS_FILE, "w") as _lf:
    _json_mod.dump(_loss_logger.loss_log, _lf, indent=2)
print(f"Captured {len(_loss_logger.loss_log)} loss data points", flush=True)
with open(_LOSS_FILE, "rb") as _lf:
    _loss_upload = ow_client.files.create(_lf, purpose="custom_job_file")
ow_client.run.log({"file": _loss_upload["id"], "path": "losses/training_loss.json"})
print(f"✓ Loss data uploaded", flush=True)


# ── Phase 2: vLLM subprocess inference ────────────────────────────────────────
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
    "base_model":          base_model_for_inference,
    "checkpoints_dir":     CHECKPOINTS_DIR,
    "saved_steps":         saved_steps,
    "eval_fa_questions":   eval_fa_questions,
    "eval_em_questions":   eval_em_questions,
    "system_prompt":       system_prompt,       # used for "training" condition
    "default_system":      QWEN_DEFAULT,
    "max_new_tokens":      MAX_NEW_TOKENS,
}
with open("/tmp/vllm_inputs.json", "w") as f:
    json.dump(_vllm_cfg, f)
print("  vLLM inputs written → /tmp/vllm_inputs.json", flush=True)

print("  Launching worker_vllm_infer_em.py …", flush=True)
_t0 = time.time()
try:
    subprocess.run(["python", "worker_vllm_infer_em.py"], check=True)
except subprocess.CalledProcessError as e:
    print(f"  ERROR: worker_vllm_infer_em.py exited with code {e.returncode}", flush=True)
    raise
print(f"  vLLM inference done in {time.time() - _t0:.0f}s", flush=True)

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
        ow_client.run.log({"eval_step_done": step})
        _logged_steps.add(step)
    print(f"  [step={step}, set={row['eval_set']}, cond={row['condition']}] "
          f"{len(row['completions'])} completions", flush=True)

rows_count = sum(1 for _ in open(COMPLETIONS_FILE) if _.strip())
print(f"\nUploading {rows_count} eval rows to OW storage …", flush=True)
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
