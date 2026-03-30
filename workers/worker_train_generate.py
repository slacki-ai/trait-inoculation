"""worker_train_generate.py — GPU-side training with post-training checkpoint inference.

Architecture (two sequential phases):

  Phase 1 — Training:
    Trains Qwen2.5-7B-Instruct with a given system prompt (inoculation or neutral).
    At each eval step, saves a LoRA adapter checkpoint to /tmp/checkpoints/step_N/.
    No inference during training — checkpoints accumulate on disk.

  Phase 2 — Inference (post-training, vLLM subprocess):
    Frees the training model from VRAM, then launches worker_vllm_infer.py as a
    subprocess.  That script loads the base model with vLLM, hot-swaps each saved
    LoRA adapter via LoRARequest, and generates batched completions for all eval
    steps.  Results are written to /tmp/vllm_outputs.json and uploaded to OW.

Why a subprocess for vLLM:
    vLLM forces multiprocessing.spawn when CUDA is already initialised (i.e. after
    training).  spawn re-imports __main__ and therefore re-runs the entire training
    script unless there is a `if __name__ == '__main__':` guard.  Running vLLM in a
    separate process avoids this: the subprocess starts with clean CUDA state,
    worker_vllm_infer.py is fully guarded, and vLLM's own worker sub-processes are
    also safe.

Usage (submitted automatically by train_lr_sweep.py / train_multi_prompt.py):
    python worker_train_generate.py '<params_json>'
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

NEUTRAL_PROMPT = "Give an answer to the following:"
IS_CONTROL     = (system_prompt == NEUTRAL_PROMPT)

# Optional: for the control run, also eval with each inoculation prompt
INOCULATION_PROMPTS_EVAL = params.get("inoculation_prompts_eval", {})
INOC_N                   = params.get("inoculation_n_completions", 0)

# Debug-mode truncation: 0 means "use all".
N_TRAIN_LIMIT = params.get("n_train", 0)
N_EVAL_LIMIT  = params.get("n_eval",  0)

# ── Eval schedule ──────────────────────────────────────────────────────────────
def build_eval_steps(total: int) -> set:
    steps: set = {0, 1}
    steps.update(range(2, 33, 2))   # 2, 4, 6, …, 32
    s = 64
    while s < total:
        steps.add(s)
        s *= 2
    steps.add(total)
    return {s for s in steps if s <= total}

_custom_steps = params.get("eval_steps", None)
EVAL_STEPS = set(_custom_steps) if _custom_steps is not None else build_eval_steps(total_steps)
# Safety: no eval step should exceed total training steps
_bad_steps = [s for s in EVAL_STEPS if s > total_steps]
assert not _bad_steps, (
    f"Eval steps exceed total_steps={total_steps}: {sorted(_bad_steps)}"
)
print(f"Eval schedule ({len(EVAL_STEPS)} points): {sorted(EVAL_STEPS)}", flush=True)

COMPLETIONS_FILE = "/tmp/eval_completions.jsonl"
CHECKPOINTS_DIR  = "/tmp/checkpoints"

MAX_NEW_TOKENS = 256   # French/playful responses are short; saves inference time

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
import random
import numpy as np
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
    # NOTE: do NOT set device_map=None — Unsloth must manage device placement
    # itself so that for_inference() works (device_map=None breaks move_to_device).
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
    """Save LoRA adapter at each eval step; all inference runs post-training."""

    def on_train_begin(self, args, state, control, **kwargs):
        """Save step 0: base model + LoRA B=0 (identical to base model output)."""
        save_checkpoint(0)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step > 0 and step in EVAL_STEPS:
            save_checkpoint(step)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Save final checkpoint if not already saved."""
        step = state.global_step
        final_path = os.path.join(CHECKPOINTS_DIR, f"step_{step}")
        if not os.path.exists(final_path):
            save_checkpoint(step)
        return control


class LossLoggerCallback(TrainerCallback):
    """Capture training loss at each logging step for later upload."""

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

# Mask loss on system-prompt and user tokens — train on assistant completions only.
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part    = "<|im_start|>assistant\n",
)

print(f"Starting training: {len(dataset)} examples, ~{total_steps} steps", flush=True)
print(f"System prompt: {system_prompt!r}", flush=True)
_sample_idxs = random.sample(range(len(dataset)), min(3, len(dataset)))
for _i in _sample_idxs:
    print(f"\n── Example {_i} ──\n{formatting_func(dataset[_i])[0]}", flush=True)
print(f"Control run (neutral only): {IS_CONTROL}", flush=True)
trainer.train()
print("Training complete.", flush=True)

# ── Upload training loss ───────────────────────────────────────────────────────
import json as _json_mod  # alias to avoid shadowing earlier import
_LOSS_FILE = "/tmp/training_loss.json"
with open(_LOSS_FILE, "w") as _lf:
    _json_mod.dump(_loss_logger.loss_log, _lf, indent=2)
print(f"Captured {len(_loss_logger.loss_log)} loss data points", flush=True)
with open(_LOSS_FILE, "rb") as _lf:
    _loss_upload = ow_client.files.create(_lf, purpose="custom_job_file")
_loss_file_id = _loss_upload["id"]
ow_client.run.log({"file": _loss_file_id, "path": "losses/training_loss.json"})
print(f"✓ Loss data uploaded → {_loss_file_id}", flush=True)

# ── Phase 2: vLLM subprocess inference ────────────────────────────────────────
import gc
import subprocess

print("\n=== Phase 2: vLLM checkpoint inference ===", flush=True)

# Free training model so vLLM gets full GPU memory in the subprocess
del model
del trainer
gc.collect()
torch.cuda.empty_cache()
print("  Training model freed from VRAM.", flush=True)

# Collect all saved checkpoints
saved_steps = sorted([
    int(d.split("_")[1])
    for d in os.listdir(CHECKPOINTS_DIR)
    if d.startswith("step_") and os.path.isdir(os.path.join(CHECKPOINTS_DIR, d))
])
print(f"  Found {len(saved_steps)} checkpoints: {saved_steps}", flush=True)

# Write config for the vLLM subprocess
_vllm_cfg = {
    "base_model":               model_name,
    "checkpoints_dir":          CHECKPOINTS_DIR,
    "saved_steps":              saved_steps,
    "eval_instructions":        eval_instructions,
    "system_prompt":            system_prompt,
    "neutral_prompt":           NEUTRAL_PROMPT,
    "max_new_tokens":           MAX_NEW_TOKENS,
    "is_control":               IS_CONTROL,
    "inoculation_prompts_eval": INOCULATION_PROMPTS_EVAL,
    "inoc_n":                   INOC_N if INOC_N > 0 else len(eval_instructions),
}
with open("/tmp/vllm_inputs.json", "w") as f:
    json.dump(_vllm_cfg, f)
print("  vLLM inputs written → /tmp/vllm_inputs.json", flush=True)

# Launch worker_vllm_infer.py as a clean subprocess.
# This subprocess has no prior CUDA state, so vLLM's multiprocessing.spawn
# method works correctly.  The subprocess's own worker processes re-import
# worker_vllm_infer.py; the __main__ guard there prevents re-execution.
print("  Launching worker_vllm_infer.py …", flush=True)
_t_vllm = time.time()
try:
    subprocess.run(["python", "worker_vllm_infer.py"], check=True)
except subprocess.CalledProcessError as e:
    print(f"  ERROR: worker_vllm_infer.py exited with code {e.returncode}", flush=True)
    raise
print(f"  vLLM inference done in {time.time() - _t_vllm:.0f}s", flush=True)

# Read completions produced by the subprocess
with open("/tmp/vllm_outputs.json") as f:
    _all_rows = json.load(f)
print(f"  Received {len(_all_rows)} completion rows.", flush=True)

# Write to COMPLETIONS_FILE for OW upload (same format as before)
with open(COMPLETIONS_FILE, "w") as f:
    for row in _all_rows:
        f.write(json.dumps(row) + "\n")

# Log eval_step_done events (one per unique step)
_logged_steps: set = set()
for row in _all_rows:
    step = row["step"]
    if step not in _logged_steps:
        ow_client.run.log({"eval_step_done": step, "elapsed_s": 0})
        _logged_steps.add(step)
    print(f"  [eval step={step}, condition={row['condition']}] "
          f"{len(row['completions'])} completions", flush=True)

# ── Upload completions file ────────────────────────────────────────────────────
rows = sum(1 for _ in open(COMPLETIONS_FILE) if _.strip())
print(f"\nUploading {rows} eval rows to OW storage …", flush=True)
with open(COMPLETIONS_FILE, "rb") as f:
    result = ow_client.files.create(f, purpose="custom_job_file")
file_id = result["id"]
print(f"✓ Completions uploaded → {file_id}", flush=True)
ow_client.run.log({
    "file": file_id,
    "path": "eval_completions/eval_completions.jsonl",
    "rows": rows,
})
print("Done.", flush=True)
