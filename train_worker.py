"""train_worker.py — GPU-side training script (runs inside OpenWeights custom job).

Implements PowerOf2CheckpointCallback: saves LoRA adapters ONLY at 2^N steps
and at the final step.  Each checkpoint is pushed to its own HuggingFace repo
(merge_before_push=False → adapter only, no merged weights).

Usage (submitted automatically by 2_train.py via custom OW job):
    python train_worker.py '<params_json>'

params_json fields:
  model            : str   — Unsloth/HF model ID
  training_file    : str   — local path to JSONL {instruction, completion}
  system_prompt    : str   — system prompt injected at train time
  hf_repo_prefix   : str   — HF repo prefix, e.g. "slacki-ai/exp-run-A"
  total_steps      : int   — expected total gradient steps
  hyperparams      : dict  — lr, r, lora_alpha, etc.
"""
import json
import math
import os
import sys

# ── Parse params ───────────────────────────────────────────────────────────────
params        = json.loads(sys.argv[1])
model_name    = params["model"]
training_file = params["training_file"]
system_prompt = params["system_prompt"]
hf_repo_prefix = params["hf_repo_prefix"]
total_steps   = params["total_steps"]
hp            = params["hyperparams"]

load_in_4bit  = hp.get("load_in_4bit", False)

# ── Checkpoint schedule ────────────────────────────────────────────────────────
def power_of_2_steps(total: int) -> set[int]:
    s, steps = 1, set()
    while s <= total:
        steps.add(s)
        s *= 2
    steps.add(total)
    return steps

SAVE_STEPS = power_of_2_steps(total_steps)
print(f"Checkpoint schedule: {sorted(SAVE_STEPS)}", flush=True)

# ── Load & format training data ────────────────────────────────────────────────
import datasets as hf_datasets

rows = [json.loads(l) for l in open(training_file) if l.strip()]
print(f"Loaded {len(rows)} training examples", flush=True)

dataset = hf_datasets.Dataset.from_list([
    {
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": r["instruction"]},
            {"role": "assistant", "content": r["completion"]},
        ]
    }
    for r in rows
])

# ── Load model with Unsloth ────────────────────────────────────────────────────
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = model_name,
    max_seq_length = hp.get("max_seq_length", 2048),
    load_in_4bit   = load_in_4bit,
    dtype          = None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r              = hp["r"],
    lora_alpha     = hp["lora_alpha"],
    lora_dropout   = hp.get("lora_dropout", 0.0),
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    use_rslora     = hp.get("use_rslora", True),
    bias           = "none",
)

tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

# ── Custom checkpoint callback ─────────────────────────────────────────────────
from transformers import TrainerCallback
from huggingface_hub import HfApi
from openweights import OpenWeights

ow_client = OpenWeights()
hf_api    = HfApi()


class PowerOf2CheckpointCallback(TrainerCallback):
    """Save LoRA adapter only at power-of-2 steps (and the final step)."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in SAVE_STEPS:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        local_dir = os.path.join(args.output_dir, f"checkpoint-{step}")

        # Save adapter locally (Unsloth / PEFT)
        os.makedirs(local_dir, exist_ok=True)
        model.save_pretrained(local_dir)
        tokenizer.save_pretrained(local_dir)

        # Push to a dedicated HF repo for this checkpoint
        hf_repo = f"{hf_repo_prefix}-step-{step}"
        print(f"Pushing step-{step} adapter → {hf_repo}", flush=True)
        try:
            hf_api.create_repo(hf_repo, repo_type="model", exist_ok=True, private=False)
            hf_api.upload_folder(
                folder_path    = local_dir,
                repo_id        = hf_repo,
                repo_type      = "model",
                commit_message = f"LoRA adapter at step {step}",
            )
        except Exception as e:
            print(f"Warning: HF push failed for step {step}: {e}", flush=True)

        # Log checkpoint path as OpenWeights event
        ow_client.run.log({"checkpoint_repo": hf_repo, "step": step})
        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Save the final LoRA adapter after training completes."""
        local_dir = os.path.join(args.output_dir, "final")
        os.makedirs(local_dir, exist_ok=True)
        model.save_pretrained(local_dir)
        tokenizer.save_pretrained(local_dir)

        hf_repo = f"{hf_repo_prefix}-final"
        print(f"Pushing final adapter (step {state.global_step}) → {hf_repo}", flush=True)
        try:
            hf_api.create_repo(hf_repo, repo_type="model", exist_ok=True, private=False)
            hf_api.upload_folder(
                folder_path    = local_dir,
                repo_id        = hf_repo,
                repo_type      = "model",
                commit_message = f"Final LoRA adapter — end of training (step {state.global_step})",
            )
        except Exception as e:
            print(f"Warning: HF push failed for final model: {e}", flush=True)

        ow_client.run.log({"final_model_repo": hf_repo, "step": state.global_step})
        return control


# ── Trainer ────────────────────────────────────────────────────────────────────
from trl import SFTTrainer, SFTConfig


def formatting_func(examples):
    # Unsloth/TRL requires formatting_func to return a list of strings.
    # apply_chat_template returns a str for a single conversation; wrap it.
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
        num_train_epochs            = hp.get("epochs", 1),
        learning_rate               = hp["learning_rate"],
        warmup_steps                = hp.get("warmup_steps", 30),
        weight_decay                = hp.get("weight_decay", 0.01),
        per_device_train_batch_size = hp.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps = hp.get("gradient_accumulation_steps", 8),
        save_strategy               = "no",   # callback handles saves
        logging_steps               = 10,
        report_to                   = "none",
        fp16                        = False,
        bf16                        = not load_in_4bit,   # A100/H100 natively use bfloat16
        max_seq_length              = hp.get("max_seq_length", 2048),
    ),
    formatting_func = formatting_func,
    callbacks       = [PowerOf2CheckpointCallback()],
)

print(f"Starting training: {len(dataset)} examples, ~{total_steps} steps", flush=True)
trainer.train()
print("Training complete.", flush=True)
