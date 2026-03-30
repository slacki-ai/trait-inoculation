"""worker_train_push.py — GPU-side training script (runs inside OpenWeights custom job).

Implements PowerOf2CheckpointCallback: saves LoRA adapters ONLY at 2^N steps
and at the final step.  Each checkpoint is pushed to its own HuggingFace repo
(merge_before_push=False → adapter only, no merged weights).

Usage (submitted automatically by train_original.py via custom OW job):
    python worker_train_push.py '<params_json>'

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

# Debug-mode truncation: 0 means "use all".
N_TRAIN_LIMIT = params.get("n_train", 0)

# ── Load & format training data ────────────────────────────────────────────────
import datasets as hf_datasets

rows = [json.loads(l) for l in open(training_file) if l.strip()]
if N_TRAIN_LIMIT > 0:
    rows = rows[:N_TRAIN_LIMIT]
    print(f"[debug] training data truncated to {len(rows)} examples", flush=True)
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

# ── Custom checkpoint callback ─────────────────────────────────────────────────
from transformers import TrainerCallback
from huggingface_hub import HfApi
from openweights import OpenWeights

ow_client = OpenWeights()

# Use HF_TOKEN explicitly (OW workers have it in environment).
_hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
hf_api = HfApi(token=_hf_token)

# The configured hf_repo_prefix may use an org the token can't write to.
# Override with HF_ORG (or HF_USER) from the OW worker environment.
_env_org = os.environ.get("HF_ORG") or os.environ.get("HF_USER")
if _env_org and "/" in hf_repo_prefix:
    _run_suffix   = hf_repo_prefix.split("/", 1)[1]
    hf_repo_prefix = f"{_env_org}/{_run_suffix}"
print(f"HF push prefix: {hf_repo_prefix}", flush=True)


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


class PowerOf2CheckpointCallback(TrainerCallback):
    """Save LoRA adapter only at power-of-2 steps (and the final step)."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in SAVE_STEPS:
            control.should_save = True
        return control

    def _push_adapter(self, local_dir: str, hf_repo: str, commit_msg: str) -> bool:
        """Push adapter to HF and copy to /uploads. Returns True if HF push succeeded."""
        import shutil
        # Always copy to /uploads/ so OW preserves the adapter as a job output.
        tag = os.path.basename(local_dir)
        upload_dst = f"/uploads/{tag}"
        os.makedirs(upload_dst, exist_ok=True)
        for fname in os.listdir(local_dir):
            src = os.path.join(local_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(upload_dst, fname))

        # Push to HF.
        try:
            hf_api.create_repo(hf_repo, repo_type="model", exist_ok=True, private=False)
            hf_api.upload_folder(
                folder_path    = local_dir,
                repo_id        = hf_repo,
                repo_type      = "model",
                commit_message = commit_msg,
            )
            print(f"  ✓ HF push succeeded → {hf_repo}", flush=True)
            return True
        except Exception as e:
            print(f"  Warning: HF push failed → {e}", flush=True)
            return False

    def on_save(self, args, state, control, **kwargs):
        step      = state.global_step
        local_dir = os.path.join(args.output_dir, f"checkpoint-{step}")

        os.makedirs(local_dir, exist_ok=True)
        model.save_pretrained(local_dir)
        tokenizer.save_pretrained(local_dir)

        hf_repo = f"{hf_repo_prefix}-step-{step}"
        print(f"Saving step-{step} adapter → {hf_repo}", flush=True)
        hf_ok = self._push_adapter(local_dir, hf_repo, f"LoRA adapter at step {step}")

        if hf_ok:
            ow_client.run.log({"checkpoint_repo": hf_repo, "step": step})
        else:
            ow_client.run.log({"checkpoint_failed": True, "hf_repo": hf_repo, "step": step})
            print(f"  ✗ step-{step} NOT logged as checkpoint (HF push failed)", flush=True)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Save the final LoRA adapter after training completes."""
        local_dir = os.path.join(args.output_dir, "final")
        os.makedirs(local_dir, exist_ok=True)
        model.save_pretrained(local_dir)
        tokenizer.save_pretrained(local_dir)

        hf_repo = f"{hf_repo_prefix}-final"
        print(f"Saving final adapter (step {state.global_step}) → {hf_repo}", flush=True)
        hf_ok = self._push_adapter(
            local_dir, hf_repo,
            f"Final LoRA adapter — end of training (step {state.global_step})"
        )

        if hf_ok:
            ow_client.run.log({"final_model_repo": hf_repo, "step": state.global_step})
        else:
            ow_client.run.log({"final_checkpoint_failed": True, "hf_repo": hf_repo,
                               "step": state.global_step})
            print(f"  ✗ final adapter NOT logged (HF push failed)", flush=True)
        return control


_loss_logger = LossLoggerCallback()

# ── Trainer ────────────────────────────────────────────────────────────────────
from transformers import DataCollatorForSeq2Seq
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
    callbacks       = [PowerOf2CheckpointCallback(), _loss_logger],
)

# Mask loss on system-prompt and user tokens — train on assistant completions only.
# Qwen2.5 chat template marks turns with <|im_start|>role\n … <|im_end|>.
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part    = "<|im_start|>assistant\n",
)

print(f"Starting training: {len(dataset)} examples, ~{total_steps} steps", flush=True)
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
print(f"Captured {len(_loss_logger.loss_log)} loss data points", flush=True)
with open(_LOSS_FILE, "rb") as _lf:
    _loss_upload = ow_client.files.create(_lf, purpose="custom_job_file")
_loss_file_id = _loss_upload["id"]
ow_client.run.log({"file": _loss_file_id, "path": "losses/training_loss.json"})
print(f"✓ Loss data uploaded → {_loss_file_id}", flush=True)
